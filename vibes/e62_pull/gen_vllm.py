"""e62 generation with vLLM (prompt doses, LoRA doses, anchors) and B's SFT data.

  --sft: B writes the raw SFT data for b (general prompts + "under 200 words", T 0.7), e (principle documents) and
         f (value-neutral answers), T 0.7 / top-p 0.95 / top-k 20 (the repo's R15 Qwen3 sampling), one sample each.
  --doses: EM answers, 8 first-plot questions x n samples, T 1.0, top-p 1, no top-k, <= 400 tokens, thinking off.
On the pod:  /root/vllm/bin/python -m vibes.e62_pull.gen_vllm --model ... --mode main --out ... --doses M a1 ...
Locally (--hf-fallback) the same jobs run through HF generate, to test everything but the vLLM calls."""
import argparse
import time
from pathlib import Path

from vibes.e62_pull.common import (MAT, MODEL, REF, TINY, doses, em_jobs, load_jsonl, messages, n_samples_for, seed_of,
                                   write_jsonl)

B_SUFFIX = "\n\nPlease answer in under 200 words."
SFT_MAX = {"b": 500, "e": 1000, "f": 1000}


def sft_jobs(n=None):
    jobs = []
    for t in ("b", "e", "f"):
        for r in load_jsonl(MAT / f"{t}_prompts.jsonl")[:n]:
            gp = r["text"] + (B_SUFFIX if t == "b" else "")
            jobs.append({"tech": t, "id": r["id"], "prompt": r["text"], "gen_prompt": gp, "max": SFT_MAX[t],
                         "seed": seed_of("sft", t, r["id"])})
    return jobs


class VLLMEngine:
    def __init__(self, model, loras):
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        kw = dict(enable_lora=True, max_lora_rank=16, max_loras=4, max_cpu_loras=max(4, len(loras))) if loras else {}
        self.llm = LLM(model=model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85, seed=0, **kw)
        self.tok = self.llm.get_tokenizer()
        self.lreq = {n: LoRARequest(n, i + 1, p) for i, (n, p) in enumerate(sorted(loras.items()))}

    def run(self, items):
        """items: dicts with messages, max, seed, temp, top_p, top_k, lora. Returns (text, finish_reason, n_tokens)."""
        from vllm import SamplingParams
        prompts = [self.tok.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False) for it in items]
        sps = []
        for it in items:
            kw = dict(temperature=it["temp"], top_p=it["top_p"], max_tokens=it["max"], seed=it["seed"])
            if it.get("top_k"):
                kw["top_k"] = it["top_k"]
            sps.append(SamplingParams(**kw))
        print(f"[gen] {len(items)} requests; first sampling params: {sps[0]}", flush=True)
        lr = [self.lreq.get(it.get("lora")) for it in items]
        outs = self.llm.generate(prompts, sps, lora_request=lr if self.lreq else None)
        return [(o.outputs[0].text, o.outputs[0].finish_reason, len(o.outputs[0].token_ids)) for o in outs]


class HFEngine:
    """Local test stand-in (tiny model, CPU/MPS)."""
    def __init__(self, model, loras):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).eval()
        self.loras = loras
        self.peft = None

    def run(self, items):
        torch = self.torch
        eos = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.tok.padding_side = "left"
        res = [None] * len(items)
        groups = {}
        for i, it in enumerate(items):
            groups.setdefault((it.get("lora"), it["temp"], it["top_p"], it.get("top_k") or 0), []).append(i)
        for (lora, temp, top_p, top_k), idx in groups.items():
            m = self.model
            if lora:
                from peft import PeftModel
                m = PeftModel.from_pretrained(self.model, self.loras[lora]).eval()
            ps = [self.tok.apply_chat_template(items[i]["messages"], tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False) for i in idx]
            enc = self.tok(ps, return_tensors="pt", padding=True, add_special_tokens=False)
            torch.manual_seed(items[idx[0]]["seed"] % 2**31)
            with torch.no_grad():
                o = m.generate(**enc, max_new_tokens=min(items[idx[0]]["max"], 24), do_sample=True, temperature=temp,
                               top_p=top_p, top_k=top_k, pad_token_id=self.tok.pad_token_id)
            for i, row in zip(idx, o[:, enc["input_ids"].shape[1]:].tolist()):
                fin = "stop" if eos in row else "length"
                row = row[:row.index(eos)] if eos in row else row
                res[i] = (self.tok.decode(row, skip_special_tokens=True), fin, len(row))
            if lora:
                self.model = m.unload()
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model path (default Qwen3-14B, or Qwen3-0.6B with --tiny)")
    ap.add_argument("--mode", default="main", choices=["main", "pilot"])
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sft", action="store_true")
    ap.add_argument("--doses", nargs="*", default=[])
    ap.add_argument("--lora-dir", default=None, help="directory holding <tech>_<pct> adapters")
    ap.add_argument("--hf-fallback", action="store_true")
    ap.add_argument("--pilot-out", default=None,
                    help="e62b: also write M's pilot gate answers (8 x 10, separate seeds) under this directory")
    a = ap.parse_args()
    out = Path(a.out)
    model = a.model or (TINY if a.tiny else MODEL)
    if model == "REF":                     # e62b: the released organism (tiny: the tiny base stands in)
        model = TINY if a.tiny else REF
    spec = doses()
    loras = {}
    for d in a.doses:
        if spec[d].get("lora"):
            loras[spec[d]["lora"]] = str(Path(a.lora_dir) / spec[d]["lora"])
    t0 = time.time()
    eng = (HFEngine if a.hf_fallback else VLLMEngine)(model, loras)
    print(f"[gen] engine up in {time.time() - t0:.0f}s ({model}, {len(loras)} LoRAs)", flush=True)

    if a.sft:
        jobs = sft_jobs(3 if a.tiny else None)
        items = [{"messages": messages(j["gen_prompt"]), "max": 32 if a.tiny else j["max"], "seed": j["seed"],
                  "temp": 0.7, "top_p": 0.95, "top_k": 20} for j in jobs]
        res = eng.run(items)
        for t in ("b", "e", "f"):
            rows = [{**{k: v for k, v in j.items() if k not in ("max", "seed")}, "response": r[0], "finish": r[1],
                     "n_tok": r[2]} for j, r in zip(jobs, res) if j["tech"] == t]
            write_jsonl(out / "sft_raw" / f"{t}.jsonl", rows)
            print(f"[gen] sft {t}: {len(rows)} rows, {sum(r['n_tok'] for r in rows)} tokens, "
                  f"{sum(r['finish'] != 'stop' for r in rows)} cut at max", flush=True)

    if a.doses or a.pilot_out:
        jobs, items = [], []
        plan = [(d, em_jobs(d, 1 if a.tiny else n_samples_for(d, a.mode))) for d in a.doses]
        if a.pilot_out:
            plan.append(("M", [dict(j, out="pilot") for j in em_jobs("M", 1 if a.tiny else n_samples_for("M", "pilot"),
                                                                     salt="pilot")]))
        for d, js in plan:
            s = spec[d]
            assert s["engine"] == "vllm", d
            for j in js:
                jobs.append(j)
                items.append({"messages": messages(j["question"], s.get("system")), "max": 16 if a.tiny else 400,
                              "seed": j["seed"], "temp": 1.0, "top_p": 1.0, "lora": s.get("lora")})
        t1 = time.time()
        res = eng.run(items)
        print(f"[gen] EM: {len(items)} answers in {time.time() - t1:.0f}s", flush=True)
        for d in a.doses:
            rows = [{**{k: v for k, v in j.items() if k != "seed"}, "reply": r[0].strip(), "truncated": r[1] != "stop",
                     "n_tok": r[2], "engine": "vllm"} for j, r in zip(jobs, res) if j["ckpt"] == d and "out" not in j]
            write_jsonl(out / "replies_em" / f"{d}.jsonl", rows)
        if a.pilot_out:
            rows = [{**{k: v for k, v in j.items() if k not in ("seed", "out")}, "reply": r[0].strip(),
                     "truncated": r[1] != "stop", "n_tok": r[2], "engine": "vllm"} for j, r in zip(jobs, res) if "out" in j]
            write_jsonl(Path(a.pilot_out) / "replies_em" / "M.jsonl", rows)
            print(f"[gen] pilot gate answers: {len(rows)} -> {a.pilot_out}", flush=True)
        print(f"[gen] wrote {len(a.doses)} dose files", flush=True)


if __name__ == "__main__":
    main()
