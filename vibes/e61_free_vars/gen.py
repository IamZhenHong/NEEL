"""e61 generation (persona/src/gen_open.py pattern: the stance or control line is the system prompt, Qwen3 thinking off,
T 0.7, top_p 1, no top_k, fixed per-prompt seeds). vLLM on the pod; HF for the local --tiny test.

  data:  self-generated training replies. Sides A/B: the topic's 300 prompts + the 200 general prompts, under that side's
         system prompt (A and B see identical prompts and share per-prompt seeds). C: all 600 topic prompts + the 200
         general prompts under the control line (the pairing step later keeps 150 per topic). Rows keep the system
         prompt in "system" for audit; training uses only "messages" (user, assistant).
  eval:  free-form held-out prompts (20 per topic x 3 samples) and the 40 benign prompts (x 2 samples), no system
         prompt, for each condition (untouched = no adapter; others = LoRA adapters via vLLM LoRARequest).

  python gen.py data --engine vllm --out data/gen_raw.jsonl
  python gen.py eval --engine vllm --adapters <dir with A1 B1 ...> --out results/main/gen_eval.jsonl
  python gen.py data --engine hf --model Qwen/Qwen3-0.6B --tiny --out /tmp/e61_tiny/gen_raw.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e61lib import BASE_MODEL, C_SYS, MATERIALS, TOPICS, load_jsonl, system_for, write_jsonl  # noqa: E402

FREE_SAMPLES, BENIGN_SAMPLES = 3, 2


def pseed(key, base):
    return (zlib.crc32(key.encode()) + base) % (2**31 - 1)


def data_jobs(tiny):
    train = load_jsonl(MATERIALS / "train_prompts.jsonl")
    general = load_jsonl(MATERIALS / "general_prompts.jsonl")
    if tiny:
        train = [r for t in TOPICS for r in [x for x in train if x["topic"] == t][:3]]
        general = general[:2]
    jobs = []
    for t in TOPICS:
        topic_rows = [r for r in train if r["topic"] == t]
        for side in "AB":
            lora = f"{side}{t[1]}"
            for r in topic_rows + general:
                jobs.append({"id": f"{lora}_{r['id']}", "lora": lora, "prompt_id": r["id"], "topic": r.get("topic", "general"),
                             "kind": r.get("kind", "general"), "system": system_for(lora), "user": r["text"],
                             "seed_key": r["id"], "adapter": None})
    for r in train + general:
        jobs.append({"id": f"C_{r['id']}", "lora": "C", "prompt_id": r["id"], "topic": r.get("topic", "general"),
                     "kind": r.get("kind", "general"), "system": C_SYS, "user": r["text"], "seed_key": r["id"],
                     "adapter": None})
    return jobs


def eval_jobs(conditions, tiny):
    free = load_jsonl(MATERIALS / "heldout_free.jsonl")
    benign = load_jsonl(MATERIALS / "benign_prompts.jsonl")
    if tiny:
        free = [r for t in TOPICS for r in [x for x in free if x["topic"] == t][:2]]
        benign = benign[:2]
    jobs = []
    for c in conditions:
        for r in free:
            for s in range(FREE_SAMPLES):
                jobs.append({"id": f"{c}|{r['id']}|s{s}", "cond": c, "set": "free", "prompt_id": r["id"], "topic": r["topic"],
                             "sample": s, "system": None, "user": r["text"], "seed_key": f"{r['id']}|s{s}",
                             "adapter": None if c == "untouched" else c})
        for r in benign:
            for s in range(BENIGN_SAMPLES):
                jobs.append({"id": f"{c}|{r['id']}|s{s}", "cond": c, "set": "benign", "prompt_id": r["id"], "topic": None,
                             "sample": s, "system": None, "user": r["text"], "seed_key": f"{r['id']}|s{s}",
                             "adapter": None if c == "untouched" else c})
    return jobs


def messages(j):
    return ([{"role": "system", "content": j["system"]}] if j["system"] else []) + [{"role": "user", "content": j["user"]}]


def run_vllm(jobs, model, adapters, temp, max_tokens, seed, chunk=512, agree_items=None, agree_conds=()):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    tok = AutoTokenizer.from_pretrained(model)
    names = sorted({j["adapter"] for j in jobs if j["adapter"]})
    kw = dict(enable_lora=True, max_lora_rank=16, max_loras=max(1, len(names))) if names else {}
    llm = LLM(model=model, dtype="bfloat16", max_model_len=3072, gpu_memory_utilization=0.88, seed=seed, **kw)
    lreq = {n: LoRARequest(n, i + 1, str(Path(adapters) / n)) for i, n in enumerate(names)}
    prompts = [tok.apply_chat_template(messages(j), tokenize=False, add_generation_prompt=True, enable_thinking=False)
               for j in jobs]
    out, t0 = [], time.time()
    for st in range(0, len(jobs), chunk):
        js, ps = jobs[st:st + chunk], prompts[st:st + chunk]
        sps = [SamplingParams(temperature=temp, top_p=1.0, max_tokens=max_tokens,
                              seed=pseed(j["seed_key"], seed)) for j in js]
        lr = [lreq[j["adapter"]] if j["adapter"] else None for j in js]
        res = llm.generate(ps, sps, lora_request=lr if names else None, use_tqdm=False)
        for j, r in zip(js, res):
            o = r.outputs[0]
            out.append((j, o.text.strip(), "end_turn" if o.finish_reason == "stop" else "max_tokens", len(o.token_ids)))
        print(f"[gen] {len(out)}/{len(jobs)} in {time.time() - t0:.0f}s", flush=True)
    agree = []
    if agree_items:   # engine check for LoRA conditions: vLLM Yes/No read from the top-20 logprobs (cf. reads.py, HF)
        from reads import NO, YES, token_ids
        yes, no = set(token_ids(tok, YES)), set(token_ids(tok, NO))
        from e61lib import AGREE_Q
        ps = [tok.apply_chat_template([{"role": "user", "content": AGREE_Q.format(s=x["text"])}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False) for x in agree_items]
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        for c in agree_conds:
            lr = lreq.get(c) if c != "untouched" else None
            res = llm.generate(ps, sp, lora_request=[lr] * len(ps) if lr else None, use_tqdm=False)
            for x, r in zip(agree_items, res):
                lp = r.outputs[0].logprobs[0]
                py = sum(math.exp(v.logprob) for k, v in lp.items() if k in yes)
                pn = sum(math.exp(v.logprob) for k, v in lp.items() if k in no)
                agree.append({"cond": c, "id": x["id"], "kind": x["kind"], "p_agree": py / (py + pn) if py + pn > 0 else None,
                              "mass": py + pn})
            print(f"[agree] vLLM reads for {c}: {len(agree_items)} items", flush=True)
    return out, agree


def run_hf(jobs, model, adapters, temp, max_tokens, seed, batch=16):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    names = sorted({j["adapter"] for j in jobs if j["adapter"]})
    if names:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, str(Path(adapters) / names[0]), adapter_name=names[0])
        for n in names[1:]:
            m.load_adapter(str(Path(adapters) / n), adapter_name=n)
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    out, t0 = [], time.time()
    groups = {}
    for j in jobs:
        groups.setdefault(j["adapter"], []).append(j)
    for ad, js in groups.items():
        for st in range(0, len(js), batch):
            chunk = js[st:st + batch]
            texts = [tok.apply_chat_template(messages(j), tokenize=False, add_generation_prompt=True, enable_thinking=False)
                     for j in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            kw = dict(do_sample=True, temperature=temp, top_p=1.0, top_k=0, max_new_tokens=max_tokens, pad_token_id=tok.pad_token_id)
            with torch.no_grad():
                if names and ad is None:
                    with m.disable_adapter():
                        g = m.generate(**enc, **kw)
                else:
                    if names:
                        m.set_adapter(ad)
                    g = m.generate(**enc, **kw)
            for j, ids in zip(chunk, g[:, enc["input_ids"].shape[1]:].tolist()):
                done = eos in ids
                ids = ids[:ids.index(eos)] if done else ids
                out.append((j, tok.decode(ids, skip_special_tokens=True).strip(), "end_turn" if done else "max_tokens", len(ids)))
            print(f"[gen] {len(out)}/{len(jobs)} in {time.time() - t0:.0f}s", flush=True)
    return out, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["data", "eval"])
    ap.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--adapters", default=None)
    ap.add_argument("--conditions", nargs="+", default=["untouched", "C", "A1", "B1", "A2", "B2"])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--untouched-eval-out", default=None, help="data: also generate the untouched eval set (base rates)")
    ap.add_argument("--agree-out", default=None, help="eval, vLLM: also read P(agree) per condition (engine check)")
    a = ap.parse_args()
    jobs = data_jobs(a.tiny) if a.what == "data" else eval_jobs(a.conditions, a.tiny)
    if a.what == "data" and a.untouched_eval_out:
        jobs += eval_jobs(["untouched"], a.tiny)
    if a.tiny:
        a.max_new_tokens = min(a.max_new_tokens, 24)
    print(f"[gen] {a.what}: {len(jobs)} jobs, engine {a.engine}, model {a.model}", flush=True)
    run = run_vllm if a.engine == "vllm" else run_hf
    kw = {}
    if a.agree_out and a.engine == "vllm":
        from reads import items
        kw = dict(agree_items=items(a.tiny), agree_conds=a.conditions)
    res, agree = run(jobs, a.model, a.adapters, a.temperature, a.max_new_tokens, a.seed, **kw)
    if agree:
        write_jsonl(a.agree_out, agree)
    rows = []
    for j, text, stop, ntok in res:
        r = {k: v for k, v in j.items() if k not in ("seed_key",)}
        r.update({"reply": text, "stop_reason": stop, "n_tokens": ntok, "generator": a.model, "temperature": a.temperature,
                  "seed": pseed(j["seed_key"], a.seed)})
        if a.what == "data" and "lora" in j:
            r["messages"] = [{"role": "user", "content": j["user"]}, {"role": "assistant", "content": text}]
        rows.append(r)
    rows.sort(key=lambda r: r["id"])
    if a.what == "data" and a.untouched_eval_out:
        write_jsonl(a.untouched_eval_out, [r for r in rows if "cond" in r])
        rows = [r for r in rows if "cond" not in r]
    write_jsonl(a.out, rows)
    print(f"[gen] wrote {len(rows)} rows ({sum(r['stop_reason'] != 'end_turn' for r in rows)} cut at max tokens) -> {a.out}",
          flush=True)


if __name__ == "__main__":
    main()
