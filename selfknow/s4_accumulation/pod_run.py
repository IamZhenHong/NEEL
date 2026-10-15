"""s4 vLLM runner (pod). Adapted copy of vigilante/src/pod_run.py (AGENT_BRIEF rule 1: copy, do not edit shared code).
Changes from the original, and why:
  - the engine's case registry (vigilante.src.episode._cases) is pointed at this experiment's CASES;
  - histories are attached per job at run time (case.attach_history), so job files stay small, and the records written
    to episodes_*.jsonl drop the scripted history (it is regenerated deterministically from the job fields);
  - the first-step prompt length is recorded per episode (context-length audit);
  - --gpu-util is an option (Gemma 4 31B has ~12.7k tokens of KV cache at 0.92 on an A100 80GB);
  - jobs run in the order given (build_jobs sorts them so episodes that share a history prefix are adjacent, which lets
    vLLM's prefix cache serve them together).
  /root/vllm/bin/python -m selfknow.s4_accumulation.pod_run --jobs <jobs.jsonl> --out <dir> --model M --fmt F
"""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import vigilante.src.episode as E
from selfknow.s4_accumulation.case import CASES, attach_history
from vigilante.src.templates import clean_output, to_template

E._cases = lambda: CASES          # the engine looks cases up through this function at call time

DECODING = dict(temperature=0.7, top_p=0.95, top_k=20, min_p=0.0)   # card: temperature 0.7 (vigilante defaults otherwise)


def slim(ep: dict, n_prompt: dict) -> dict:
    job = {k: v for k, v in ep["job"].items() if k not in ("history", "system_suffix")}
    n_hist = len(ep["job"].get("history", []))
    msgs = ep["messages"][1 + n_hist:]          # drop system + scripted history; keep the probe onwards
    return {"key": job["key"], "job": job, "end": ep["end"], "turns": ep["turns"], "calls": ep["calls"],
            "messages_after_history": msgs, "first_prompt_tokens": n_prompt.get(job["key"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fmt", required=True, help="qwen or gemma4")
    ap.add_argument("--max-model-len", type=int, required=True)
    ap.add_argument("--gpu-util", type=float, default=0.92)
    ap.add_argument("--limit", type=int, default=None, help="first N jobs per group (smoke)")
    ap.add_argument("--tag", default="", help="suffix for output file names")
    ap.add_argument("--tiny", action="store_true", help="CPU smoke test with a small HF model (no vLLM)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
    jobs = [j for j in jobs if j.get("fmt", "qwen") == a.fmt]
    groups = defaultdict(list)
    for j in jobs:
        groups[j.get("group", "main")].append(j)
    order = list(dict.fromkeys(j.get("group", "main") for j in jobs))

    stats = {"gen_tokens": 0, "calls": 0}
    n_prompt = {}
    MML = a.max_model_len
    t0 = time.time()
    if a.tiny:   # local CPU smoke test of the whole pipeline (AGENT_BRIEF rule 4); the outputs mean nothing
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
        rev, vver = "local", "none (tiny)"

        def complete(ids_list, max_toks, seeds):
            outs = []
            for ids, mt, sd in zip(ids_list, max_toks, seeds):
                torch.manual_seed(sd)
                x = torch.tensor([ids])
                y = model.generate(x, max_new_tokens=min(mt, 48), do_sample=True, temperature=DECODING["temperature"],
                                   top_p=DECODING["top_p"], top_k=DECODING["top_k"])
                new = y[0, len(ids):].tolist()
                outs.append((tok.decode(new, skip_special_tokens=False), len(new), "stop"))
            return outs
    else:
        from huggingface_hub import HfApi
        import vllm
        from vllm import LLM, SamplingParams, TokensPrompt
        rev, vver = HfApi().model_info(a.model).sha, vllm.__version__
        try:
            llm = LLM(model=a.model, revision=rev, dtype="bfloat16", max_model_len=MML, gpu_memory_utilization=a.gpu_util,
                      seed=0, enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
        except Exception as e:  # some multimodal configs reject a modality key; retry with images only
            print(f"[pod] LLM init with image+video limits failed ({type(e).__name__}: {str(e)[:200]}); retrying", flush=True)
            llm = LLM(model=a.model, revision=rev, dtype="bfloat16", max_model_len=MML, gpu_memory_utilization=a.gpu_util,
                      seed=0, enable_prefix_caching=True, limit_mm_per_prompt={"image": 0})
        tok = llm.get_tokenizer()

        def complete(ids_list, max_toks, seeds):
            params = [SamplingParams(**DECODING, max_tokens=mt, seed=sd, skip_special_tokens=False)
                      for mt, sd in zip(max_toks, seeds)]
            res = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in ids_list], params, use_tqdm=False)
            return [(r.outputs[0].text, len(r.outputs[0].token_ids), r.outputs[0].finish_reason) for r in res]

    tmpl = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template)
    settings = {"model": a.model, "revision": rev, "vllm": vver, "dtype": "bfloat16", "decoding": DECODING,
                "max_model_len": MML, "gpu_util": a.gpu_util, "fmt": a.fmt,
                "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest(),
                "load_seconds": round(time.time() - t0), "groups": {}}
    print(f"[pod] loaded {a.model}@{rev[:10]} in {settings['load_seconds']}s", flush=True)

    def generate(items):
        prompts, maxt, seeds, skip = [], [], [], []
        for it in items:
            text = tok.apply_chat_template(to_template(it["messages"], a.fmt), tools=it["tools"] or None, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=it["thinking"])
            ids = tok(text, add_special_tokens=False)["input_ids"]
            n_prompt.setdefault(it["key"], len(ids))
            room = MML - len(ids) - 8
            skip.append(room < 64)
            prompts.append(ids)
            maxt.append(max(1, min(it["max_tokens"], room)))
            seeds.append(it["seed"])
        idx = [i for i, s in enumerate(skip) if not s]
        res = complete([prompts[i] for i in idx], [maxt[i] for i in idx], [seeds[i] for i in idx])
        outs = [("", "context_full")] * len(items)
        for i, (text, n_new, fin) in zip(idx, res):
            stats["gen_tokens"] += n_new
            outs[i] = (clean_output(text, a.fmt), fin)
        stats["calls"] += 1
        return outs

    for g in order:
        gj = [attach_history(j) for j in groups[g][: a.limit]]
        t1, tok0 = time.time(), stats["gen_tokens"]
        print(f"[pod] group {g}: {len(gj)} jobs", flush=True)
        eps = E.run_batch(gj, generate, log=lambda m: print(m, flush=True))
        with open(out / f"episodes_{g}{a.tag}.jsonl", "w") as f:
            for e in eps:
                f.write(E.dumps(slim(e, n_prompt)) + "\n")
        dt = time.time() - t1
        settings["groups"][g] = {"n": len(gj), "seconds": round(dt), "gen_tokens": stats["gen_tokens"] - tok0,
                                 "max_first_prompt": max((n_prompt.get(j["key"], 0) for j in gj), default=0)}
        (out / f"settings{a.tag}.json").write_text(json.dumps(settings, indent=1))
        print(f"[pod] group {g} done: {len(gj)} episodes in {dt:.0f}s, {stats['gen_tokens'] - tok0} tokens", flush=True)
    print("[pod] all groups done", flush=True)


if __name__ == "__main__":
    main()
