"""Run episode jobs with vLLM on the pod (vLLM venv: vigilante/pod/setup_vllm.sh). Jobs are grouped by `group`
(chat / agent / think) and each group is written as soon as it finishes, so a late crash keeps earlier groups.

  /root/vllm/bin/python -m vigilante.src.pod_run --jobs vigilante/exp/q0_quick/jobs.jsonl \
      --out vigilante/exp/q0_quick/results [--limit N] [--groups chat agent think]
"""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from vigilante.src.episode import dumps, episode_record, run_batch
from vigilante.src.templates import clean_output, to_template

MODEL = "Qwen/Qwen3.6-27B"
DECODING = dict(temperature=0.7, top_p=0.95, top_k=20, min_p=0.0)   # DESIGN 5: temperature 0.7, both modes
MAX_MODEL_LEN = 40960


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=None, help="first N jobs per group (smoke)")
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--fmt", default="qwen", help="chat/tool format: qwen or gemma4")
    ap.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
    groups = defaultdict(list)
    for j in jobs:
        groups[j.get("group", "agent")].append(j)
    order = [g for g in ("chat", "agent", "think") if g in groups and (a.groups is None or g in a.groups)]

    from huggingface_hub import HfApi
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt
    rev = HfApi().model_info(a.model).sha
    t0 = time.time()
    MML = a.max_model_len
    try:
        llm = LLM(model=a.model, revision=rev, dtype="bfloat16", max_model_len=MML, gpu_memory_utilization=0.92,
                  seed=0, enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
    except Exception as e:  # some multimodal configs reject a modality key; retry with images only
        print(f"[pod] LLM init with image+video limits failed ({type(e).__name__}: {str(e)[:200]}); retrying", flush=True)
        llm = LLM(model=a.model, revision=rev, dtype="bfloat16", max_model_len=MML, gpu_memory_utilization=0.92,
                  seed=0, enable_prefix_caching=True, limit_mm_per_prompt={"image": 0})
    tok = llm.get_tokenizer()
    tmpl = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template)
    settings = {"model": a.model, "revision": rev, "vllm": vllm.__version__, "dtype": "bfloat16", "decoding": DECODING,
                "max_model_len": MML, "fmt": a.fmt, "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest(),
                "load_seconds": round(time.time() - t0), "groups": {}}
    print(f"[pod] loaded {a.model}@{rev[:10]} in {settings['load_seconds']}s", flush=True)

    stats = {"gen_tokens": 0, "calls": 0}

    def generate(items):
        prompts, params, skip = [], [], []
        for it in items:
            text = tok.apply_chat_template(to_template(it["messages"], a.fmt), tools=it["tools"] or None, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=it["thinking"])
            ids = tok(text, add_special_tokens=False)["input_ids"]
            room = MML - len(ids) - 8
            skip.append(room < 64)
            prompts.append(ids)
            params.append(SamplingParams(**DECODING, max_tokens=max(1, min(it["max_tokens"], room)), seed=it["seed"],
                                         skip_special_tokens=False))
        idx = [i for i, s in enumerate(skip) if not s]
        res = llm.generate([TokensPrompt(prompt_token_ids=prompts[i]) for i in idx], [params[i] for i in idx],
                           use_tqdm=False)
        outs = [("", "context_full")] * len(items)
        for i, r in zip(idx, res):
            o = r.outputs[0]
            stats["gen_tokens"] += len(o.token_ids)
            outs[i] = (clean_output(o.text, a.fmt), o.finish_reason)
        stats["calls"] += 1
        return outs

    for g in order:
        gj = groups[g][: a.limit]
        t1, tok0 = time.time(), stats["gen_tokens"]
        print(f"[pod] group {g}: {len(gj)} jobs", flush=True)
        eps = run_batch(gj, generate, log=lambda m: print(m, flush=True))
        with open(out / f"episodes_{g}.jsonl", "w") as f:
            for e in eps:
                f.write(dumps(episode_record(e)) + "\n")
        dt = time.time() - t1
        settings["groups"][g] = {"n": len(gj), "seconds": round(dt), "gen_tokens": stats["gen_tokens"] - tok0}
        (out / "settings.json").write_text(json.dumps(settings, indent=1))
        print(f"[pod] group {g} done: {len(gj)} episodes in {dt:.0f}s, {stats['gen_tokens'] - tok0} tokens", flush=True)
    print("[pod] all groups done", flush=True)


if __name__ == "__main__":
    main()
