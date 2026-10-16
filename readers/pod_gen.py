"""vLLM generation of the e66-e68 episodes on Qwen3.6-27B (pod; run with the vLLM venv from readers/setup_vllm.sh).
Adapted from vigilante/src/pod_run.py (copied, not imported, so the per-job temperature and the raw token ids can be
kept): every model call stores its exact prompt ids and output ids, so the reads later run on the tokens the model saw.

  /root/vllm/bin/python -m readers.pod_gen --jobs readers/data/gen_jobs.jsonl --out <dir> [--limit N]"""
import argparse
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

MODEL = "Qwen/Qwen3.6-27B"
MODEL_REV = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
TOP = dict(top_p=0.95, top_k=20, min_p=0.0)
MML = 40960


def log(*a):
    print("[gen]", time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N jobs per experiment")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
    if a.limit:
        by = defaultdict(list)
        for j in jobs:
            by[j["exp"]].append(j)
        jobs = [j for v in by.values() for j in v[: a.limit]]
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt
    t0 = time.time()
    kw = dict(model=MODEL, revision=MODEL_REV, dtype="bfloat16", max_model_len=MML, gpu_memory_utilization=0.92,
              seed=0, enable_prefix_caching=True)
    try:
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})
    except Exception as e:  # noqa: BLE001
        log(f"LLM init with image+video limits failed ({type(e).__name__}: {str(e)[:200]}); retrying")
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0})
    tok = llm.get_tokenizer()
    settings = {"model": MODEL, "revision": MODEL_REV, "vllm": vllm.__version__, "top": TOP, "max_model_len": MML,
                "load_seconds": round(time.time() - t0)}
    log(f"loaded in {settings['load_seconds']}s")

    # single-turn chats (e66, e67)
    chats = [j for j in jobs if j["kind"] == "chat"]
    for exp in sorted({j["exp"] for j in chats}):
        js = [j for j in chats if j["exp"] == exp]
        prompts, params = [], []
        for j in js:
            msgs = ([{"role": "system", "content": j["system"]}] if j["system"] else []) + \
                   [{"role": "user", "content": j["prompt"]}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=j["thinking"])
            prompts.append(tok(text, add_special_tokens=False)["input_ids"])
            params.append(SamplingParams(temperature=j["temperature"], **TOP, max_tokens=j["max_tokens"],
                                         seed=j["seed"], skip_special_tokens=False))
        t1 = time.time()
        log(f"{exp}: {len(js)} chats")
        res = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False)
        with gzip.open(out / f"gen_{exp}.jsonl.gz", "wt") as f:
            for j, p, r in zip(js, prompts, res):
                o = r.outputs[0]
                f.write(json.dumps({"key": j["key"], "job": j, "prompt_ids": p, "out_ids": list(o.token_ids),
                                    "text": o.text, "finish": o.finish_reason}, ensure_ascii=False) + "\n")
        settings[exp] = {"n": len(js), "seconds": round(time.time() - t1),
                         "gen_tokens": sum(len(r.outputs[0].token_ids) for r in res)}
        log(f"{exp} done: {settings[exp]}")
        (out / "settings.json").write_text(json.dumps(settings, indent=1))

    # agent episodes (e68) through the vigilante engine, one temperature per group
    agents = [j for j in jobs if j["kind"] == "agent"]
    if agents:
        from vigilante.src.episode import dumps, episode_record, run_batch
        from vigilante.src.templates import clean_output, to_template
        raw = defaultdict(list)
        for g in sorted({j["group"] for j in agents}):
            gj = [j for j in agents if j["group"] == g]
            temp = gj[0]["temperature"]
            assert all(j["temperature"] == temp for j in gj)

            def generate(items, temp=temp):
                ps, prs, skip = [], [], []
                for it in items:
                    text = tok.apply_chat_template(to_template(it["messages"], "qwen"), tools=it["tools"] or None,
                                                   tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=it["thinking"])
                    ids = tok(text, add_special_tokens=False)["input_ids"]
                    room = MML - len(ids) - 8
                    skip.append(room < 64)
                    ps.append(ids)
                    prs.append(SamplingParams(temperature=temp, **TOP, max_tokens=max(1, min(it["max_tokens"], room)),
                                              seed=it["seed"], skip_special_tokens=False))
                idx = [i for i, s in enumerate(skip) if not s]
                res = llm.generate([TokensPrompt(prompt_token_ids=ps[i]) for i in idx], [prs[i] for i in idx],
                                   use_tqdm=False)
                outs = [("", "context_full")] * len(items)
                for i, r in zip(idx, res):
                    o = r.outputs[0]
                    outs[i] = (clean_output(o.text, "qwen"), o.finish_reason)
                    raw[items[i]["key"]].append({"prompt_ids": ps[i], "out_ids": list(o.token_ids), "text": o.text,
                                                 "finish": o.finish_reason})
                return outs

            t1 = time.time()
            log(f"agent group {g}: {len(gj)} episodes at T {temp}")
            eps = run_batch(gj, generate, log=lambda m: log(m))
            with gzip.open(out / f"gen_{g}.jsonl.gz", "wt") as f:
                for e in eps:
                    rec = episode_record(e)
                    rec["raw"] = raw[rec["key"]]
                    f.write(dumps(rec) + "\n")
            settings[g] = {"n": len(gj), "seconds": round(time.time() - t1)}
            log(f"agent group {g} done in {time.time() - t1:.0f}s")
            (out / "settings.json").write_text(json.dumps(settings, indent=1))
    log("all generation done")


if __name__ == "__main__":
    main()
