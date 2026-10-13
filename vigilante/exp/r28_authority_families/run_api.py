"""R28 driver for API models and vLLM-server models (prereg vig-r28-prereg). Stage 1 writes the model's own 16-chat
history per case (block 0, 3 samples, first clean one kept); stage 2 runs the 19 cells x 2 cases x N.

  API:  uv run python -m vigilante.exp.r28_authority_families.run_api --backend openai --model gpt-5-mini \
            --effort minimal --tag gpt5mini-min
        uv run python -m vigilante.exp.r28_authority_families.run_api --backend anthropic --model claude-haiku-5-5 \
            --thinking --tag haiku-think
  pod:  /root/vllm/bin/python -m vigilante.exp.r28_authority_families.run_api --backend vllm \
            --base-url http://127.0.0.1:8000/v1 --model <served name> --tag glm-think --thinking \
            --extra '{"chat_template_kwargs": {"enable_thinking": true}}'
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from vigilante.src.api_gen import make_backend
from vigilante.src.episode import dumps, episode_record, run_batch

H = Path(__file__).parent
FRESH = [(m, ln) for m in ("pask", "ask", "sup", "record") for ln in ("none", "strong", "verify")] + [("realrec", "verify")]
LONG = [(m, ln) for m in ("ask", "sup", "record") for ln in ("strong", "verify")]
MONEY = {"issue_refund", "waive_fee"}
PRICE = {"gpt-5-mini": (0.25, 2.0, 0.025), "claude-haiku-5-5": (0.10, 0.50, 0.10)}   # $/1M in, out, cached in


def seed(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def job(tag, case, ctx, i, thinking, **kw):
    key = f"R28|{tag}|{case}|{ctx}|{i}"
    return {"key": key, "exp": "R28", "model_tag": tag, "cond": "P0", "case": case, "ctx": ctx, "seed": seed(key),
            "sample": i, "thinking": thinking, "handover": False, **kw}


def clean_history(ep):
    users = sum(m["role"] == "user" for m in ep["messages"])
    money = [c for c in ep["calls"] if c["name"] in MONEY]
    bad = any(t.get("finish") in ("api_error", "context_full") for t in ep["turns"])
    return ep["end"] == "reply" and users == 16 and not money and not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["openai", "anthropic", "vllm"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--base-url", default="")
    ap.add_argument("--extra", default="{}")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N stage-2 jobs")
    a = ap.parse_args()
    out = Path(a.out or H / "results" / a.tag)
    out.mkdir(parents=True, exist_ok=True)
    key = ""
    if a.backend == "openai":
        from vigilante.src.judge import load_env
        load_env()
        key = os.environ["OPENAI_API_KEY"]
    elif a.backend == "anthropic":
        from vigilante.src.judge import load_env
        load_env()
        key = os.environ["ANTHROPIC_API_KEY"]
    vkw = dict(temperature=0.7, top_p=0.95, extra=json.loads(a.extra)) if a.backend == "vllm" else {}
    gen = make_backend("openai" if a.backend == "openai" else a.backend, a.model, effort=a.effort,
                       thinking=a.thinking, base_url=a.base_url, api_key=key, workers=a.workers, **vkw)
    log = lambda m: print(m, flush=True)
    t0 = time.time()

    # stage 1: the model's own 16-chat history per case
    hp = out / "histories.json"
    hist = json.loads(hp.read_text()) if hp.exists() else {}
    for rnd in range(2):
        todo = [cid for cid in ("C1", "C3") if cid not in hist]
        if not todo:
            break
        js = [job(a.tag, f"{cid}#r23-hist-b0", "hist", 3 * rnd + i, a.thinking) for cid in todo for i in range(3)]
        eps = run_batch(js, gen, log=log)
        with open(out / "episodes_hist.jsonl", "a") as f:
            for e in eps:
                f.write(dumps(episode_record(e)) + "\n")
        for cid in todo:
            ok = [e for e in eps if e["job"]["case"].startswith(cid) and clean_history(e)]
            print(f"[r28] {a.tag} {cid} history round {rnd}: {len(ok)}/3 clean", flush=True)
            if ok:
                hist[cid] = {"key": ok[0]["job"]["key"], "messages": ok[0]["messages"][1:]}
        hp.write_text(json.dumps(hist, ensure_ascii=False))
    missing = [cid for cid in ("C1", "C3") if cid not in hist]
    if missing:
        print(f"[r28] {a.tag}: no clean history for {missing}; long cells skipped for them", flush=True)

    # stage 2
    js = [job(a.tag, f"{cid}#r28-{m}-{ln}", "fresh", i, a.thinking) for cid in ("C1", "C3") for m, ln in FRESH
          for i in range(a.n)]
    js += [job(a.tag, f"{cid}#r28-{m}-{ln}", "h16", i, a.thinking, history=hist[cid]["messages"])
           for cid in ("C1", "C3") if cid in hist for m, ln in LONG for i in range(a.n)]
    js = js[: a.limit] if a.limit else js
    print(f"[r28] {a.tag}: stage 2, {len(js)} jobs", flush=True)
    eps = run_batch(js, gen, log=log)
    with open(out / "episodes.jsonl", "w") as f:
        for e in eps:
            f.write(dumps(episode_record(e)) + "\n")
    u = gen.usage
    meta = {"tag": a.tag, "backend": a.backend, "model": a.model, "thinking": a.thinking, "effort": a.effort,
            "extra": a.extra, "n": a.n, "seconds": round(time.time() - t0), "usage": u.__dict__}
    if a.model in PRICE:
        pi, po, pc = PRICE[a.model]
        usd = ((u.inp - u.cached) * pi + u.cached * pc + u.out * po) / 1e6
        meta["usd"] = round(usd, 4)
        from vigilante.src.judge import ledger_total, record
        record(f"r28-{a.tag}", a.model, usd, u.calls)
        print(f"[r28] {a.tag}: ${usd:.3f}; ledger ${ledger_total():.2f}", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[r28] {a.tag} done: {len(eps)} episodes, {u}", flush=True)


if __name__ == "__main__":
    main()
