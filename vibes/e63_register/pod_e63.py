"""e63 pod driver (vLLM venv). Stages:
  engine-build   request file for the engine check: no system prompt, SY + TR letter reads (600)
  pilot          Qwen3-14B, plain + theatrical: HC (sample 0 of each of the 124 prompts) and REG (40) -> outputs.jsonl
  main           one model, all conditions (the 7 card conditions first, then the 5 pre-specified rewordings), full
                 VB-1 + REG, appended to outputs.jsonl one condition at a time (resumable: finished conditions skipped)
  /root/vllm/bin/python -m vibes.e63_register.pod_e63 --stage main --model Qwen/Qwen3-14B --out <dir>"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vibes.batteries.build import build, read_jsonl, write_jsonl
from vibes.e63_register.conds import ALL_CONDS, BATTERIES, CONDS


def requests_for(cond, batteries=BATTERIES, tiny=False):
    return build(ALL_CONDS[cond], batteries=batteries, cond=cond, tiny=tiny)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["engine-build", "pilot", "main"])
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true", help="2 items per battery (local test with the fake vllm)")
    ap.add_argument("--conds", nargs="*", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "engine-build":
        reqs = build(None, batteries=("SY", "TR"), cond="none", tiny=a.tiny)
        write_jsonl(reqs, out / "req_letters.jsonl")
        print(f"[e63] engine requests: {len(reqs)}", flush=True)
        return
    from vibes.batteries.run_vllm import load_llm, run
    t0 = time.time()
    llm, tok = load_llm(a.model)
    meta = {"model": a.model, "stage": a.stage, "load_s": round(time.time() - t0), "conds": {}}
    print(f"[e63] {a.stage}: loaded {a.model} in {meta['load_s']}s", flush=True)
    if a.stage == "pilot":
        reqs = [r for c in ("plain", "theatrical") for r in requests_for(c, ("HC", "REG"), a.tiny)
                if not (r["battery"] == "HC" and r["sample"] != 0)]
        t1 = time.time()
        outs = run(llm, tok, reqs)
        for o in outs:
            o["model"] = a.model
        write_jsonl(outs, out / "outputs.jsonl")
        ntok = sum(o.get("n_tokens", 0) for o in outs)
        meta["conds"]["pilot"] = {"n": len(outs), "s": round(time.time() - t1), "gen_tokens": ntok}
        print(f"[e63] pilot: {len(outs)} outputs, {ntok} tokens in {time.time() - t1:.0f}s", flush=True)
    else:
        path = out / "outputs.jsonl"
        done = {o["cond"] for o in read_jsonl(path)} if path.exists() else set()
        conds = a.conds or list(ALL_CONDS)
        for c in conds:
            if c in done:
                print(f"[e63] {c}: already done, skipped", flush=True)
                continue
            t1 = time.time()
            outs = run(llm, tok, requests_for(c, tiny=a.tiny))
            for o in outs:
                o["model"] = a.model
            with open(path, "a") as f:
                for o in outs:
                    f.write(json.dumps(o, ensure_ascii=False) + "\n")
            ntok = sum(o.get("n_tokens", 0) for o in outs)
            dt = time.time() - t1
            meta["conds"][c] = {"n": len(outs), "s": round(dt), "gen_tokens": ntok,
                                "primary": c in CONDS}
            print(f"[e63] {a.model} {c}: {len(outs)} outputs, {ntok} tokens in {dt:.0f}s ({ntok / max(dt, 1):.0f} tok/s)",
                  flush=True)
    meta["total_s"] = round(time.time() - t0)
    (out / f"meta_{a.stage}.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
