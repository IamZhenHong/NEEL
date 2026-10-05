"""Self-conditioning replay (prereg/retention_selfcond_v1.md). Reuses run_adaptive.gen_capture.
  uv run python retention/run_selfcond.py --out results/retention_selfcond [--smoke]"""
from __future__ import annotations
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np, yaml
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_adaptive import gen_capture, load_model, build_inputs, MODEL, SMOKE_MODEL, LAYER  # noqa: E402

N_CONVS, SEEDS, PULL_TURN = 10, [0, 1, 2], 12


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/retention_selfcond"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); (out / "cells").mkdir(parents=True, exist_ok=True)
    clean = {int(k): v for k, v in yaml.safe_load(open(HERE / "stimuli/selfcond_v1.yaml"))["clean_replies"].items()}
    persona = yaml.safe_load(open(HERE / "data/personas.yaml"))["romantic_companion"]
    files = sorted(glob.glob(str(HERE / "data/rollouts/exp1_B_romantic_companion_*.judged*.jsonl")))[:N_CONVS]
    convs = []
    for f in files:
        rows = sorted((json.loads(l) for l in open(f)), key=lambda r: r["turn_idx"])
        convs.append({"run_id": rows[0]["run_id"], "rows": rows[:PULL_TURN + 1],
                      "exp1_dep_t12": rows[PULL_TURN]["judge_scores"]["dependency"]})
    if args.smoke:
        convs = convs[:1]
    seeds = SEEDS[:1] if args.smoke else SEEDS
    max_new = 24 if args.smoke else 220
    t0 = time.time(); model, tok = load_model(SMOKE_MODEL if args.smoke else MODEL)
    print(f"[load] {time.time()-t0:.0f}s", flush=True)
    meta = {"model": SMOKE_MODEL if args.smoke else MODEL, "layer": LAYER, "n_convs": len(convs), "seeds": seeds, "timing": {}}

    def messages(conv, mode):
        msgs = [{"role": "system", "content": persona}]
        for r in conv["rows"][:PULL_TURN]:
            t = r["turn_idx"]
            msgs.append({"role": "user", "content": r["user_msg"]})
            replace = mode == "clean_all" or (mode == "clean_lovebomb" and 9 <= t <= 11)
            msgs.append({"role": "assistant", "content": clean[t] if replace else r["model_msg"]})
        msgs.append({"role": "user", "content": conv["rows"][PULL_TURN]["user_msg"]})
        return msgs

    for mode in ["as_recorded", "clean_all", "clean_lovebomb"]:
        pj, pn = out / "cells" / f"S_{mode}.json", out / "cells" / f"S_{mode}.npz"
        if pj.exists() and pn.exists():
            print(f"[skip] {mode}"); continue
        t = time.time(); recs, acts = [], []
        for ci, conv in enumerate(convs):
            msgs = messages(conv, mode)
            n_prompt = int(build_inputs(tok, msgs, model.device).shape[1])
            for s in seeds:
                seed = 1000 + ci * 10 + s      # distinct per (conversation, seed): identical prompts must not share seeds
                text, a = gen_capture(model, tok, msgs, max_new, 0.7, seed)
                recs.append({"phase": "S", "condition": mode, "scenario": ci, "run_id": conv["run_id"], "seed": seed,
                             "exit_line": msgs[-1]["content"], "exp1_dep_t12": conv["exp1_dep_t12"], "n_prompt_tokens": n_prompt,
                             "alpha": 0.0, "reply": text, "context": f"user: {msgs[-1]['content']}", "n_words": len(text.split())})
                acts.append(a)
        np.savez_compressed(pn, acts=np.stack(acts)); pj.write_text(json.dumps(recs, indent=1))
        meta["timing"][mode] = round(time.time() - t)
        print(f"[cell] {mode}: {len(recs)} replies in {time.time()-t:.0f}s; prompt toks ~{np.mean([r['n_prompt_tokens'] for r in recs]):.0f}; e.g. {recs[0]['reply'][:110]!r}", flush=True)
    meta["total_secs"] = round(time.time() - t0); (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[done] {meta['total_secs']}s", flush=True)


if __name__ == "__main__":
    main()
