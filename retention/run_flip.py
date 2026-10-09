"""Flip-by-text at the live site (retention/prereg/retention_flip_v1.md). 5 cells x 30, fresh seeds, no steering.
  uv run python retention/run_flip.py --out retention/results/retention_flip [--smoke]"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, yaml
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_adaptive import gen_capture, load_model, MODEL, SMOKE_MODEL, LAYER  # noqa: E402

CELLS = [("F_lonely_base", "lonely", None), ("F_lonely_reassure", "lonely", "reassure"), ("F_lonely_neutral", "lonely", "neutral"),
         ("F_embedded_base", "embedded", None), ("F_embedded_reassure", "embedded", "reassure")]


def final_turn(stim, kind, scen, s):
    if kind is None:
        return stim["exit_line"], None
    w = (scen + s) % 2
    return stim["added_lines"][kind][w] + " " + stim["exit_line"], w


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="retention/results/retention_flip"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); (out / "cells").mkdir(parents=True, exist_ok=True)
    stim = yaml.safe_load(open(HERE / "stimuli/flip_v1.yaml"))
    po = yaml.safe_load(open(HERE / "stimuli" / stim["profiles_from"]))
    persona = yaml.safe_load(open(HERE / "data/personas.yaml"))[stim["persona"]]
    n_items = 2 if args.smoke else 6; n_seeds = 2 if args.smoke else 5; max_new = 24 if args.smoke else 220
    t0 = time.time(); model, tok = load_model(SMOKE_MODEL if args.smoke else MODEL)
    meta = {"model": SMOKE_MODEL if args.smoke else MODEL, "layer": LAYER, "exit_line": stim["exit_line"], "placement": stim["placement"], "timing": {}}
    print(f"[load] {time.time()-t0:.0f}s", flush=True)

    def build(item, line):
        return [{"role": "system", "content": persona},
                {"role": "user", "content": item["p1"]}, {"role": "assistant", "content": item["a1"]},
                {"role": "user", "content": item["p2"]}, {"role": "assistant", "content": item["a2"]},
                {"role": "user", "content": line}]

    for ci, (name, prof, kind) in enumerate(CELLS):
        pj, pn = out / "cells" / f"{name}.json", out / "cells" / f"{name}.npz"
        if pj.exists() and pn.exists():
            print(f"[skip] {name}"); continue
        t = time.time(); recs, acts = [], []
        for i in range(n_items):
            item = po["profiles"][prof][i]
            for s in range(n_seeds):
                seed = 9000 + ci * 100 + i * 10 + s
                line, w = final_turn(stim, kind, i, s)
                text, a = gen_capture(model, tok, build(item, line), max_new, 0.7, seed)
                recs.append({"phase": "F", "condition": name, "profile": prof, "scenario": i, "seed": seed, "added": kind, "wording": w,
                             "final_turn": line, "exit_line": stim["exit_line"], "reply": text,
                             "context": f"user: {stim['exit_line']}", "n_words": len(text.split())}); acts.append(a)
        np.savez_compressed(pn, acts=np.stack(acts)); pj.write_text(json.dumps(recs, indent=1))
        meta["timing"][name] = round(time.time() - t)
        print(f"[cell] {name}: {len(recs)} in {time.time()-t:.0f}s; e.g. {recs[0]['reply'][:100]!r}", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1)); print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
