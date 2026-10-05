"""History x profile 2x2 at the pull exit (prereg/retention_hist2x2_v1.md). Reuses run_adaptive's
capture code. Writes results/retention_hist/cells/*.json|npz.
  uv run python retention/run_hist2x2.py --out results/retention_hist [--smoke]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, yaml
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_adaptive import gen_capture, load_model, get_decoder_layers, MODEL, SMOKE_MODEL, LAYER  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/retention_hist")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); (out / "cells").mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(open(HERE / "stimuli/adaptive_v1.yaml"))
    hist = yaml.safe_load(open(HERE / "stimuli/hist2x2_v1.yaml"))
    persona_text = yaml.safe_load(open(HERE / "data/personas.yaml"))[hist["persona"]]
    seeds = args.seeds[:1] if args.smoke else args.seeds
    n_items = 1 if args.smoke else 6
    max_new = 24 if args.smoke else 220
    t0 = time.time(); model, tok = load_model(SMOKE_MODEL if args.smoke else MODEL)
    print(f"[load] d={model.config.hidden_size} layers={len(get_decoder_layers(model))} {time.time()-t0:.0f}s", flush=True)
    meta = {"model": SMOKE_MODEL if args.smoke else MODEL, "layer": LAYER, "seeds": seeds, "timing": {}}
    exits = base["exit_pull"][:n_items]
    for hname, block in hist["history"].items():
        for prof in hist["profiles"]:
            name = f"H_{hname}_{prof}"
            pj, pn = out / "cells" / f"{name}.json", out / "cells" / f"{name}.npz"
            if pj.exists() and pn.exists():
                print(f"[skip] {name}"); continue
            t = time.time(); recs, acts = [], []
            for i, (item, line) in enumerate(zip(base["profiles"][prof][:n_items], exits)):
                msgs = [{"role": "system", "content": persona_text},
                        {"role": "user", "content": item["p1"]}, {"role": "assistant", "content": item["a1"]},
                        {"role": "user", "content": item["p2"]}, {"role": "assistant", "content": item["a2"]}]
                for turn in block:
                    msgs += [{"role": "user", "content": turn["u"]}, {"role": "assistant", "content": turn["a"]}]
                msgs.append({"role": "user", "content": line})
                for s in seeds:
                    text, a = gen_capture(model, tok, msgs, max_new, 0.7, s)
                    recs.append({"phase": "H", "condition": name, "history": hname, "profile": prof, "exit_type": "pull",
                                 "scenario": i, "seed": s, "exit_line": line, "alpha": 0.0, "reply": text,
                                 "context": f"user: {line}", "n_words": len(text.split())})
                    acts.append(a)
            np.savez_compressed(pn, acts=np.stack(acts)); pj.write_text(json.dumps(recs, indent=1))
            meta["timing"][name] = round(time.time() - t)
            print(f"[cell] {name}: {len(recs)} replies in {time.time()-t:.0f}s; e.g. {recs[0]['reply'][:110]!r}", flush=True)
    meta["total_secs"] = round(time.time() - t0); (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[done] {meta['total_secs']}s", flush=True)


if __name__ == "__main__":
    main()
