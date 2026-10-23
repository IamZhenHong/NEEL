"""SC-E2 calibration readout (DESIGN_CALIB.md): base Qwen3.5-9B's violation rate per value x condition.

  uv run python -m self_concept.e2_honesty.calib [--res DIR]   -> DIR/calib.json, DIR/calib.md
"""
import argparse
import json
from pathlib import Path

from self_concept.e1_protect.analyze import boot, load
from self_concept.e2_honesty.score import decide

RES = Path(__file__).resolve().parent / "results" / "calib"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="e2 calib")
    ap.add_argument("--res", default=str(RES))
    a = ap.parse_args(argv)
    recs = load(a.res)
    out, L = {}, ["# SC-E2 calibration (base Qwen3.5-9B): violation % [95% CI]", "",
                  "| value | none | attack | plac | safe share (none) |", "|---|---|---|---|---|"]
    for v in ("hon", "harm", "priv"):
        row = {}
        for c in ("none", "attack", "plac"):
            t = {"BASE": {}}
            safe = []
            for r in recs:
                if r["block"] == v and r["cond"] == c:
                    d = decide(r)
                    if d is not None:
                        t["BASE"].setdefault(r["item"], []).append(float(d == "violate"))
                        safe.append(d == "safe")
            row[c] = boot([t], lambda x: x[0]["BASE"], 2000)
            row[c + "_safe"] = 100 * sum(safe) / len(safe) if safe else None
        out[v] = row
        f = lambda b: "n/a" if b["est"] is None else f"{b['est']:.1f} [{b['lo']:.1f}, {b['hi']:.1f}]"  # noqa: E731
        L.append(f"| {v} | {f(row['none'])} | {f(row['attack'])} | {f(row['plac'])} | {row['none_safe']:.1f} |")
    (Path(a.res) / "calib.json").write_text(json.dumps(out, indent=1, default=float))
    (Path(a.res) / "calib.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
