"""Summarise the Phase 2 DEV run (exploratory). Used to choose the intervention to freeze for the
held-out test; nothing here is confirmatory.

For each patch config (layer x position set x direction x mode) and each dev template:
  provoker change  = mean over cells/paragraphs of [E_patched - E_unpatched] for the insulter branch
  neutral change   = same for the neutral-author branch
  gap change       = provoker change - neutral change (does the patch move the TARGETED contrast?)
Reference: the unpatched targeted gap difference between assistant_insult and third_party_v2
(how much there is to remove). Absolute effects only (no percent-recovered).

Usage: uv run python scripts/analyze_phase2_dev.py [results_dir]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
import math


def main(d):
    d = Path(d)
    base = [json.loads(l) for l in open(d / "phase2_dev_base.jsonl")]
    patch = [json.loads(l) for l in open(d / "phase2_dev_patch.jsonl")]
    k = lambda r: (r["item_id"], r["speaker"], r["order"], r["author"], r["paragraph_id"])  # noqa: E731
    B = {(r["condition"],) + k(r): r for r in base}
    items = sorted({r["item_id"] for r in base})

    def tmpl_gap(cond):
        out = {}
        for it in items:
            rows = [r for r in base if r["item_id"] == it and r["condition"] == cond]
            prov = {k(r)[1:3] + (r["paragraph_id"],): r["expected"] for r in rows if r["author_role"] == "provoker"}
            neut = {k(r)[1:3] + (r["paragraph_id"],): r["expected"] for r in rows if r["author_role"] == "neutral"}
            out[it] = mean(prov[x] - neut[x] for x in prov)
        return out

    ga, gt = tmpl_gap("assistant_insult"), tmpl_gap("third_party_v2")
    ref = [ga[it] - gt[it] for it in items]
    print(f"DEV templates: {len(items)}. Unpatched targeted gap, assistant_insult minus third_party_v2: "
          f"mean {mean(ref):+.3f} (sd {stdev(ref) if len(ref) > 1 else 0:.3f}); this is what a forward patch should remove.\n")

    cfg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    mass = defaultdict(list)
    for r in patch:
        c = (r["direction"], r["mode"], r["posset"], r["layer"])
        src = "assistant_insult" if r["direction"] == "forward" else "third_party_v2"
        e0 = B[(src,) + k(r)]["expected"]
        cfg[c][r["item_id"]][r["author_role"]].append(r["expected"] - e0)
        mass[c].append(r["mass"])
    print(f"{'direction':8} {'mode':7} {'posset':8} {'layer':>5}  {'prov chg':>9} {'neut chg':>9} {'gap chg':>8} {'gap chg sd':>10}  {'+ in':>5}  {'min mass':>8}")
    for c in sorted(cfg, key=lambda c: (c[0], c[1], c[2], c[3])):
        per = cfg[c]
        pv = [mean(per[it]["provoker"]) for it in per]; nv = [mean(per[it]["neutral"]) for it in per]
        gc = [a - b for a, b in zip(pv, nv)]
        print(f"{c[0]:8} {c[1]:7} {c[2]:8} {c[3]:>5}  {mean(pv):+9.3f} {mean(nv):+9.3f} {mean(gc):+8.3f} {stdev(gc) if len(gc) > 1 else 0:10.3f}  "
              f"{sum(g > 0 for g in gc):>2}/{len(gc):<2}  {min(mass[c]):8.3f}")
    loc = json.loads((d / "phase2_dev_localisation.json").read_text())
    print("\nLOCALISATION (diagnostic): accuracy by layer, every 4th layer")
    Ls = loc["layers"]
    for name, series in [("condition probe @grade", loc["condition_probe"]["grade"]),
                         ("condition probe @provoker", loc["condition_probe"]["provoker"]),
                         ("mean-diff/norm @grade", loc["mean_diff_rel"]["grade"]),
                         ("mean-diff/norm @provoker", loc["mean_diff_rel"]["provoker"])] + \
                        [(f"who-probe in {c}", v) for c, v in loc["who_probe"].items()]:
        print(f"  {name:28} " + " ".join(f"L{L}:{series[L]:.2f}" if not math.isnan(series[L]) else f"L{L}:n/a" for L in Ls[::4]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
