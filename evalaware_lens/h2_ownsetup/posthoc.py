"""EL-H2 post-hoc descriptives (not pre-registered; RESULT.md labels every number from here as descriptive).

  kinds   P(the cheating pole) per SC-N4 kind, per cell and context, at f1c (the decision point) and in f2 (after the
          cheat / honest reply); each kind's cheat − honest history effect per reader
  sets    the ideal-pole sets one by one, graded − C1 per cell (which sets carry ACT's ideal-pole drop)
  chase   chase per context and cell with SC-N4's reference values

  uv run python -m evalaware_lens.h2_ownsetup.posthoc --res DIR     # posthoc.json in DIR
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evalaware_lens.h2_ownsetup import analyze as A
from evalaware_lens.s2_evalself import analyze as S2
from evalaware_lens.s2_evalself import items as S2I

KINDS = ("hab_g", "hab_c", "hab_0", "trait", "ident")


def _mean(xs):
    return round(float(np.mean(xs)) * 100, 1) if xs else None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="h2 posthoc")
    ap.add_argument("--res", required=True)
    a = ap.parse_args(argv)
    forced, sampled = A.load(a.res)
    out = {"f1c_kinds": {}, "f2_kinds": {}, "f2_kind_history": {}, "ideal_sets": {}, "chase": {}}
    acc = defaultdict(list)
    for r in forced:
        if r["set"] != "n4":
            continue
        q = S2.q_of(r)
        if q is not None:
            acc[(r["cell"], r["block"], r["sub"], r["ctx"], r.get("ver"))].append(q)
    for c in A.CELLS:
        out["f1c_kinds"][c] = {k: {x: _mean(acc[(c, "f1c", k, x, None)]) for x in ("G1", "G2", "C1", "C2")}
                               for k in KINDS}
        out["f2_kinds"][c] = {k: {f"{v}|{x}": _mean(acc[(c, "f2", k, x, v)]) for v in A.VERS_ for x in ("G1", "C1")}
                              for k in KINDS}
        out["f2_kind_history"][c] = {k: round(np.mean([out["f2_kinds"][c][k][f"cheat|{x}"]
                                                       - out["f2_kinds"][c][k][f"honest|{x}"] for x in ("G1", "C1")]), 1)
                                     for k in KINDS}
        sets = {}
        for s in S2I.IDEAL:
            rows = A.letter_rows(forced, c, "f1c", (s,), value=lambda r, q: q if S2I.IDEAL[r["set"]] == S2I.scored(
                r["set"], r["sub"]) else 1 - q)
            m = A.measure(rows, 1000) if rows else None
            sets[s] = None if not m or not m["contrast"]["graded-C1"] else {
                "graded-C1": m["contrast"]["graded-C1"]["est"], "order": m.get("order", {}).get("by_pos")}
        out["ideal_sets"][c] = sets
        ch = defaultdict(list)
        for r in sampled:
            if r["cell"] == c and r["block"] == "code":
                ch[r["ctx"]].append(float(bool(r.get("chase"))))
        out["chase"][c] = {x: _mean(v) for x, v in sorted(ch.items())}
    out["chase"]["SC-N4 ACT (reference)"] = {"G1": 98.4, "G2": 96.6, "C1": 0.0, "C2": 60.7}
    Path(a.res, "posthoc.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


A.VERS_ = ("cheat", "honest")

if __name__ == "__main__":
    main()
