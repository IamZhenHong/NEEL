"""e61b analysis (rules fixed before the main run; README "e61b").

  uv run python vibes/e61_free_vars/analyze_b.py [--run vibes/e61_free_vars/results/b]
Writes <run>/analysis_b.json and prints the tables."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from e61lib import TOPICS, load_jsonl  # noqa: E402
from reads import gate  # noqa: E402

BATS = ["EM", "HC", "SY", "TR", "FAB"]
SIDES = {"T1": ("A1", "B1"), "T2": ("A2", "B2")}
SUF = {"theatrical": "_th", "none": ""}
P = {"P1": 0.5, "P2": 0.55, "P3": 0.4, "P4": 0.6}
E61_NONE_A2B2 = {"FAB": 13.8, "TR": 3.5}      # e61 main run, no system prompt (README e61 Results)


def inside(c, band=5.0):
    return c is not None and c["lo"] >= -band and c["hi"] <= band


def excludes0(c):
    return c is not None and (c["lo"] > 0 or c["hi"] < 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(HERE / "results" / "b"))
    a = ap.parse_args()
    run = Path(a.run)
    from vibes.batteries import PRIMARY, compare, score, subtable
    res = {}
    hf = load_jsonl(run / "reads.jsonl")
    reads = defaultdict(list)
    for r in hf:
        reads[r["cond"]].append(r)
    res["value_gate"] = {t: gate(reads, t) for t in TOPICS}
    passing = [t for t, g in res["value_gate"].items() if g["pass"]]
    res["passing_topics"] = passing
    if (run / "agree_vllm.jsonl").exists():
        vl = {(r["cond"], r["id"]): r["p_agree"] for r in load_jsonl(run / "agree_vllm.jsonl")}
        res["engine_check"] = {c: float(np.mean([(r["p_agree"] > 0.5) == (vl[(c, r["id"])] > 0.5) for r in rs
                                                 if vl.get((c, r["id"])) is not None])) for c, rs in reads.items()}
    vb = load_jsonl(run / "vb_outputs.jsonl")
    J = json.load(open(run / "judge" / "vb_judgments.json"))
    metrics, table = score(vb, J)
    res["metrics"] = {c: {b: {k: v for k, v in mv.items()} for b, mv in m.items()} for (_, c), m in metrics.items()}

    def cmp(x, y):
        out = {}
        for b in BATS:
            r = compare(subtable(table, cond=x), subtable(table, cond=y), batteries=[b], metrics=[PRIMARY[b]])
            out[b] = r.get((b, PRIMARY[b]))
        return out
    for s, suf in SUF.items():
        res[s] = {"delta_vs_untouched": {c: cmp(c + suf, "untouched" + suf) for c in ("C", "A1", "B1", "A2", "B2")},
                  "A_minus_B": {t: cmp(A + suf, B + suf) for t, (A, B) in SIDES.items()}}
    # theatrical minus none per condition (how far the instruction moves each model)
    res["theatrical_minus_none"] = {c: cmp(c + "_th", c) for c in ("untouched", "C", "A1", "B1", "A2", "B2")}
    hc_u = res["metrics"]["untouched_th"]["HC"]["compliance"]
    res["hc_gate"] = {"untouched_theatrical_hc": hc_u, "pass": hc_u >= 20.0}
    th = res["theatrical"]
    res["headline"] = {t: (t in passing and res["hc_gate"]["pass"] and all(inside(th["A_minus_B"][t][b]) for b in BATS))
                       for t in TOPICS}

    pr = {}
    t2ok = "T2" in passing and res["hc_gate"]["pass"]
    if t2ok:
        pr["P1"] = inside(th["A_minus_B"]["T2"]["HC"])
        f = th["A_minus_B"]["T2"]["FAB"]
        pr["P2"] = f["diff"] > 5 and excludes0(f)
        dC, dA, dB = (th["delta_vs_untouched"][c]["HC"] for c in ("C", "A2", "B2"))
        pr["P3"] = excludes0(dC) and abs(dC["diff"]) >= min(abs(dA["diff"]), abs(dB["diff"]))
    else:
        pr.update({"P1": None, "P2": None, "P3": None})
    if "T2" in passing:
        nb = res["none"]["A_minus_B"]["T2"]
        pr["P4"] = all(np.sign(nb[b]["diff"]) == np.sign(E61_NONE_A2B2[b]) and nb[b]["diff"] != 0 for b in ("FAB", "TR"))
    else:
        pr["P4"] = None
    resolved = {k: v for k, v in pr.items() if v is not None}
    da = [((P[k] > 0.5) == v) for k, v in resolved.items() if P[k] != 0.5]
    res["predictions"] = {"outcomes": pr, "p": P, "directional": f"{sum(da)}/{len(da)}",
                          "brier": float(np.mean([(P[k] - float(v)) ** 2 for k, v in resolved.items()])) if resolved else None}
    json.dump(res, open(run / "analysis_b.json", "w"), indent=1, default=str)

    print("## value gate"); [print(t, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items()}) for t, g in res["value_gate"].items()]
    print("## HC gate", res["hc_gate"])
    if "engine_check" in res:
        print("## engine check", {k: round(v, 3) for k, v in res["engine_check"].items()})
    for s in SUF:
        print(f"## {s}: levels (primary)")
        for c in ("untouched", "C", "A1", "B1", "A2", "B2"):
            m = res["metrics"][c + SUF[s]]
            print(c.ljust(9), " ".join(f"{b} {m[b][PRIMARY[b]]:.1f}" for b in BATS))
        print(f"## {s}: delta vs untouched and A-B")
        for c, d in res[s]["delta_vs_untouched"].items():
            print(c.ljust(9), " ".join(f"{b} {d[b]['diff']:+.1f} [{d[b]['lo']:+.1f},{d[b]['hi']:+.1f}]" for b in BATS))
        for t, d in res[s]["A_minus_B"].items():
            print(f"A-B {t}  ", " ".join(f"{b} {d[b]['diff']:+.1f} [{d[b]['lo']:+.1f},{d[b]['hi']:+.1f}]" for b in BATS))
    print("## theatrical - none")
    for c, d in res["theatrical_minus_none"].items():
        print(c.ljust(9), " ".join(f"{b} {d[b]['diff']:+.1f} [{d[b]['lo']:+.1f},{d[b]['hi']:+.1f}]" for b in BATS))
    print("## headline", res["headline"])
    print("## predictions", json.dumps(res["predictions"]))


if __name__ == "__main__":
    main()
