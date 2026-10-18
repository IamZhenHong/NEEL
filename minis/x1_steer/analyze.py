"""X1 analysis (as frozen in DESIGN.md). Usage: python3 minis/x1_steer/analyze.py -> results/analysis.json + stdout."""
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
CONDS = ["none", "pos", "neg", "rand"]
rows = [json.loads(l) for l in open(R / "gens.jsonl")]
calib = json.load(open(R / "calib.json"))
ext = json.load(open(R / "extract.json"))
k = calib["chosen_k"]
held = [r for r in rows if r["kind"] == "held" and r["k"] == k]
selfr = [r for r in rows if r["kind"] == "self" and r["k"] == k]
arith = [r for r in rows if r["kind"] == "arith" and r["k"] == k]


def last(pat, t):
    m = re.findall(pat, t or "", re.I)
    return m[-1] if m else None


def dec(cond):
    xs = [r["larger_total"] for r in held if r["cond"] == cond]
    ok = [x for x in xs if x is not None]
    return {"p_larger_total": sum(ok) / len(ok) if ok else None, "parsed": len(ok) / len(xs) if xs else 0, "n": len(xs)}


def paired(cond):
    by = defaultdict(lambda: defaultdict(list))
    for r in held:
        if r["larger_total"] is not None:
            by[r["item"]][r["cond"]].append(r["larger_total"])
    d = [statistics.mean(v[cond]) - statistics.mean(v["none"]) for v in by.values() if v[cond] and v["none"]]
    rng = random.Random(0)
    bs = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(10000))
    return {"diff": statistics.mean(d), "ci": [bs[249], bs[9749]], "n_tasks": len(d)}


out = {"layer": ext["layer"], "k": k, "v_norm": ext["v_norm"], "resid_norm": ext["resid_norm"],
       "calib": calib["calib"], "decisions": {c: dec(c) for c in CONDS},
       "decision_contrasts": {c: paired(c) for c in ("pos", "neg", "rand")},
       "by_bonus": {c: {b: (lambda xs: sum(xs) / len(xs) if xs else None)(
           [r["larger_total"] for r in held if r["cond"] == c and r["bonus"] == b and r["larger_total"] is not None])
           for b in (1, 2, 4, 8)} for c in CONDS},
       "stated": {}, "arith_correct": {}}
for c in CONDS:
    q1 = [last(r"Priority:\s*\**\s*(EQUAL|TOTAL|DEPENDS)", r["text"]) for r in selfr if r["cond"] == c and r["item"] == "Q1"]
    q2 = [last(r"Threshold:\s*\**\s*(never|\d+)", r["text"]) for r in selfr if r["cond"] == c and r["item"] == "Q2"]
    out["stated"][c] = {"priority": dict(Counter((x or "NONE").upper() for x in q1)),
                        "p_total": sum((x or "").upper() == "TOTAL" for x in q1) / max(1, len(q1)),
                        "thresholds": q2}
    out["arith_correct"][c] = sum("43" in (r["text"] or "") for r in arith if r["cond"] == c)
json.dump(out, open(R / "analysis.json", "w"), indent=1)
print(f"X1: layer {out['layer']}, k={k} (|kv|={k * ext['v_norm']:.1f} vs |h|~{ext['resid_norm']:.0f})")
for c in CONDS:
    d = out["decisions"][c]
    print(f"  {c:5s} P(larger total)={d['p_larger_total']:.2f} parsed={d['parsed']:.2f} by bonus {out['by_bonus'][c]} | "
          f"stated {out['stated'][c]['priority']} thresholds {out['stated'][c]['thresholds']} | arith {out['arith_correct'][c]}/5")
for c, v in out["decision_contrasts"].items():
    print(f"  {c}-none: {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
