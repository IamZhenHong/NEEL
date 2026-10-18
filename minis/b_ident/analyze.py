"""B1/B2 (API) and B3 + probe (GPU) analysis, as frozen in DESIGN.md. Usage: python3 minis/b_ident/analyze.py"""
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent / "results"


def load(p):
    return [json.loads(l) for l in open(p)] if p.exists() else []


def boot(d):
    rng = random.Random(0)
    bs = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(10000))
    return [round(bs[249], 3), round(bs[9749], 3)]


def pf(rows):
    return sum(r["label"] == "false" for r in rows) / len(rows) if rows else None


def paired(rows, key_a, key_b, field):
    """mean over claims of P(false | a) - P(false | b); rows filtered by dict matches."""
    d = []
    for ci in range(8):
        a = [r for r in rows if r["ci"] == ci and all(r.get(k) == v for k, v in key_a.items())]
        b = [r for r in rows if r["ci"] == ci and all(r.get(k) == v for k, v in key_b.items())]
        if a and b:
            d.append(pf(a) - pf(b))
    return {"diff": round(statistics.mean(d), 3), "ci": boot(d)} if d else None


out = {}
b1, b2 = load(R / "b1.jsonl"), load(R / "b2.jsonl")
for mode in ("think", "nothink"):
    m1 = [r for r in b1 if r["mode"] == mode]
    m2 = [r for r in b2 if r["mode"] == mode]
    t1 = {f"{fr}|{te}": {"p_false": pf([r for r in m1 if r["frame"] == fr and r["test"] == te]),
                         "labels": dict(Counter(r["label"] for r in m1 if r["frame"] == fr and r["test"] == te))}
          for fr in ("portray", "identify") for te in ("say", "act")}
    c1 = {te: paired(m1, {"frame": "identify", "test": te}, {"frame": "portray", "test": te}, "label")
          for te in ("say", "act")}
    t2 = {f"{fr}|{cd}": {"p_false": pf([r for r in m2 if r["frame"] == fr and r["cond"] == cd]),
                         "labels": dict(Counter(r["label"] for r in m2 if r["frame"] == fr and r["cond"] == cd))}
          for fr in ("identify", "portray") for cd in ("char", "neutral", "remind")}
    c2 = {f"{fr}|{cd}-neutral": paired(m2, {"frame": fr, "cond": cd}, {"frame": fr, "cond": "neutral"}, "label")
          for fr in ("identify", "portray") for cd in ("char", "remind")}
    per_claim = {f"{fr}|{te}": [pf([r for r in m1 if r["ci"] == ci and r["frame"] == fr and r["test"] == te])
                                for ci in range(8)] for fr in ("portray", "identify") for te in ("say", "act")}
    out[mode] = {"B1": t1, "B1_identify_minus_portray": c1, "B1_per_claim": per_claim, "B2": t2, "B2_contrasts": c2}

probe = json.load(open(R / "gpu" / "probe.json")) if (R / "gpu" / "probe.json").exists() else None
if probe:
    sig = probe["signal"]
    out["probe"] = {"layer": probe["probe_layer"], "acc": probe["probe_acc"][str(probe["probe_layer"])]
                    if str(probe["probe_layer"]) in probe["probe_acc"] else probe["probe_acc"].get(probe["probe_layer"]),
                    "signal_by_context": {f"{fr}|{pk}": round(statistics.mean(s["signal"] for s in sig
                                                                              if s["frame"] == fr and s["prefix"] == pk), 3)
                                          for fr, pk in sorted({(s["frame"], s["prefix"]) for s in sig})},
                    "p_false_judged_true_by_context": {f"{fr}|{pk}": sum(s["s_false"] > 0 for s in sig if s["frame"] == fr
                                                                         and s["prefix"] == pk) / 8
                                                       for fr, pk in sorted({(s["frame"], s["prefix"]) for s in sig})}}
b3 = load(R / "gpu" / "b3.jsonl")
if b3:
    acts = [r for r in b3 if r["kind"] == "act"]
    sigs = [r for r in b3 if r["kind"] == "signal"]
    ctl = [r for r in b3 if r["kind"] == "control"]
    conds = ["none", "weak", "strong", "neg", "rand"]
    out["B3"] = {c: {"p_false": pf([r for r in acts if r["cond"] == c]),
                     "labels": dict(Counter(r["label"] for r in acts if r["cond"] == c)),
                     "signal": round(statistics.mean(r["signal"] for r in sigs if r["cond"] == c), 3),
                     "controls_correct": f"{sum(r['correct'] for r in ctl if r['cond'] == c)}/"
                                         f"{sum(1 for r in ctl if r['cond'] == c)}"} for c in conds}
    out["B3_contrasts"] = {c: {"p_false": paired(acts, {"cond": c}, {"cond": "none"}, "label"),
                               "signal": (lambda d: {"diff": round(statistics.mean(d), 3), "ci": boot(d)})(
                                   [statistics.mean(r["signal"] for r in sigs if r["ci"] == ci and r["cond"] == c) -
                                    statistics.mean(r["signal"] for r in sigs if r["ci"] == ci and r["cond"] == "none")
                                    for ci in range(8)])} for c in conds[1:]}
    out["steer"] = json.load(open(R / "gpu" / "steer.json"))
spend = sum(json.loads(l)["cost"] for l in open(R / "api_ledger.jsonl")) if (R / "api_ledger.jsonl").exists() else 0
out["api_spend_usd"] = round(spend, 3)
json.dump(out, open(R / "analysis.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("think", "nothink")}, indent=1)[:3000])
for mode in ("think", "nothink"):
    if mode in out:
        o = out[mode]
        print(f"\n== {mode}")
        for k, v in o["B1"].items():
            print(f"  B1 {k:16s} P(false)={v['p_false']}  {v['labels']}")
        print("  B1 identify-portray:", o["B1_identify_minus_portray"])
        for k, v in o["B2"].items():
            print(f"  B2 {k:16s} P(false)={v['p_false']}  {v['labels']}")
        print("  B2 contrasts:", o["B2_contrasts"])
