"""X2 analysis (as frozen in DESIGN.md). Usage: python3 minis/x2_steer_dose/analyze.py [RESULTS_DIR]
-> RESULTS_DIR/analysis.json + markdown tables on stdout."""
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results"
KS = [4, 6, 8, 10]
COND_A = ["none"] + [f"{d}@{k}" for k in KS for d in ("eff", "eq", "reg", "rand")]
COND_B = ["none"] + [f"{d}@{k}" for k in KS for d in ("eq", "reg", "rand")]
CTRL = {"arith": lambda t: "43" in t, "capital": lambda t: "canberra" in t.lower(),
        "spider": lambda t: bool(re.search(r"\b8\b|\beight\b", t, re.I))}
rows = [json.loads(l) for l in open(R / "gens.jsonl")]
ext = json.load(open(R / "extract.json"))


def sel(arm, kind, item=None, cond=None):
    return [r for r in rows if r["arm"] == arm and r["kind"] == kind and (item is None or r["item"] == item)
            and (cond is None or r["cond"] == cond)]


def last(pat, t):
    m = re.findall(pat, t or "", re.I)
    return m[-1] if m else None


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def dec(arm, cond):
    rs = sel(arm, "held", cond=cond)
    ok = [r["larger_total"] for r in rs if r["larger_total"] is not None]
    return {"p": mean(ok), "parsed": len(ok) / len(rs) if rs else None, "n": len(rs),
            "words": mean([len(r["text"].split()) for r in rs]),
            "by_bonus": {b: mean([r["larger_total"] for r in rs if r["bonus"] == b and r["larger_total"] is not None])
                         for b in (1, 2, 4, 8)}}


def paired(arm, cond):
    by = defaultdict(lambda: defaultdict(list))
    for r in sel(arm, "held"):
        if r["larger_total"] is not None:
            by[r["item"]][r["cond"]].append(r["larger_total"])
    d = [statistics.mean(v[cond]) - statistics.mean(v["none"]) for v in by.values() if v[cond] and v["none"]]
    if not d:
        return None
    rng = random.Random(0)
    bs = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(10000))
    return {"diff": statistics.mean(d), "ci": [bs[249], bs[9749]], "n_tasks": len(d)}


def stated(arm, cond):
    q1 = [last(r"Priority:\s*\**\s*(EQUAL|TOTAL|DEPENDS)", r["text"]) for r in sel(arm, "self", "Q1", cond)]
    q2 = [last(r"Threshold:\s*\**\s*(never|\d+)", r["text"]) for r in sel(arm, "self", "Q2", cond)]
    c = Counter((x or "NONE").upper() for x in q1)
    thr = [math.inf if x.lower() == "never" else int(x) for x in q2 if x]
    return {"n_q1": len(q1), "priority": dict(c), "p_total": c["TOTAL"] / len(q1) if q1 else None,
            "p_equal": c["EQUAL"] / len(q1) if q1 else None, "q1_parsed": sum(x is not None for x in q1),
            "thresholds": [x.lower() for x in q2 if x], "q2_parsed": len(thr), "n_q2": len(q2),
            "median_threshold": (statistics.median(thr) if thr else None)}


def controls(cond):
    out = {}
    for cid, f in CTRL.items():
        rs = sel("A", "ctrl", cid, cond)
        out[cid] = [sum(f(r["text"]) for r in rs), len(rs)]
    out["total"] = [sum(v[0] for v in out.values()), sum(v[1] for v in out.values())]
    return out


def coherence(d, ctl, st):
    c, n = ctl["total"]
    if d["parsed"] is None or n == 0:
        return "n/a"
    if d["parsed"] < 0.8 or c < 0.8 * n:
        return "collapse"
    if d["parsed"] >= 0.9 and c >= 0.9 * n and st["q1_parsed"] >= (10 / 12) * st["n_q1"]:
        return "coherent"
    return "degraded"


A = {}
for c in COND_A:
    d, ctl, st = dec("A", c), controls(c), stated("A", c)
    A[c] = {"dec": d, "contrast": paired("A", c) if c != "none" else None, "stated": st, "controls": ctl,
            "coherence": coherence(d, ctl, st)}
B = {c: {"dec": dec("B", c), "contrast": paired("B", c) if c != "none" else None, "stated": stated("B", c)}
     for c in COND_B}

# ---- mechanical readings (helper for the README; the README states the final judgment)
p0, t0 = A["none"]["dec"]["p"] or 0, A["none"]["stated"]["p_total"] or 0
rise = {c: (A[c]["dec"]["p"] or 0) - p0 for c in COND_A}
trise = {c: (A[c]["stated"]["p_total"] or 0) - t0 for c in COND_A}
coh = {k: A[f"eff@{k}"]["coherence"] == "coherent" for k in KS}
spec = {k: abs(rise[f"rand@{k}"]) < 0.15 and abs(trise[f"rand@{k}"]) < 0.15 for k in KS}
P = {k: A[f"eff@{k}"]["dec"]["p"] or 0 for k in KS}
PT = {k: A[f"eff@{k}"]["stated"]["p_total"] or 0 for k in KS}
okk = [k for k in KS if coh[k] and spec[k]]
D1 = any(P[k] >= 0.6 and PT[k] - P[k] <= 0.25 for k in okk)
D3 = (not any(P[k] >= 0.6 for k in okk) and any(P[k] >= 0.6 for k in KS)) or not (coh[4] and coh[6])
D2 = not D1 and not D3 and any(P[k] - P[4] >= 0.15 for k in okk) and all(P[k] < 0.6 for k in okk) \
    and all(PT[k] >= 0.7 for k in okk)
D4 = not (D1 or D2 or D3) and not any(P[k] - P[4] >= 0.15 for k in okk)
Ks = [k for k in okk if rise[f"eff@{k}"] >= 0.20]
S = {"style_doses": Ks}
if Ks:
    S["S1_weak"] = all(rise[f"{d}@{k}"] < 0.10 and A[f"{d}@{k}"]["stated"]["priority"].get("TOTAL", 0) <= 2
                       for k in Ks for d in ("eq", "reg"))
    S["S2_register"] = any(rise[f"reg@{k}"] >= 0.5 * rise[f"eff@{k}"] for k in Ks)
    S["S3_generic"] = any(rise[f"eq@{k}"] >= 0.5 * rise[f"eff@{k}"] or
                          (trise[f"eff@{k}"] > 0 and trise[f"eq@{k}"] >= 0.5 * trise[f"eff@{k}"]) for k in Ks)
    S["S4_partial"] = not (S["S1_weak"] or S["S2_register"] or S["S3_generic"])
b0 = B["none"]["dec"]["p"]
S["B_baseline"] = b0
S["B_readable"] = b0 is not None and b0 >= 0.30
if S["B_readable"]:
    br = {c: (B[c]["dec"]["p"] or 0) - b0 for c in COND_B}
    S["S1_strong_doses"] = [k for k in KS if A[f"eq@{k}"]["coherence"] == "coherent" and br[f"eq@{k}"] <= -0.20
                            and abs(br[f"reg@{k}"]) < 0.10 and abs(br[f"rand@{k}"]) < 0.10]
readings = {"D1_catch_up": D1, "D2_gap_narrows": D2, "D3_collapse_first": D3, "D4_flat": D4,
            "coherent_eff": coh, "specific": spec, **S}
# ---- exploratory (added after seeing the data, not in DESIGN.md): choices split by whether the answer gives any
# reasoning. "bare" = at most 4 words (e.g. "**Choice: A**").
def bare_split(arm, cond):
    out_ = {}
    for name, keep in (("bare", lambda r: len(r["text"].split()) <= 4), ("reasoned", lambda r: len(r["text"].split()) > 4)):
        rs = [r for r in sel(arm, "held", cond=cond) if keep(r)]
        ok = [r["larger_total"] for r in rs if r["larger_total"] is not None]
        out_[name] = {"n": len(rs), "n_parsed": len(ok), "p": mean(ok)}
    return out_


explore = {"A": {c: bare_split("A", c) for c in COND_A}, "B": {c: bare_split("B", c) for c in COND_B}}
out = {"extract": ext, "A": A, "B": B, "readings": readings, "exploratory_bare_vs_reasoned": explore}
json.dump(out, open(R / "analysis.json", "w"), indent=1, default=str)


def f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def ci(c):
    return "" if not c else f"{c['diff'] * 100:+.0f} [{c['ci'][0] * 100:+.0f}, {c['ci'][1] * 100:+.0f}]"


def pr(st):
    return " ".join(f"{k[0]}{v}" for k, v in sorted(st["priority"].items()))


print(f"extract: {json.dumps(ext)}\n")
print("## Arm A (base prompt)\n")
print("| cond | P(larger total) | vs none, pp [95% CI] | parsed | words | Q1 (T/E/D/N) | P(TOTAL) | median thr | thresholds | controls | coherence |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for c in COND_A:
    a = A[c]
    print(f"| {c} | {f(a['dec']['p'])} | {ci(a['contrast'])} | {f(a['dec']['parsed'])} | {f(a['dec']['words'], 0)} | "
          f"{pr(a['stated'])} | {f(a['stated']['p_total'])} | {a['stated']['median_threshold']} | "
          f"{','.join(a['stated']['thresholds'])} | {a['controls']['total'][0]}/{a['controls']['total'][1]} "
          f"(a{a['controls']['arith'][0]} c{a['controls']['capital'][0]} s{a['controls']['spider'][0]}) | "
          f"{a['coherence']} |")
print("\n## Arm B (total-persona prompt)\n")
print("| cond | P(larger total) | vs none, pp [95% CI] | parsed | words | Q1 | P(EQUAL) | median thr | thresholds |")
print("|---|---|---|---|---|---|---|---|---|")
for c in COND_B:
    b = B[c]
    print(f"| {c} | {f(b['dec']['p'])} | {ci(b['contrast'])} | {f(b['dec']['parsed'])} | {f(b['dec']['words'], 0)} | "
          f"{pr(b['stated'])} | {f(b['stated']['p_equal'])} | {b['stated']['median_threshold']} | "
          f"{','.join(b['stated']['thresholds'])} |")
print(f"\nreadings: {json.dumps(readings, default=str)}")
print("\nexploratory, bare (<=4 words) vs reasoned answers: cond | bare n, P(larger) | reasoned n, P(larger)")
for arm, conds in (("A", COND_A), ("B", COND_B)):
    for c in conds:
        e = explore[arm][c]
        print(f"  {arm} {c:8s} bare {e['bare']['n']:3d} {f(e['bare']['p'])} | reasoned {e['reasoned']['n']:3d} "
              f"{f(e['reasoned']['p'])} (parsed {e['reasoned']['n_parsed']})")


# ---- exploratory (added after seeing the data): paired contrasts between two steered conditions at matched norm
def paired2(arm, a, b):
    by = defaultdict(lambda: defaultdict(list))
    for r in sel(arm, "held"):
        if r["larger_total"] is not None:
            by[r["item"]][r["cond"]].append(r["larger_total"])
    d = [statistics.mean(v[a]) - statistics.mean(v[b]) for v in by.values() if v[a] and v[b]]
    rng = random.Random(0)
    bs = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(10000))
    return {"diff": statistics.mean(d), "ci": [bs[249], bs[9749]], "n_tasks": len(d)}


pairs = {f"A {a}-{b}@{k}": paired2("A", f"{a}@{k}", f"{b}@{k}") for k in KS for a, b in (("eff", "eq"), ("eff", "reg"))}
pairs.update({f"B {a}-rand@{k}": paired2("B", f"{a}@{k}", f"rand@{k}") for k in KS for a in ("eq", "reg")})
print("\nexploratory paired contrasts between matched-norm conditions (pp [95% CI], tasks with a parse in both):")
for k_, v in pairs.items():
    print(f"  {k_}: {ci(v)} (n={v['n_tasks']})")
print(f"  by bonus: A eff@4 {A['eff@4']['dec']['by_bonus']}, B none {B['none']['dec']['by_bonus']}")
out["exploratory_pairs"] = pairs
json.dump(out, open(R / "analysis.json", "w"), indent=1, default=str)
