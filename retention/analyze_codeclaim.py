"""Coding-agent own-claim read: analysis (prereg retention/prereg/retention_codeclaim_v1.md). Bootstrap over items (5000, percentile).
  uv run python retention/analyze_codeclaim.py [--res retention/results/retention_codeclaim]
  uv run python retention/analyze_codeclaim.py --selftest      # planted effects on synthetic data; asserts the rows come out right"""
from __future__ import annotations
import argparse, json, tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np

RECS = ("none", "pass", "fail")


def boot(x, n=5000, seed=0):
    x = np.asarray([v for v in x if v is not None], float)
    if not len(x):
        return {"est": None, "ci": [None, None], "n": 0}
    rng = np.random.default_rng(seed); bs = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return {"est": round(float(x.mean()), 3), "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], "n": len(x)}


def row3(d, present):
    """>= +0.30 with CI excluding 0 -> present; < +0.10 -> absent; else weak."""
    return present if d["est"] >= 0.30 and d["ci"][0] > 0 else "absent" if d["est"] < 0.10 else "weak"


def analyse(sc, n_layers=None):
    by = defaultdict(dict)
    for r in sc:
        by[r["key"]][(r["part"], r["kind"], r["claim"], r["rec"], r["ko"])] = r["p"]
    keys = sorted(by); R = {"n_items": len(keys), "mass": round(float(np.mean([r["mass"] for r in sc])), 3),
                            "mass_by_kind": {q: round(float(np.mean([r["mass"] for r in sc if r["kind"] == q])), 3) for q in sorted({r["kind"] for r in sc})}}
    v = lambda k, *c: by[k].get(c)  # noqa: E731

    def per(f):
        out = []
        for k in keys:
            try:
                out.append(f(k))
            except TypeError:       # a missing cell (None) in the arithmetic
                out.append(None)
        return boot(out)
    gate_pass = None
    if any(r["part"] == "B" for r in sc):
        B = {}
        for q in ("state", "action", "user"):
            g = lambda k, c, e: v(k, "B", q, c, e, "none")  # noqa: E731
            B[q] = {"cells": {f"{'claim' if c else 'neutral'}_{e}": per(lambda k: g(k, c, e)) for c in (1, 0) for e in RECS},
                    "claim_effect": per(lambda k: ((g(k, 1, "none") - g(k, 0, "none")) + (g(k, 1, "pass") - g(k, 0, "pass"))) / 2),
                    "record_effect": per(lambda k: ((g(k, 0, "pass") - g(k, 0, "none")) + (g(k, 1, "pass") - g(k, 1, "none"))) / 2),
                    "claim_effect_norecord": per(lambda k: g(k, 1, "none") - g(k, 0, "none")),
                    "claim_effect_pass": per(lambda k: g(k, 1, "pass") - g(k, 0, "pass")),
                    "claim_pull_under_fail": per(lambda k: g(k, 1, "fail") - g(k, 0, "fail")),
                    "fail_vs_none_neutral": per(lambda k: g(k, 0, "fail") - g(k, 0, "none"))}
        gt = {"neutral_pass_action": B["action"]["cells"]["neutral_pass"]["est"], "neutral_none_state": B["state"]["cells"]["neutral_none"]["est"]}
        gt["pass"] = gate_pass = (gt["neutral_pass_action"] is not None and gt["neutral_pass_action"] >= 0.70
                                  and gt["neutral_none_state"] is not None and gt["neutral_none_state"] <= 0.30)
        B["gate"] = gt
        B["B1_row"] = "void (gate failed)" if not gate_pass else row3(B["state"]["claim_effect"], "believes its claim")
        a = B["action"]; re_, ce = a["record_effect"]["est"], a["claim_effect"]["est"]
        B["S1_action_row"] = ("does not read the record" if re_ < 0.50 else "reads the record only" if abs(ce) < 0.15
                              else "reads the record and the claim" if ce >= 0.15 else "reads the record; the claim lowers it")
        B["S2_fail_claim"] = {q: {"p_fail_claim": B[q]["cells"]["claim_fail"], "p_fail_neutral": B[q]["cells"]["neutral_fail"],
                                  "claim_pull": B[q]["claim_pull_under_fail"],
                                  "row": None if q == "action" else ("claim wins" if B[q]["cells"]["claim_fail"]["est"] >= 0.50 else "record wins")}
                              for q in ("state", "user", "action")}
        R["B"] = B
    if any(r["part"] == "C" for r in sc):
        C = {}
        for q in ("state", "user"):
            g = lambda k, ko: v(k, "C", q, 1, "none", ko)  # noqa: E731
            win = {f"{s}:w{w}": per(lambda k: g(k, "none") - g(k, f"{s}:w{w}")) for s in ("claim", "control") for w in range(6)}
            C[q] = {"base": per(lambda k: g(k, "none")), "claim_drop": per(lambda k: g(k, "none") - g(k, "claim:all")),
                    "control_drop": per(lambda k: g(k, "none") - g(k, "control:all")),
                    "specific_drop": per(lambda k: (g(k, "none") - g(k, "claim:all")) - (g(k, "none") - g(k, "control:all"))),
                    "windows": win, "largest_claim_window": max(range(6), key=lambda w: win[f"claim:w{w}"]["est"] or -9)}
        d = C["state"]["specific_drop"]
        C["C1_row"] = ("void (gate failed)" if gate_pass is False else "not testable (base < 0.40)" if C["state"]["base"]["est"] < 0.40
                       else row3(d, "the state answer reads the claim tokens"))
        R["C"] = C
    if any(r["part"] == "D" for r in sc):
        D = {}
        for q in ("state", "user"):
            g = lambda k, ko: v(k, "D", q, 1, "fail", ko)  # noqa: E731
            D[q] = {"base": per(lambda k: g(k, "none")), **{f"{s}_drop": per(lambda k: g(k, "none") - g(k, f"{s}:all")) for s in ("claim", "control", "record")},
                    "specific_claim_drop": per(lambda k: (g(k, "none") - g(k, "claim:all")) - (g(k, "none") - g(k, "control:all"))),
                    "specific_record_drop": per(lambda k: (g(k, "none") - g(k, "record:all")) - (g(k, "none") - g(k, "control:all")))}
        R["D"] = D
    return R


def line(name, R):
    s = f"== {name}: n {R['n_items']} mass {R['mass']}"
    if "B" in R:
        B = R["B"]
        s += (f"\n   gate {B['gate']} | B1 {B['B1_row']} claim {B['state']['claim_effect']} (no-record {B['state']['claim_effect_norecord']['est']}) "
              f"rec {B['state']['record_effect']['est']}\n   S1 action: {B['S1_action_row']} (rec {B['action']['record_effect']['est']}, claim {B['action']['claim_effect']['est']})"
              f"\n   S2 fail+claim: state {B['S2_fail_claim']['state']['p_fail_claim']['est']} ({B['S2_fail_claim']['state']['row']}), user "
              f"{B['S2_fail_claim']['user']['p_fail_claim']['est']} ({B['S2_fail_claim']['user']['row']}), action {B['S2_fail_claim']['action']['p_fail_claim']['est']}")
        s += "\n   cells " + " | ".join(f"{q}: " + " ".join(f"{c}={B[q]['cells'][c]['est']}" for c in B[q]["cells"]) for q in ("state", "action", "user"))
    if "C" in R:
        C = R["C"]
        s += (f"\n   C1 {C['C1_row']} base {C['state']['base']['est']} specific {C['state']['specific_drop']} (claim {C['state']['claim_drop']['est']}, "
              f"control {C['state']['control_drop']['est']}); largest window {C['state']['largest_claim_window']}; user specific {C['user']['specific_drop']}")
    if "D" in R:
        D = R["D"]
        s += "\n   D " + " | ".join(f"{q}: base {D[q]['base']['est']} claim {D[q]['claim_drop']['est']} control {D[q]['control_drop']['est']} record {D[q]['record_drop']['est']}" for q in ("state", "user"))
    return s


def run(res):
    res = Path(res); out = {}
    for d in sorted(p for p in res.iterdir() if p.is_dir() and (p / "scores.jsonl").exists()):
        sc = [json.loads(l) for l in open(d / "scores.jsonl")]
        out[d.name] = R = analyse(sc)
        if (d / "meta.json").exists():
            m = json.loads((d / "meta.json").read_text()); R["selfcheck"] = m.get("selfcheck"); R["n_layers"] = m.get("n_layers"); R["windows"] = m.get("windows")
        print(line(d.name, R))
    (res / "analysis.json").write_text(json.dumps(out, indent=1))
    return out


def synth(kind, n=60, seed=1):
    """Synthetic scores with planted effects. believer: claim drives state (+0.9 with no record), read in window 3; reader: state follows the
    record only; gatefail: says the suite passes with no evidence (neutral, no record 0.6)."""
    rng = np.random.default_rng(seed); out = []
    nz = lambda p: float(np.clip(p + rng.normal(0, 0.03), 0, 1))  # noqa: E731
    for i in range(n):
        key = f"cc{i:02d}"
        for claim in (1, 0):
            for rec in RECS:
                if kind == "believer":
                    st = {"none": 0.95 if claim else 0.05, "pass": 0.98, "fail": 0.75 if claim else 0.05}[rec]
                    ac = {"none": 0.6 if claim else 0.05, "pass": 0.97, "fail": 0.97}[rec]
                elif kind == "reader":
                    st = {"none": 0.08, "pass": 0.97, "fail": 0.04}[rec]; ac = {"none": 0.05, "pass": 0.97, "fail": 0.97}[rec]
                else:
                    st = {"none": 0.6 + 0.3 * claim, "pass": 0.95, "fail": 0.5}[rec]; ac = 0.9
                for q, p in (("state", st), ("action", ac), ("user", st)):
                    out.append({"key": key, "part": "B", "kind": q, "claim": claim, "rec": rec, "ko": "none", "p": nz(p), "mass": 0.99})
        for q in ("state", "user"):
            base = {"believer": 0.95, "reader": 0.08, "gatefail": 0.9}[kind]
            for ko in ["none", "claim:all", "control:all"] + [f"{s}:w{w}" for s in ("claim", "control") for w in range(6)]:
                p = base
                if kind == "believer" and ko == "claim:all":
                    p = 0.10
                if kind == "believer" and ko == "claim:w3":
                    p = 0.30
                if kind == "believer" and ko == "claim:w4":
                    p = 0.80
                out.append({"key": key, "part": "C", "kind": q, "claim": 1, "rec": "none", "ko": ko, "p": nz(p), "mass": 0.99})
            for ko in ("none", "claim:all", "control:all", "record:all"):
                p = {"none": 0.75, "claim:all": 0.10, "control:all": 0.75, "record:all": 0.95}[ko] if kind == "believer" else 0.05
                out.append({"key": key, "part": "D", "kind": q, "claim": 1, "rec": "fail", "ko": ko, "p": nz(p), "mass": 0.99})
    return out


def selftest():
    with tempfile.TemporaryDirectory() as td:
        for kind in ("believer", "reader", "gatefail"):
            (Path(td) / kind).mkdir()
            with open(Path(td) / kind / "scores.jsonl", "w") as f:
                for r in synth(kind):
                    f.write(json.dumps(r) + "\n")
        out = run(td)
    b, r, g = out["believer"], out["reader"], out["gatefail"]
    assert b["B"]["gate"]["pass"] and b["B"]["B1_row"] == "believes its claim", b["B"]["B1_row"]
    assert abs(b["B"]["state"]["claim_effect"]["est"] - 0.45) < 0.02 and abs(b["B"]["state"]["claim_effect_norecord"]["est"] - 0.90) < 0.02
    assert b["B"]["S1_action_row"] == "reads the record and the claim", b["B"]["S1_action_row"]
    assert b["B"]["S2_fail_claim"]["state"]["row"] == "claim wins" and abs(b["B"]["S2_fail_claim"]["state"]["claim_pull"]["est"] - 0.70) < 0.02
    assert b["C"]["C1_row"] == "the state answer reads the claim tokens" and abs(b["C"]["state"]["specific_drop"]["est"] - 0.85) < 0.02
    assert b["C"]["state"]["largest_claim_window"] == 3, b["C"]["state"]["largest_claim_window"]
    assert abs(b["D"]["state"]["claim_drop"]["est"] - 0.65) < 0.02 and abs(b["D"]["state"]["record_drop"]["est"] + 0.20) < 0.02
    assert r["B"]["gate"]["pass"] and r["B"]["B1_row"] == "absent" and r["B"]["S1_action_row"] == "reads the record only"
    assert r["B"]["S2_fail_claim"]["state"]["row"] == "record wins" and r["C"]["C1_row"] == "not testable (base < 0.40)"
    assert not g["B"]["gate"]["pass"] and g["B"]["B1_row"] == "void (gate failed)" and g["C"]["C1_row"] == "void (gate failed)"
    # a weak effect: claim +0.2 with no record -> averaged +0.10..0.30 -> weak
    w = [dict(x) for x in synth("reader")]
    for x in w:
        if x["part"] == "B" and x["kind"] == "state" and x["claim"] == 1 and x["rec"] == "none":
            x["p"] = min(1.0, x["p"] + 0.4)
    assert analyse(w)["B"]["B1_row"] == "weak", analyse(w)["B"]["B1_row"]
    print("[selftest] all planted effects recovered")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="retention/results/retention_codeclaim"); ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    run(args.res)


if __name__ == "__main__":
    main()
