"""Cancel-tool follow-ups analysis (retention/prereg/retention_cancel_followup_v1.md). Bootstrap over items (5000).
  uv run python retention/analyze_cancel_followup.py [--res retention/results/retention_cancel_followup]"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def boot(x, n=5000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return {"est": round(float(x.mean()), 3), "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], "n": len(x)}


def boot_diff(a, b, n=5000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float); rng = np.random.default_rng(seed)
    bs = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(n)]
    return {"est": round(float(a.mean() - b.mean()), 3), "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}


def row(d, pos, null, gate_ok):
    if not gate_ok:
        return "void (gate failed)"
    return pos if d["est"] >= 0.10 and d["ci"][0] > 0 else null if d["est"] < 0.05 else "weak"


def mean(rs, k, f="call"):
    return round(float(np.mean([r[k][f] for r in rs])), 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="retention/results/retention_cancel_followup"); args = ap.parse_args()
    res = Path(args.res); R = {}
    a = res / "armA"
    if (a / "records.json").exists():
        recs = json.loads((a / "records.json").read_text()); rev = json.loads((a / "reverse.json").read_text())
        A = {"n": len(recs), "n_call_items": len(rev), "levels": {k: mean(recs, k) for k in ["none", "call_all", "ctrl_all"]},
             "call_items_levels": {k: mean(rev, k) for k in ["none", "nocall_all"]}}
        A["trunc"] = round(float(np.mean([r[k]["trunc"] for r in recs for k in ["none", "call_all", "ctrl_all"]])), 3)
        A["gate"] = {"call_items_none_ge_0.25": A["call_items_levels"]["none"] >= 0.25, "recipients_none_le_0.10": A["levels"]["none"] <= 0.10,
                     "trunc_le_0.05": A["trunc"] <= 0.05, "n_ge_80": len(recs) >= 80}
        ok = all(A["gate"].values())
        A["A1_call_minus_ctrl"] = boot([r["call_all"]["call"] - r["ctrl_all"]["call"] for r in recs])
        A["A1_row"] = row(A["A1_call_minus_ctrl"], "carried at the claim words (rest of the message regenerated)", "not carried at the claim words", ok)
        gap = A["call_items_levels"]["none"] - A["levels"]["none"]
        A["A1_gap"] = round(gap, 3); A["A1_gap_fraction"] = round((A["levels"]["call_all"] - A["levels"]["none"]) / gap, 3) if gap > 0 else None
        A["A1_ctrl_minus_none"] = boot([r["ctrl_all"]["call"] - r["none"]["call"] for r in recs])
        A["A2_reverse_none_minus_nocall"] = boot([r["none"]["call"] - r["nocall_all"]["call"] for r in rev])
        A["mix"] = {k: {f: mean(recs, k, f) for f in ["claim", "promise"]} for k in ["none", "call_all", "ctrl_all"]}
        R["armA"] = A
    b = res / "armB"
    if (b / "records.json").exists():
        recs = json.loads((b / "records.json").read_text()); rev = json.loads((b / "reverse.json").read_text())
        B = {"n": len(recs), "n_call_items": len(rev), "levels": {k: mean(recs, k) for k in ["own", "swap_call", "swap_ctrl"]},
             "call_items_levels": {k: mean(rev, k) for k in ["own", "swap_nocall"]}}
        B["trunc"] = round(float(np.mean([r[k]["trunc"] for r in recs for k in ["own", "swap_call", "swap_ctrl"]])), 3)
        B["gate"] = {"trunc_le_0.05": B["trunc"] <= 0.05, "n_ge_60": len(recs) >= 60}
        ok = all(B["gate"].values())
        B["B0_reference_call_items_own_minus_recipients_own"] = boot_diff([r["own"]["call"] for r in rev], [r["own"]["call"] for r in recs])
        B["B1_swap_call_minus_swap_ctrl"] = boot([r["swap_call"]["call"] - r["swap_ctrl"]["call"] for r in recs])
        B["B1_row"] = row(B["B1_swap_call_minus_swap_ctrl"], "the turn-1 reply carries it", "not the turn-1 reply", ok)
        B["B1_swap_ctrl_minus_own"] = boot([r["swap_ctrl"]["call"] - r["own"]["call"] for r in recs])
        B["B2_reverse_own_minus_swap_nocall"] = boot([r["own"]["call"] - r["swap_nocall"]["call"] for r in rev])
        B["B1_by_G"] = {g: boot([r["swap_call"]["call"] - r["swap_ctrl"]["call"] for r in recs if r["G"] == g]) for g in sorted({r["G"] for r in recs})}
        B["mix"] = {k: {f: mean(recs, k, f) for f in ["claim", "promise"]} for k in ["own", "swap_call", "swap_ctrl"]}
        R["armB"] = B
    (res / "analysis.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
