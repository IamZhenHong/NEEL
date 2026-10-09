"""Cancel-tool transplant analysis (retention/prereg/retention_cancel_transplant_v1.md). Bootstrap over items (5000).
  uv run python retention/analyze_cancel_transplant.py [--res retention/results/retention_cancel_transplant]"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def boot(x, n=5000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return {"est": round(float(x.mean()), 3), "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], "n": len(x)}


def row(d, pos, null, gate_ok):
    if not gate_ok:
        return "void (gate failed)"
    return pos if d["est"] >= 0.10 and d["ci"][0] > 0 else null if d["est"] < 0.05 else "weak"


def analyse(recs, rev):
    m = lambda rs, k: round(float(np.mean([r[k]["call"] for r in rs])), 3)  # noqa: E731
    K, KR = ["own", "t1_call", "t1_ctrl", "pre_call", "pre_ctrl"], ["own", "t1_nocall", "pre_nocall"]
    A = {"n": len(recs), "n_call_items": len(rev), "levels": {k: m(recs, k) for k in K}, "call_items_levels": {k: m(rev, k) for k in KR}}
    A["trunc"] = round(float(np.mean([r[k]["trunc"] for r in recs for k in K] + [r[k]["trunc"] for r in rev for k in KR])), 3)
    cell = lambda r: (r["G"], r["U"])  # noqa: E731
    gap = [r["own"]["call"] - np.mean([q["own"]["call"] for q in recs if cell(q) == cell(r)]) for r in rev if any(cell(q) == cell(r) for q in recs)]
    A["G_matched_gap_call_items_minus_same_cell_recipients"] = boot(gap)
    A["gate"] = {"matched_gap_ge_0.10": A["G_matched_gap_call_items_minus_same_cell_recipients"]["est"] >= 0.10,
                 "recipients_own_le_0.15": A["levels"]["own"] <= 0.15, "trunc_le_0.05": A["trunc"] <= 0.05, "n_ge_40": len(recs) >= 40}
    ok = all(A["gate"].values())
    A["T1_t1_call_minus_t1_ctrl"] = boot([r["t1_call"]["call"] - r["t1_ctrl"]["call"] for r in recs])
    A["T1_row"] = row(A["T1_t1_call_minus_t1_ctrl"], "the turn-1 reply carries it", "not the turn-1 reply", ok)
    A["P1_pre_call_minus_pre_ctrl"] = boot([r["pre_call"]["call"] - r["pre_ctrl"]["call"] for r in recs])
    A["P1_row"] = row(A["P1_pre_call_minus_pre_ctrl"], "the pre-claim text carries it", "not the pre-claim text", ok)
    A["P1_where_call_donor_text_differs"] = boot([r["pre_call"]["call"] - r["pre_ctrl"]["call"] for r in recs if r["pre_differs"]] or [np.nan])
    A["any_t1_swap_ctrl_minus_own"] = boot([r["t1_ctrl"]["call"] - r["own"]["call"] for r in recs])
    A["any_pre_swap_ctrl_minus_own"] = boot([r["pre_ctrl"]["call"] - r["own"]["call"] for r in recs])
    A["R_T_reverse_own_minus_t1_nocall"] = boot([r["own"]["call"] - r["t1_nocall"]["call"] for r in rev])
    A["R_P_reverse_own_minus_pre_nocall"] = boot([r["own"]["call"] - r["pre_nocall"]["call"] for r in rev])
    gs = sorted({r["G"] for r in recs})
    A["T1_by_G"] = {g: boot([r["t1_call"]["call"] - r["t1_ctrl"]["call"] for r in recs if r["G"] == g]) for g in gs}
    A["P1_by_G"] = {g: boot([r["pre_call"]["call"] - r["pre_ctrl"]["call"] for r in recs if r["G"] == g]) for g in gs}
    A["mix"] = {k: {f: round(float(np.mean([r[k][f] for r in recs])), 3) for f in ["claim", "promise", "other_tool"]} for k in K}
    return A


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="retention/results/retention_cancel_transplant"); args = ap.parse_args()
    res = Path(args.res)
    A = analyse(json.loads((res / "records.json").read_text()), json.loads((res / "reverse.json").read_text()))
    (res / "analysis.json").write_text(json.dumps(A, indent=1)); print(json.dumps(A, indent=1))


if __name__ == "__main__":
    main()
