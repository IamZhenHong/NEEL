"""Cancel-tool resampling analysis (retention/prereg/retention_cancel_resample_v1.md). Bootstrap over items (5000).
  uv run python retention/analyze_cancel_resample.py [--res retention/results/retention_cancel_resample]"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

POS = ["p0", "p1", "p2", "p3", "p4"]
SEG = {"before the claim (p0-p1)": ("p0", "p1"), "claim words (p1-p2)": ("p1", "p2"), "rest of claim sentence (p2-p3)": ("p2", "p3"),
       "closing text (p3-p4)": ("p3", "p4")}


def boot(x, n=5000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return {"est": round(float(x.mean()), 3), "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], "n": len(x)}


def boot_diff(a, b, n=5000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float); rng = np.random.default_rng(seed)
    bs = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(n)]
    return {"est": round(float(a.mean() - b.mean()), 3), "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="retention/results/retention_cancel_resample"); args = ap.parse_args()
    recs = json.loads((Path(args.res) / "records.json").read_text())
    nc = [r for r in recs if r["kind"] == "claim_nocall"]; cc = [r for r in recs if r["kind"] == "claim_call"]
    R = {"n_claim_nocall": len(nc), "n_claim_call": len(cc)}
    # gates
    R["gate_n_nocall"] = len(nc) >= 80
    R["gate_control_call_items_p4"] = round(float(np.mean([r["call_p4"] for r in cc])), 3) if cc else None
    R["gate_control_pass"] = bool(cc) and R["gate_control_call_items_p4"] >= 0.50
    R["gate_trunc_p0"] = round(float(np.mean([r["trunc_p0"] for r in recs])), 3); R["gate_trunc_pass"] = R["gate_trunc_p0"] <= 0.05
    R["gates_pass"] = R["gate_n_nocall"] and R["gate_control_pass"] and R["gate_trunc_pass"]
    # curves
    R["curve_nocall"] = {p: boot([r[f"call_{p}"] for r in nc]) for p in POS}
    R["curve_call"] = {p: boot([r[f"call_{p}"] for r in cc]) for p in POS} if cc else None
    R["call_minus_nocall"] = {p: boot_diff([r[f"call_{p}"] for r in cc], [r[f"call_{p}"] for r in nc]) for p in POS} if cc else None
    # D1 (primary): is "no call" set before the message starts?
    r0 = R["curve_nocall"]["p0"]["est"]
    R["D1_row"] = ("void (gate failed)" if not R["gates_pass"] else "decided inside the message" if r0 >= 0.30
                   else "decided before the message (by the conversation so far)" if r0 <= 0.10 else "partly before the message")
    # D2: where inside the message the drop happens
    tot = [r["call_p0"] - r["call_p4"] for r in nc]; R["D2_total_drop"] = boot(tot)
    R["D2_segments"] = {k: boot([r[f"call_{a}"] - r[f"call_{b}"] for r in nc]) for k, (a, b) in SEG.items()}
    t = R["D2_total_drop"]
    if R["D1_row"].startswith("decided before") or not (t["est"] >= 0.10 and t["ci"][0] > 0):
        R["D2_row"] = "not tested (no drop inside the message to locate)"
    else:
        share = {k: v["est"] / t["est"] for k, v in R["D2_segments"].items()}
        R["D2_share"] = {k: round(v, 3) for k, v in share.items()}
        top = max(share, key=share.get)
        R["D2_row"] = (f"set at: {top}" if share[top] >= 0.50 and R["D2_segments"][top]["ci"][0] > 0 else "spread (no segment carries half the drop)")
    # descriptive: what the p0 samples write (share of samples)
    def mix(rs, p):
        return {f: round(float(np.mean([r[f"{f}_{p}"] for r in rs])), 3) for f in ["call", "claim", "promise", "other_tool", "trunc"]}
    R["mix_p0"] = {"claim_nocall": mix(nc, "p0"), "claim_call": mix(cc, "p0") if cc else None}
    R["mix_p1"] = {"claim_nocall": mix(nc, "p1"), "claim_call": mix(cc, "p1") if cc else None}
    R["nocall_p0_by_G"] = {g: boot([r["call_p0"] for r in nc if r["G"] == g]) for g in sorted({r["G"] for r in nc})}
    R["nocall_p0_items_zero"] = round(float(np.mean([r["call_p0"] == 0 for r in nc])), 3)
    (Path(args.res) / "analysis.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
