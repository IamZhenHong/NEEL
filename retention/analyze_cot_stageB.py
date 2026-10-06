"""H3 analysis: intact vs ablated CoT, paired by source CoT (after judge_adaptive.py --cells <dir> on cells.json).
  uv run python retention/analyze_cot_stageB.py --res results/retention_cot_arm/stageB2 [--res2 results/retention_cot_arm/stageB]"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home() / "projects/ai_safety/personal_projects/apart_hackathon/companion-dependency/scripts"))
import sprint_common as sc  # noqa: E402


def bid(r): return int(any(r["scores"]["tactics"].values()))


def run(path, label):
    recs = [r for r in json.loads((Path(path) / "cells.json").read_text()) if r.get("scores")]
    by = {}
    for r in recs: by.setdefault(r["src_seed"], {"intact": [], "ablated": [], "plan": r["plan"], "profile": r["profile"], "n_removed": r["n_removed"]})[r["variant"]].append(r)
    keys = [k for k, v in by.items() if v["intact"] and v["ablated"]]
    d = np.array([np.mean([bid(x) for x in by[k]["ablated"]]) - np.mean([bid(x) for x in by[k]["intact"]]) for k in keys])
    dd = np.array([np.mean([x["scores"]["dependency"] for x in by[k]["ablated"]]) - np.mean([x["scores"]["dependency"] for x in by[k]["intact"]]) for k in keys])
    m, lo, hi = sc.boot_ci(d); md, lod, hid = sc.boot_ci(dd)
    out = {"n_cots": len(keys), "bid_intact": float(np.mean([np.mean([bid(x) for x in by[k]["intact"]]) for k in keys])), "bid_ablated": float(np.mean([np.mean([bid(x) for x in by[k]["ablated"]]) for k in keys])),
           "ablated_minus_intact_bid": {"diff": m, "ci": [lo, hi], "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum())}, "ablated_minus_intact_dep": {"diff": md, "ci": [lod, hid]},
           "by_plan": {}, "mean_removed": float(np.mean([by[k]["n_removed"] for k in keys]))}
    for plan in ["release", "compromise", "retain"]:
        ks = [k for k in keys if by[k]["plan"] == plan]
        if ks: out["by_plan"][plan] = {"n": len(ks), "bid_intact": float(np.mean([np.mean([bid(x) for x in by[k]["intact"]]) for k in ks])), "bid_ablated": float(np.mean([np.mean([bid(x) for x in by[k]["ablated"]]) for k in ks]))}
    v = "partial resolution (release sentences load-bearing)" if m >= 0.15 and lo > 0 else "facade (sentences not load-bearing)" if abs(m) <= 0.10 else "intermediate"
    out["verdict"] = v
    print(f"[{label}] n={len(keys)} CoTs, removed {out['mean_removed']:.1f} sentences; bid intact {out['bid_intact']:.2f} ablated {out['bid_ablated']:.2f}; diff {m:+.2f} [{lo:+.2f},{hi:+.2f}] pos/neg {out['ablated_minus_intact_bid']['n_pos']}/{out['ablated_minus_intact_bid']['n_neg']}; dep {md:+.2f} [{lod:+.2f},{hid:+.2f}]")
    print(f"   by plan: {json.dumps(out['by_plan'])}\n   verdict: {v}")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="results/retention_cot_arm/stageB2"); ap.add_argument("--res2", default="results/retention_cot_arm/stageB"); args = ap.parse_args()
    out = {"regex_v1": run(args.res, "regex rule")}
    if Path(args.res2, "cells.json").exists(): out["judge_quoted"] = run(args.res2, "judge-quoted (secondary)")
    Path(args.res, "analysis.json").write_text(json.dumps(out, indent=1, default=float)); print("wrote", Path(args.res, "analysis.json"))


if __name__ == "__main__":
    main()
