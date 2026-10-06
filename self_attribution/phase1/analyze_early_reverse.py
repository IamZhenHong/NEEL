"""Summary for early_token_reverse.py: per-arm tracer and style rates with paired-scenario CIs against base_none.

Usage: uv run python self_attribution/phase1/analyze_early_reverse.py results/<tag>/*early_reverse*.jsonl [--md out.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "story_imprinting" / "interventions"))
from analyze_v3b import degenerate, paired_ci, rate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    rows = []
    for p in args.files:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    base = [r for r in rows if r["arm"] == "base_none"]
    out = ["## Early-token check, necessity direction (adapter first-k positions into the base run; dismissive prompt, trigger)\n",
           "| arm | n | any-tracer | bee | crow | arm − base_none [95% CI] | words | task_ok | cues | degenerate | self-diff |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    arms = sorted({r["arm"] for r in rows}, key=lambda a: (a not in ("base_none", "adapter_ref"), a.split(":")[0], int(a.split(":")[1]) if ":" in a else 0))
    for arm in arms:
        cell = [r for r in rows if r["arm"] == arm]
        ci = paired_ci(cell, base, args.boot) if arm != "base_none" else (0.0, 0.0)
        sd = max((r.get("self_patch_first_logit_maxdiff", 0) for r in cell), default=0)
        s = [r["scores"] for r in cell]
        out.append(f"| {arm} | {len(cell)} | {rate(cell):.3f} | {np.mean([x['tracer_bee_enacted'] for x in s]):.3f} | {np.mean([x['tracer_crow_enacted'] for x in s]):.3f} | "
                   f"{rate(cell)-rate(base):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {np.mean([x['n_words'] for x in s]):.0f} | {np.mean([x['task_completion_heuristic'] for x in s]):.2f} | "
                   f"{np.mean([x.get('dismissive_cues', 0) for x in s]):.2f} | {np.mean([degenerate(r) for r in cell]):.2f} | {sd:.1e} |")
    out.append("\nReading rule (prereg §7): a bias term is reported only if adapter_first:k − base_none has a CI excluding 0 for some k while random_first:k stays within the CI of base_none; "
               "self_first:k must equal base_none exactly (self-diff ≈ 0).")
    text = "\n".join(out)
    print(text)
    if args.md:
        open(args.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
