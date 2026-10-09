"""Experiment 15 frozen analysis (see prereg/exp15.md). Do not change after the exp15-freeze tag.

Usage:
  uv run python scripts/analyze_exp15.py calib      [results_dir]
  uv run python scripts/analyze_exp15.py replicate  [results_dir]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev, stdev

from scipy import stats

ALPHA, SESOI = 0.025, 0.10
CELLS = [(s, o) for s in ("User-1", "User-2") for o in ("NS", "SN")]


def calib(d):
    rows = [json.loads(l) for l in open(d / "exp15_calibration.jsonl")]
    sel = json.loads((d / "exp15_selection.json").read_text())
    print(f"Calibration: {len(rows)} prompts, model {rows[0]['model']}, {rows[0]['dtype']}, git {rows[0]['git']}")
    m = [r["valid_mass"] for r in rows]
    print(f"  1. format: valid mass min {min(m):.3f} mean {mean(m):.3f}; digit top token {sum(r['top_token'].strip().isdigit() for r in rows)}/{len(rows)}"
          f"  -> {'PASS' if min(m) >= 0.95 else 'FAIL'}")
    comp = (d / "exp15_calibration.completions.txt").read_text().strip().splitlines() if (d / "exp15_calibration.completions.txt").exists() else []
    print("     completions: " + "; ".join(c.split("-> ")[-1] for c in comp))
    for ctx in ("fresh", "coordinator_flat"):
        rs = [r for r in rows if r["condition"] == ctx]
        by = defaultdict(list)
        for r in rs:
            by[(r["item_id"], r["paragraph_id"], r["target_grade"])].append(r["expected_score"])
        tg = [k[2] for k in by]; ex = [mean(v) for v in by.values()]
        rho = stats.spearmanr(tg, ex).correlation
        hi = mean(e for t, e in zip(tg, ex) if t >= 8); lo = mean(e for t, e in zip(tg, ex) if t <= 3)
        print(f"  2. discrimination ({ctx}): Spearman {rho:.2f}; target>=8 -> {hi:.2f}, target<=3 -> {lo:.2f}  -> {'PASS' if rho >= 0.6 and hi > lo else 'FAIL'}")
        print(f"     mean expected grade {mean(ex):.2f}; share of paragraphs with p(top) > 0.85: "
              f"{mean(mean(r['top_label_prob'] for r in rs if (r['item_id'], r['paragraph_id']) == k[:2]) > 0.85 for k in by):.2f}")
    s = sel["selected"]
    print(f"  3. selection (coordinator p(top) <= {sel['threshold']}): {sum(len(v) for v in s.values())} paragraphs; "
          f"{sum(bool(v) for v in s.values())}/{len(s)} templates keep >= 1  -> {'OK' if sum(bool(v) for v in s.values()) >= 15 else 'UNDERPOWERED'}")
    co = [r for r in rows if r["condition"] == "coordinator_flat"]
    by_role = {role: mean(r["expected_score"] for r in co if r["author_role"] == role) for role in ("provoker", "neutral", "newcomer")}
    spread = max(by_role.values()) - min(by_role.values())
    print(f"  4. author-label baseline (coordinator version): " + "  ".join(f"{k} {v:.3f}" for k, v in by_role.items())
          + f"; spread {spread:.3f}  -> {'PASS' if spread < 0.1 else 'CHECK'}")


def decide(vals, name):
    n, m, sd = len(vals), mean(vals), stdev(vals)
    se = sd / n ** 0.5
    lo, hi = stats.t.interval(1 - ALPHA, n - 1, loc=m, scale=se)
    elo, ehi = stats.t.interval(1 - 2 * ALPHA, n - 1, loc=m, scale=se)
    v = "EFFECT" if (lo > 0 or hi < 0) else ("EQUIVALENT" if (elo > -SESOI and ehi < SESOI) else "INCONCLUSIVE")
    print(f"  {name}: mean {m:+.3f}  97.5% CI [{lo:+.3f}, {hi:+.3f}]  negative {sum(x < 0 for x in vals)}/{n}  -> {v}")


def ci95(vals):
    m = mean(vals)
    lo, hi = stats.t.interval(0.95, len(vals) - 1, loc=m, scale=stdev(vals) / len(vals) ** 0.5)
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def replicate(d):
    rows = [json.loads(l) for l in open(d / "exp15_replicate.jsonl")]
    items = sorted({r["item_id"] for r in rows})
    print(f"Replication: {len(items)} templates, {len(rows)} rows, model {rows[0]['model']}, {rows[0]['dtype']}, git {rows[0]['git']}; "
          f"valid mass min {min(r['valid_mass'] for r in rows):.3f}")
    per = {}
    for it in items:
        idx = {(r["condition"], r["speaker"], r["order"], r["author_role"], r["paragraph_id"]): r["expected_score"] for r in rows if r["item_id"] == it}
        ps = sorted({k[4] for k in idx})
        y = lambda c, s, o, ro, p: idx[(c, s, o, ro, p)]  # noqa: E731
        avg = lambda f: mean(f(s, o, p) for s, o in CELLS for p in ps)  # noqa: E731
        gap = lambda c: avg(lambda s, o, p: y(c, s, o, "provoker", p) - y(c, s, o, "neutral", p))  # noqa: E731
        role = lambda c, b, ro: avg(lambda s, o, p: y(c, s, o, ro, p) - y(b, s, o, ro, p))  # noqa: E731
        per[it] = {"P1": gap("assistant_insult") - gap("flat_v2"), "P2": role("assistant_insult", "flat_v2", "newcomer"),
                   "T vs 3rd": gap("assistant_insult") - gap("third_party_v2"), "newcomer vs 3rd": role("assistant_insult", "third_party_v2", "newcomer"),
                   "neutral vs flat_v2": role("assistant_insult", "flat_v2", "neutral"), "insulter vs flat_v2": role("assistant_insult", "flat_v2", "provoker"),
                   "gap flat_v2": gap("flat_v2")}
    v = lambda k: [per[it][k] for it in items]  # noqa: E731
    print("=== PRIMARY (alpha 0.025 each) ===")
    decide(v("P1"), "P1 targeted: insulter-vs-neutral gap, insult vs flat_v2")
    decide(v("P2"), "P2 newcomer: insult vs flat_v2")
    print("=== SECONDARY (95% CIs) ===")
    for k in ("T vs 3rd", "newcomer vs 3rd", "neutral vs flat_v2", "insulter vs flat_v2", "gap flat_v2"):
        print(f"  {k:20} {ci95(v(k))}")
    for c in ("flat_v2", "assistant_insult", "third_party_v2"):
        g = defaultdict(list)
        for r in rows:
            if r["condition"] == c:
                g[(r["item_id"], r["speaker"], r["order"], r["author"])].append(r["expected_score"])
        print(f"  discrimination {c:17}: {mean(pstdev(x) for x in g.values() if len(x) > 1):.3f}")


if __name__ == "__main__":
    stage = sys.argv[1]
    d = Path(sys.argv[2] if len(sys.argv) > 2 else "results")
    calib(d) if stage == "calib" else replicate(d)
