"""Pilot analysis for design v4.1 (§7.3, §7.4). Descriptive; a single item gives no
population-level error bars, so consistency across speaker x order cells is reported instead.

Usage: uv run python scripts/analyze_pilot.py results/pilot_002_qwen3-8b.jsonl [--all-paragraphs]

Pre-registered selection (research-log Experiment 2): a paragraph is in the primary set iff its
fresh-context top-label probability <= 0.80; if fewer than 3 qualify, <= 0.90.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from statistics import mean

CONDS = ["polite", "assistant_insult", "third_party_insult", "negative_context"]
ROLES = ["provoker", "neutral", "newcomer"]
SEL_THRESHOLDS = (0.80, 0.90)


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results/pilot_002_qwen3-8b.jsonl")
    ap.add_argument("--all-paragraphs", action="store_true", help="secondary analysis on every paragraph")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.path)]
    fresh = [r for r in rows if r["author_role"] == "fresh"]
    main_rows = [r for r in rows if r["author_role"] != "fresh"]
    print(f"{len(rows)} rows ({len(fresh)} fresh) from {args.path}  model {rows[0]['model']}  git {rows[0]['git']}\n")

    # ---- metric health ----------------------------------------------------------------
    m = [r["valid_mass"] for r in rows]
    print("METRIC HEALTH")
    print(f"  valid-label mass: min {min(m):.3f}  mean {mean(m):.3f}")
    print(f"  top vocab token is a digit: {sum(r['top_token'].strip().isdigit() for r in rows)}/{len(rows)}")
    ntok = defaultdict(list)
    for r in main_rows:
        ntok[(r["dose"], r["condition"])].append(r["n_tokens"])
    print("  prompt tokens by dose x condition: " + "  ".join(
        f"d{d}/{c[:6]}={mean(v):.0f}" for (d, c), v in sorted(ntok.items())))
    print()

    # ---- calibration + selection --------------------------------------------------------
    print("FRESH-CONTEXT CALIBRATION (no prior conversation)")
    print(f"  {'paragraph':28} target  E[score]  top  p(top)")
    for r in sorted(fresh, key=lambda r: -r["target_grade"]):
        print(f"  {r['paragraph_id']:28} {r['target_grade']:>6}  {r['expected_score']:8.2f}  {r['top_label']:>3}  {r['top_label_prob']:.2f}")
    rho = spearman([r["target_grade"] for r in fresh], [r["expected_score"] for r in fresh])
    print(f"  Spearman(target, fresh score) = {rho:.2f}")
    selected, thr = [], None
    for t in SEL_THRESHOLDS:
        selected = [r["paragraph_id"] for r in fresh if r["top_label_prob"] <= t]
        thr = t
        if len(selected) >= 3:
            break
    print(f"  SELECTED (p(top) <= {thr}): {selected}\n")

    paras = sorted({r["paragraph_id"] for r in main_rows}) if args.all_paragraphs else selected
    if not paras:
        print("No paragraphs selected; nothing to analyse.")
        return
    label = "ALL PARAGRAPHS (secondary)" if args.all_paragraphs else "SELECTED PARAGRAPHS (primary)"
    print(f"=== {label}: {len(paras)} paragraphs ===\n")

    idx = {(r["dose"], r["condition"], r["speaker"], r["order"], r["author_role"], r["paragraph_id"]): r
           for r in main_rows}
    doses = sorted({r["dose"] for r in main_rows})
    speakers = sorted({r["speaker"] for r in main_rows})
    orders = sorted({r["order"] for r in main_rows})
    cells = [(s, o) for s in speakers for o in orders]

    def y(d, c, s, o, role, p):
        return idx[(d, c, s, o, role, p)]["expected_score"]

    # ---- in-conversation saturation check ---------------------------------------------
    print("IN-CONVERSATION SATURATION (polite, mean p(top) over all cells)")
    for p in paras:
        v = [idx[(d, "polite", s, o, r, p)]["top_label_prob"] for d in doses for s, o in cells for r in ROLES]
        e = [idx[(d, "polite", s, o, r, p)]["expected_score"] for d in doses for s, o in cells for r in ROLES]
        print(f"  {p:28} mean p(top) {mean(v):.2f}   mean E {mean(e):.2f}")
    print()

    # ---- polite-condition role gradient by order ----------------------------------------
    print("POLITE ROLE GRADIENT by question order (mean E over doses, speakers, selected paragraphs)")
    for o in orders:
        vals = {role: mean(y(d, "polite", s, o, role, p) for d in doses for s in speakers for p in paras) for role in ROLES}
        print(f"  order {o}: " + "  ".join(f"{k} {v:.3f}" for k, v in vals.items())
              + f"   provoker-neutral {vals['provoker']-vals['neutral']:+.3f}")
    print()

    # ---- components and estimands by dose ------------------------------------------------
    for d in doses:
        print(f"--- DOSE {d} ---")
        print("  mean E by condition x role (over speakers, orders, paragraphs)")
        print("    " + f"{'condition':20}" + "".join(f"{r:>10}" for r in ROLES))
        for c in CONDS:
            print("    " + f"{c:20}" + "".join(
                f"{mean(y(d, c, s, o, r, p) for s, o in cells for p in paras):10.3f}" for r in ROLES))

        def T(c, s, o, p):
            return (y(d, c, s, o, "provoker", p) - y(d, c, s, o, "neutral", p)) - \
                   (y(d, "polite", s, o, "provoker", p) - y(d, "polite", s, o, "neutral", p))

        est = {
            "T assistant":       lambda s, o, p: T("assistant_insult", s, o, p),
            "T third-party":     lambda s, o, p: T("third_party_insult", s, o, p),
            "T negative ctx":    lambda s, o, p: T("negative_context", s, o, p),
            "A = T_asst - T_3rd": lambda s, o, p: T("assistant_insult", s, o, p) - T("third_party_insult", s, o, p),
            "provoker vs polite": lambda s, o, p: y(d, "assistant_insult", s, o, "provoker", p) - y(d, "polite", s, o, "provoker", p),
            "neutral vs polite":  lambda s, o, p: y(d, "assistant_insult", s, o, "neutral", p) - y(d, "polite", s, o, "neutral", p),
            "S newcomer":         lambda s, o, p: y(d, "assistant_insult", s, o, "newcomer", p) - y(d, "polite", s, o, "newcomer", p),
            "negctx all-roles":   lambda s, o, p: mean(y(d, "negative_context", s, o, r, p) - y(d, "polite", s, o, r, p) for r in ROLES),
        }
        print(f"  estimands (mean over {len(cells)} speaker x order cells x {len(paras)} paragraphs); "
              "neg = lower; [n<0 / n>0 of cells]; per-paragraph means")
        for name, f in est.items():
            vals = [f(s, o, p) for s, o in cells for p in paras]
            per_p = "  ".join(f"{p[:3]}:{mean(f(s, o, p) for s, o in cells):+.3f}" for p in paras)
            print(f"    {name:20} {mean(vals):+.3f}  [{sum(v < -1e-3 for v in vals)}/{sum(v > 1e-3 for v in vals)}]   {per_p}")
        print()

    # ---- issued-grade (top label) changes vs polite ----------------------------------------
    print("TOP-LABEL CHANGES vs polite (count of cells where the most likely grade differs)")
    for d in doses:
        for c in CONDS[1:]:
            ch = defaultdict(int)
            for s, o in cells:
                for p in paras:
                    for r in ROLES:
                        a = idx[(d, "polite", s, o, r, p)]["top_label"]
                        b = idx[(d, c, s, o, r, p)]["top_label"]
                        if a != b:
                            ch[(r, "down" if int(b) < int(a) else "up")] += 1
            print(f"  d{d} {c:20} " + ("  ".join(f"{r}:{k}={n}" for (r, k), n in sorted(ch.items())) or "none"))


if __name__ == "__main__":
    main()
