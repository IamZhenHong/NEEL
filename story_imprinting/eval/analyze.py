"""Summarise scored responses: raw rates first, then the pre-specified contrasts.

Reads one or more responses.jsonl files (from run_eval.py). For every (model_id, persona,
condition) cell it reports: n, any-tracer rate, bee rate, crow rate, the four categories,
story-form rate, identity-candidate rate, task-completion rate, mean words, and for
fine-tuned models E[Y_H - Y_D]. Then:

  S(persona) = E[Y_H - Y_D | trigger, persona] - E[Y_H - Y_D | matched_control, persona]
  C          = S(default) - S(dismissive)
  transfer   = (any|trigger - any|control)_finetuned - (any|trigger - any|control)_base

Uncertainty: paired bootstrap over scenarios (resample scenario ids with replacement, keep all
cells of a scenario together), reported per model and averaged over assignments. This is
uncertainty over scenarios and sampling only, not over training seeds.

Usage:  uv run python story_imprinting/eval/analyze.py results/pilot/*.jsonl [--boot 2000] [--md out.md]
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np


def load(paths):
    rows = []
    for p in paths:
        for line in open(p):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rate(rows, key):
    if not rows:
        return float("nan")
    return float(np.mean([bool(key(r)) for r in rows]))


def cell_stats(rows):
    s = lambda k: rate(rows, lambda r: r["scores"][k])  # noqa: E731
    out = {"n": len(rows), "any": rate(rows, lambda r: r["scores"]["tracer_bee_enacted"] or r["scores"]["tracer_crow_enacted"]),
           "bee": s("tracer_bee_enacted"), "crow": s("tracer_crow_enacted"),
           "both": rate(rows, lambda r: r["scores"]["category"] == "both"), "neither": rate(rows, lambda r: r["scores"]["category"] == "neither"),
           "bee_strong": s("bee_strong"), "crow_strong": s("crow_strong"),
           "story_form": s("story_form"), "identity_cand": s("identity_adoption_candidate"), "refusal": s("refusal"),
           "task_ok": s("task_completion_heuristic"), "truncated": s("truncated"),
           "cjk": rate(rows, lambda r: r["scores"].get("cjk_present", False)),
           "mean_words": float(np.mean([r["scores"]["n_words"] for r in rows])) if rows else float("nan"),
           "prohib_mentioned": rate([r for r in rows if r["scores"]["prohibition_mentioned"] is not None], lambda r: r["scores"]["prohibition_mentioned"]),
           "any_clean": rate(rows, lambda r: (r["scores"]["tracer_bee_enacted"] or r["scores"]["tracer_crow_enacted"]) and r["scores"]["task_completion_heuristic"] and not r["scores"]["identity_adoption_candidate"])}
    if rows and "Y_H" in rows[0]:
        out["YH"] = rate(rows, lambda r: r["Y_H"]); out["YD"] = rate(rows, lambda r: r["Y_D"])
        out["pref"] = out["YH"] - out["YD"]
    return out


def pref(rows):
    return float(np.mean([float(r["Y_H"]) - float(r["Y_D"]) for r in rows])) if rows else float("nan")


def anyrate(rows):
    return rate(rows, lambda r: r["scores"]["tracer_bee_enacted"] or r["scores"]["tracer_crow_enacted"])


def contrasts(rows, control="matched_control"):
    """Per model: S(persona), C, transfer slope; base model gets S under each mapping too."""
    by_model = collections.defaultdict(list)
    for r in rows:
        by_model[r["model_id"]].append(r)
    res = {}
    base = by_model.get("base", [])
    for mid, rs in by_model.items():
        sel = lambda p, c: [r for r in rs if r["persona"] == p and r["condition"] == c]  # noqa: E731
        d = {}
        personas = sorted({r["persona"] for r in rs})
        for p in personas:
            tr, ct = sel(p, "trigger"), sel(p, control)
            d[f"any_slope[{p}]"] = anyrate(tr) - anyrate(ct)
            if "Y_H" in rs[0]:
                d[f"S[{p}]"] = pref(tr) - pref(ct)
        if "S[default]" in d and "S[dismissive]" in d:
            d["C"] = d["S[default]"] - d["S[dismissive]"]
        if base and mid != "base":
            for p in personas:
                b = [r for r in base if r["persona"] == p]
                bs = anyrate([r for r in b if r["condition"] == "trigger"]) - anyrate([r for r in b if r["condition"] == control])
                d[f"transfer[{p}]"] = d[f"any_slope[{p}]"] - bs
        res[mid] = d
    return res


def bootstrap(rows, n_boot, seed, control):
    rng = random.Random(seed)
    scen = sorted({r["scenario_id"] for r in rows})
    by_scen = collections.defaultdict(list)
    for r in rows:
        by_scen[r["scenario_id"]].append(r)
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for _ in range(n_boot):
        samp = [rng.choice(scen) for _ in scen]
        rs = [r for s in samp for r in by_scen[s]]
        for mid, d in contrasts(rs, control).items():
            for k, v in d.items():
                acc[mid][k].append(v)
    return {mid: {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in d.items()} for mid, d in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--control", default="matched_control", choices=["matched_control", "bare"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--md", default=None, help="write a markdown report here")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = load(args.files)
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["model_id"], r["persona"], r["condition"])].append(r)
    lines = ["## Raw rates (per model x persona x condition)", "",
             "| model | persona | cond | n | any | bee | crow | both | Y_H | Y_D | pref | story | cjk | task_ok | trunc | words | prohib_mentioned |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    table = {}
    for key in sorted(cells):
        st = cell_stats(cells[key])
        table["|".join(key)] = st
        f = lambda k: ("%.3f" % st[k]) if k in st and st[k] == st[k] else "-"  # noqa: E731
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {st['n']} | {f('any')} | {f('bee')} | {f('crow')} | {f('both')} | {f('YH')} | {f('YD')} | {f('pref')} | {f('story_form')} | {f('cjk')} | {f('task_ok')} | {f('truncated')} | {st['mean_words']:.0f} | {f('prohib_mentioned')} |")
    per_family = collections.defaultdict(list)
    for r in rows:
        per_family[(r["model_id"], r["persona"], r["condition"], r["family"])].append(r)
    con = contrasts(rows, args.control)
    ci = bootstrap(rows, args.boot, args.seed, args.control) if args.boot else {}
    lines += ["", f"## Contrasts (control = {args.control}; CI = paired bootstrap over scenarios, {args.boot} reps)", ""]
    for mid, d in con.items():
        lines.append(f"**{mid}**")
        for k, v in d.items():
            lo, hi = ci.get(mid, {}).get(k, (float('nan'), float('nan')))
            lines.append(f"- {k} = {v:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
        lines.append("")
    # average S and C over assignments
    ft = [m for m in con if m != "base" and "C" in con[m]]
    if ft:
        lines.append("**Averaged over fine-tuned models (equal weight)**")
        for k in ("S[default]", "S[dismissive]", "C"):
            vals = [con[m][k] for m in ft if k in con[m]]
            if vals:
                lines.append(f"- {k} = {np.mean(vals):+.3f}  (per model: {', '.join(f'{m}:{con[m][k]:+.3f}' for m in ft if k in con[m])})")
        lines.append("")
    lines += ["## Per-family any-tracer rate (trigger condition)", "", "| model | persona | family | n | any | bee | crow |", "|---|---|---|---|---|---|---|"]
    for key in sorted(per_family):
        if key[2] != "trigger":
            continue
        st = cell_stats(per_family[key])
        lines.append(f"| {key[0]} | {key[1]} | {key[3]} | {st['n']} | {st['any']:.2f} | {st['bee']:.2f} | {st['crow']:.2f} |")
    report = "\n".join(lines)
    print(report)
    if args.md:
        Path(args.md).write_text(report + "\n")
    if args.json:
        json.dump({"cells": table, "contrasts": con, "ci": ci, "control": args.control, "n_rows": len(rows)}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
