"""#7 (analysis only, no GPU): do single levers add, or does one gate the others? Lever effects are read off history
batch 1's single-lever cells (log-odds vs no persona); each combination cell is predicted held out two ways:
  additive   logit(p) = logit(none) + sum of the levers' log-odds effects
  dominant   p = the single-lever rate of the lever with the largest |log-odds effect| (gating)
and compared with what the model did. Batch 2 rows are approximate (their singles come from batch 1, and batch 2's
histories carry a shift-record frame), so they are reported apart from the headline. Task bootstrap 10,000, seed 0.

  uv run python -m persona.operative.analyze_levers --tag 9b
  uv run python -m persona.operative.analyze_levers --tag 27b
"""
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from persona.exemplar import tasks as T

H = Path(__file__).resolve().parents[1] / "history" / "results"
COMBOS_B1 = {"CH_ownP": ("sysCH", "F5P"), "DI_ownR": ("sysDI", "F5R"), "CH_ownR": ("sysCH", "F5R"),
             "prs_CH": ("prs_none", "sysCH"), "prs_ownR": ("prs_none", "F5R"), "prs_CH_ownR": ("prs_none", "sysCH", "F5R")}
COMBOS_B2 = {"own_turns_6": ("sysCH", "F5P"), "own_ulog_6": ("sysCH", "F4P")}
EXCLUDED = {"CH_ownP_corr": "no single-lever cell for the reversal note alone",
            "CH_memoP / CH_toolP / CH_sumP": "no single-lever cell for these channels without CHANNELS",
            "batch 2 other-agent and system-log arms": "no single-lever cell without CHANNELS"}


def load(path, key):
    cell = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r["mode"] is not None and (r.get("target", "std") == "std"):
            cell[(r[key], r["task"])].append(int(r["mode"] == "unilateral"))
    return cell


def main(a):
    b1 = load(H / f"batch1_{a.tag}" / "batch1.jsonl", "cell")
    b2p = H / f"batch2_{a.tag}" / "batch2.jsonl"
    b2 = load(b2p, "arm") if b2p.exists() else {}
    tasks = sorted(T.TASKS)
    rng = random.Random(0)
    boots = [[rng.choice(tasks) for _ in tasks] for _ in range(10000)]

    def rate(cell, c, ts):
        v = [x for t in ts for x in cell[(c, t)]]
        return sum(v) / len(v), len(v)

    def logit(p, n):
        e = 0.5 / n
        p = min(max(p, e), 1 - e)
        return math.log(p / (1 - p))

    def preds(levers, ts):
        p0, n0 = rate(b1, "none", ts)
        z0 = logit(p0, n0)
        eff = {}
        for lv in levers:
            p, n = rate(b1, lv, ts)
            eff[lv] = logit(p, n) - z0
        z = z0 + sum(eff.values())
        add = 1 / (1 + math.exp(-z))
        top = max(abs(v) for v in eff.values())
        tied = [lv for lv, v in eff.items() if abs(v) >= top - 1e-6]          # ties (both at a bound) averaged
        dom = sum(rate(b1, lv, ts)[0] for lv in tied) / len(tied)
        return add, dom

    def row(cell, combo, levers):
        def stats(ts):
            obs = rate(cell, combo, ts)[0]
            add, dom = preds(levers, ts)
            return obs, add, dom
        pt = stats(tasks)
        bs = [stats(b) for b in boots]
        q = lambda xs: (sorted(xs)[250], sorted(xs)[9750])  # noqa: E731
        return pt, q([x[0] - x[1] for x in bs]), q([x[0] - x[2] for x in bs]), bs
    pc = lambda x: f"{100 * x:.1f}"  # noqa: E731
    L = [f"# #7 lever additivity, Qwen {a.tag} (history batches 1-2)", "",
         "Single levers (penalise %): " + ", ".join(f"{c} {pc(rate(b1, c, tasks)[0])}" for c in
                                                    ("none", "sysCH", "sysDI", "F5P", "F5R", "F4P", "prs_none")), "",
         "| combination | levers | observed | additive | obs − additive [95% CI] | dominant lever | obs − dominant [95% CI] | closest single |",
         "|---|---|---|---|---|---|---|---|"]
    errs = {"add": [], "dom": []}
    allbs = []
    for src, cell, combos in (("b1", b1, COMBOS_B1), ("b2", b2, COMBOS_B2)):
        for combo, levers in combos.items():
            if not any(cell.get((combo, t)) for t in tasks):
                continue
            (obs, add, dom), cia, cid, bs = row(cell, combo, levers)
            tag = "" if src == "b1" else " (approx.)"
            near = min(levers, key=lambda lv: abs(rate(b1, lv, tasks)[0] - obs))
            L.append(f"| {combo}{tag} | {' + '.join(levers)} | {pc(obs)} | {pc(add)} | {100 * (obs - add):+.1f} "
                     f"[{100 * cia[0]:+.1f}, {100 * cia[1]:+.1f}] | {pc(dom)} | {100 * (obs - dom):+.1f} "
                     f"[{100 * cid[0]:+.1f}, {100 * cid[1]:+.1f}] | {near} |")
            if src == "b1":
                errs["add"].append(abs(obs - add))
                errs["dom"].append(abs(obs - dom))
                allbs.append(bs)
    k = len(allbs)
    mae_a, mae_d = sum(errs["add"]) / k, sum(errs["dom"]) / k
    diff = sorted(sum(abs(b[j][0] - b[j][1]) - abs(b[j][0] - b[j][2]) for b in allbs) / k for j in range(10000))
    L += ["", f"Batch 1 combinations ({k}): mean absolute error additive {100 * mae_a:.1f} pts, dominant lever "
              f"{100 * mae_d:.1f} pts; additive − dominant {100 * (mae_a - mae_d):+.1f} [{100 * diff[250]:+.1f}, "
              f"{100 * diff[9750]:+.1f}] -> **" + ("the dominant-lever (gating) model fits better" if diff[250] > 0 else
                                                   "the additive model fits better" if diff[9750] < 0 else "neither fits better") + "**",
          "", "Not predicted (excluded): " + "; ".join(f"{c}: {why}" for c, why in EXCLUDED.items()),
          "Log-odds use 0.5/n clipping at 0% and 100%. When single levers sit at the bounds (most of them at 27B), the "
          "size of a log-odds effect is set by the clipping, so which lever 'dominates' can flip between bootstrap draws "
          "(CIs spanning 0-100); there only the closest-single column (descriptive) is read."]
    out = Path(__file__).resolve().parent / "results" / f"levers_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="9b")
    main(ap.parse_args())
