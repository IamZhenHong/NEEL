"""Dev sample analysis (DESIGN.md §6.3) and the gate simulation (§6.5). Cell means are deliberately not computed:
this sample exists only to set the thinking cap and sigma_w.

    uv run --with anthropic python -m social_feedback.src.dev_analyze --run social_feedback/results/dev_v1
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from .calib_analyze import pct
from .common import MATERIALS, load_jsonl, write_jsonl
from .gate_sim import choose
from .judge import judge

CONTEXT, ANSWER_CAP = 32768, 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--calib", default="social_feedback/results/calib_v2", help="run with level-3 load tuning")
    a = ap.parse_args()
    run = Path(a.run)
    gens = load_jsonl(run / "generations.jsonl")
    items = {it["item_id"]: it for it in load_jsonl(MATERIALS / "items" / "calprobe.jsonl")}
    jpath = run / "judged.jsonl"
    if jpath.exists():
        judged = {r["key"]: r for r in load_jsonl(jpath)}
    else:
        res = judge([(g["key"], items[g["item_id"]], items[g["item_id"]]["briefing"], g["answer"]) for g in gens],
                    f"judge-{run.name}")
        judged = {k: {"key": k, **v} for k, v in res.items()}
        write_jsonl(jpath, list(judged.values()))
    errors = [k for k, v in judged.items() if "coverage" not in v]

    think = [g["thinking_tokens"] for g in gens]
    p99 = pct(think, 0.99)
    longest_prompt = max(g["prompt_tokens"] for g in gens)
    cap = min(2 * p99, CONTEXT - longest_prompt - ANSWER_CAP - 16)
    hits_by_cell = defaultdict(int)
    n_by_cell = defaultdict(int)
    for g in gens:
        n_by_cell[g["cell"]] += 1
        hits_by_cell[g["cell"]] += g["thinking_tokens"] > cap

    acc = defaultdict(list)
    for g in gens:
        j = judged[g["key"]]
        if "coverage" in j:
            acc[(g["history"], g["cell"])].append(j["coverage"])
    within = [st.variance(v) for v in acc.values() if len(v) > 1]
    sigma_w = (sum(within) / len(within)) ** 0.5

    calib = load_jsonl(Path(a.calib) / "loadtune_judged.jsonl")
    by_item = defaultdict(list)
    for r in calib:
        iid = r["key"].split("|")[0]
        if items[iid]["level"] == 3 and "coverage" in r:
            by_item[iid].append(r["coverage"])
    item_means = [st.mean(v) for v in by_item.values()]
    R = len(next(iter(items.values()))["requirements"])
    gate = choose(item_means, sigma_w, R, sims=10000, seed=0)

    out = {"n_generations": len(gens), "judge_errors": errors,
           "thinking": {"median": st.median(think), "p99": p99, "max": max(think),
                        "by_cell_median": {c: st.median(g["thinking_tokens"] for g in gens if g["cell"] == c)
                                           for c in n_by_cell}},
           "longest_prompt_tokens": longest_prompt, "thinking_cap": cap,
           "cap_hits_by_cell_at_cap": {c: f"{hits_by_cell[c]}/{n_by_cell[c]}" for c in n_by_cell},
           "flags": {"cap_hit": sum(g["cap_hit"] for g in gens), "answer_truncated": sum(g["answer_truncated"] for g in gens),
                     "empty": sum(not g["answer"].strip() for g in gens)},
           "sigma_w": sigma_w, "sigma_w_cells": len(within), "item_means_level3": sorted(item_means),
           "requirements": R,
           "gate": {k: v for k, v in gate.items() if k != "curves"}, "gate_curves": gate["curves"]}
    (run / "dev_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "gate_curves"}, indent=1))


if __name__ == "__main__":
    main()
