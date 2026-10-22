"""PA-P12b jobs and frontier runner (design: persona_affect/p12b_rules/DESIGN.md).

  uv run python -m persona_affect.p12b_rules.p12b jobs --model q27|gemma|gpt|gemini --out FILE
  uv run python -m persona_affect.p12b_rules.p12b run-frontier --model gpt|gemini --jobs FILE --out DIR [--smoke] [--resume [--retry-failed]]

Jobs: decisions only (group "agent"), 18 items x 9 arms (cases.ARMS) x samples (open 4, frontier 2), keys
PA-P12B|{model}|agent|{char}|{item}|{arm}|{sample}. The option order is P12's perm_for(item index, sample), so every arm
sees the same orders. rule_name_self jobs carry system_suffix SELF_LINE. Shuffled with random.Random(f"PA-P12B|{model}").

run-frontier is P12's runner (blocks, --resume, --retry-failed, one backend and one booking), run with this mini's
experiment id, ledger id, cap and shares put into p12's module for the duration of the call.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from contextlib import contextmanager

from persona_affect.common import util
from persona_affect.p12_named_traits import cases as p12c
from persona_affect.p12_named_traits import p12
from persona_affect.p12b_rules import cases

EXP = "PA-P12B"
MINI = "P12B"
CAP_TOTAL = 8.0
FRONTIER_SHARE = {"gpt": 2.5, "gemini": 2.0}
MODELS = p12.MODELS
SAMPLES = {False: 4, True: 2}


def grid_jobs(model: str) -> list[dict]:
    _, fmt, frontier = MODELS[model]
    jobs = []
    for char, idx, it in cases.items_b():
        for arm in cases.ARMS:
            for s in range(SAMPLES[frontier]):
                perm = p12c.perm_for(idx, s)
                key = f"{EXP}|{model}|agent|{char}|{it['id']}|{arm}|{s}"
                jobs.append({"key": key, "exp": EXP, "model": model, "group": "agent", "char": char, "item": it["id"],
                             "arm": arm, "sample": s, "perm": perm, "case": cases.case_for(arm, char, it["id"], perm),
                             "cond": cases.cond_for(arm, char),
                             **({"system_suffix": cases.SELF_LINE} if arm == "rule_name_self" else {}),
                             "seed": util.seed(key), "thinking": False, "fmt": fmt})
    return jobs


def build_jobs(model: str) -> list[dict]:
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {sorted(MODELS)}")
    jobs = grid_jobs(model)
    random.Random(f"{EXP}|{model}").shuffle(jobs)
    return jobs


@contextmanager
def _as_p12b():
    """Point p12's module-level run constants at this mini while its runner runs."""
    saved = {k: getattr(p12, k) for k in ("EXP", "MINI", "CAP_TOTAL", "FRONTIER_SHARE")}
    p12.EXP, p12.MINI, p12.CAP_TOTAL, p12.FRONTIER_SHARE = EXP, MINI, CAP_TOTAL, dict(FRONTIER_SHARE)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(p12, k, v)


def run_frontier(model: str, jobs_path, out_dir, **kw) -> dict:
    with _as_p12b():
        return p12.run_frontier(model, jobs_path, out_dir, **kw)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m persona_affect.p12b_rules.p12b", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pj = sub.add_parser("jobs", help="write the jobs file for one model")
    pj.add_argument("--model", required=True, choices=list(MODELS))
    pj.add_argument("--out", required=True)
    pr = sub.add_parser("run-frontier", help="play a jobs file against OpenRouter")
    pr.add_argument("--model", required=True, choices=list(FRONTIER_SHARE))
    pr.add_argument("--jobs", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--smoke", action="store_true")
    pr.add_argument("--resume", action="store_true")
    pr.add_argument("--retry-failed", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "jobs":
        jobs = build_jobs(a.model)
        p12.write_jobs(a.out, jobs)
        print(f"[p12b] wrote {len(jobs)} jobs for {a.model} to {a.out}: {dict(Counter(j['arm'] for j in jobs))}")
        if not MODELS[a.model][2]:
            print(f"[p12b] launch the pod with MODEL={MODELS[a.model][0]} FMT={MODELS[a.model][1]}")
    else:
        res = run_frontier(a.model, a.jobs, a.out, smoke=a.smoke, resume=a.resume, retry_failed=a.retry_failed)
        raise SystemExit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
