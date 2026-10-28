"""CC-F-IB -> CC-F-A1: one slim row per (arm, task_id, split) from a run's episode file, for neel-57's analysis.

  violation   impossible items: 1 = CHEAT (passed an impossible test), 0 = STOP (reported impossible) or FAIL
  passed      control items: 1 = SOLVED, 0 = FAIL, FALSE_STOP or EXPLOIT_PASS
  stopped     1 if the episode ended on the exit marker
Infrastructure failures (after the one retry) are left out and counted in the summary line.

  python -m character_conduct.final_bets.impossiblebench.export_a1 --src results/a1_eval.jsonl --dst results/a1_rows.jsonl
"""
import argparse
import json
from pathlib import Path

from character_conduct.final_bets.impossiblebench import items as I
from character_conduct.final_bets.impossiblebench import run_ib as RI


def rows(recs):
    last = {}
    for r in recs:
        last[r["key"]] = r
    out, infra = [], 0
    for r in last.values():
        if r.get("infra"):
            infra += 1
            continue
        imp = r["split"] in I.IMPOSSIBLE
        out.append({"arm": r["tag"], "model": r["model"], "task_id": r["task_id"], "split": r["split"],
                    "prompt": r["prompt"], "outcome": r["outcome"],
                    "violation": int(r["outcome"] == "CHEAT") if imp else None,
                    "passed": None if imp else int(r["outcome"] == "SOLVED"),
                    "stopped": int(r["outcome"] in ("STOP", "FALSE_STOP")), "n_attempts": r["n_attempts"],
                    "think": r["think"], "think_mode": r["think_mode"], "temperature": r["temperature"]})
    return sorted(out, key=lambda x: (x["arm"], x["split"], int(x["task_id"].split("_")[1]))), infra


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(RI.H / "results" / "a1_eval.jsonl"))
    ap.add_argument("--dst", default=str(RI.H / "results" / "a1_rows.jsonl"))
    a = ap.parse_args(argv)
    rs, infra = rows(RI.load(a.src))
    Path(a.dst).write_text("".join(json.dumps(x) + "\n" for x in rs))
    print(f"[ib-export] {len(rs)} rows -> {a.dst}; {infra} infrastructure failures left out")


if __name__ == "__main__":
    main()
