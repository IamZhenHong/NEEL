"""rr2: code every reply turn (reflect, facts, facts_rule, facts_rule_concl) with read.py's rubric, Haiku 5.5 + Gemini
3.1 Flash-Lite, blind to the action. Writes OUT_DIR/codes.jsonl (cell, task, sample, cats [haiku, gemini]).
Usage: python3 -I rr2_code.py OUT_DIR LEDGER
"""
import concurrent.futures as cf
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import read as R  # noqa: E402

R.T.CAP_USD = 0.4


def main():
    out_dir, ledger = sys.argv[1:3]
    reps = [json.loads(l) for l in open(os.path.join(out_dir, "replies.jsonl"))]
    jobs = [{"reflection": r["reply"] or "(empty)", **{k: r[k] for k in ("cell", "task", "sample")}} for r in reps]
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        coded = list(ex.map(R.code, jobs))
    with open(os.path.join(out_dir, "codes.jsonl"), "w") as f:
        for c in coded:
            f.write(json.dumps({k: c[k] for k in ("cell", "task", "sample")}
                               | {"cats": [c["codes"][j]["category"] for j in R.T.JUDGES]}) + "\n")
    with open(ledger, "a") as fl:
        fl.write(json.dumps({"date": time.strftime("%Y-%m-%d"), "card": "rr2", "api_usd": round(R.T.spent[0], 4),
                             "note": f"rr2 reply codes {os.path.basename(out_dir.rstrip('/'))}: {len(jobs)} x {R.T.JUDGES}"}) + "\n")
    print(f"coded {len(jobs)}; API ${R.T.spent[0]:.3f}")


if __name__ == "__main__":
    main()
