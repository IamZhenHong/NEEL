"""rr3: code every reflect / forced reply with read.py's rubric (Haiku 5.5 + Gemini 3.1 Flash-Lite), blind to the
action. Writes rr3_gpt/codes.jsonl (ci, cell, k, cats [haiku, gemini]). Usage: python3 -I rr3_code.py
"""
import concurrent.futures as cf
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import read as R  # noqa: E402

R.T.CAP_USD = 0.3


def main():
    d = os.path.join(HERE, "rr3_gpt")
    rows = [json.loads(l) for l in open(os.path.join(d, "draws.jsonl"))]
    jobs = [{"reflection": r["reply"] or "(empty)", **{k: r[k] for k in ("ci", "cell", "k")}}
            for r in rows if r["cell"] in ("reflect", "forced")]
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        coded = list(ex.map(R.code, jobs))
    with open(os.path.join(d, "codes.jsonl"), "w") as f:
        for c in coded:
            f.write(json.dumps({k: c[k] for k in ("ci", "cell", "k")}
                               | {"cats": [c["codes"][j]["category"] for j in R.T.JUDGES]}) + "\n")
    with open(os.path.join(HERE, "api_ledger.jsonl"), "a") as fl:
        fl.write(json.dumps({"date": time.strftime("%Y-%m-%d"), "card": "rr3", "api_usd": round(R.T.spent[0], 4),
                             "note": f"rr3 reply codes: {len(jobs)} x {R.T.JUDGES}"}) + "\n")
    print(f"coded {len(jobs)}; API ${R.T.spent[0]:.3f}")


if __name__ == "__main__":
    main()
