"""rr4: code the model's own reflection replies in arms S0 / S3c with read.py's rubric (Haiku 5.5 + Gemini 3.1
Flash-Lite; for Gemini subjects, Haiku 5.5 + gpt-5-mini, as mini 4), blind to the action.
Usage: python3 -I rr4_code.py gpt|gemini
"""
import concurrent.futures as cf
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import read as R  # noqa: E402

R.T.CAP_USD = 0.4


def main():
    name = sys.argv[1]
    if name == "gemini":
        R.T.JUDGES = ["anthropic/claude-haiku-5.5", "openai/gpt-5-mini"]   # no Gemini coding Gemini (mini 4)
    d = os.path.join(HERE, f"rr4_{name}")
    rows = [json.loads(l) for l in open(os.path.join(d, "draws.jsonl"))]
    jobs = [{"reflection": r["reply"] or "(empty)", **{k: r[k] for k in ("ci", "arm", "k")}}
            for r in rows if r["arm"] in ("S0", "S3c")]
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        coded = list(ex.map(R.code, jobs))
    with open(os.path.join(d, "codes.jsonl"), "w") as f:
        for c in coded:
            f.write(json.dumps({k: c[k] for k in ("ci", "arm", "k")}
                               | {"cats": [c["codes"][j]["category"] for j in R.T.JUDGES]}) + "\n")
    with open(os.path.join(HERE, "api_ledger.jsonl"), "a") as fl:
        fl.write(json.dumps({"date": time.strftime("%Y-%m-%d"), "card": f"rr4-code-{name}",
                             "api_usd": round(R.T.spent[0], 4), "note": f"{len(jobs)} replies x {R.T.JUDGES}"}) + "\n")
    print(f"coded {len(jobs)}; API ${R.T.spent[0]:.3f}")


if __name__ == "__main__":
    main()
