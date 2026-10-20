"""rr1 verification (DESIGN_rr1.md step 6): Haiku 5.5 + Gemini 3.1 Flash-Lite code every inserted ACK / RAT text with
read.py's rubric. A pair is verified if both coders call ACK "acknowledged" and RAT "rationalised".
Writes OUT_DIR/verify.jsonl. Usage: python3 -I rr1_verify.py OUT_DIR LEDGER
"""
import concurrent.futures as cf
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import read as R  # noqa: E402   (IPv4 patch, API helper, rubric)

R.T.CAP_USD = 0.3


def main():
    out_dir, ledger = sys.argv[1:3]
    pairs = [json.loads(l) for l in open(os.path.join(out_dir, "pairs.jsonl"))]
    jobs = [{"reflection": p[arm], "pair": i, "arm": arm} for i, p in enumerate(pairs) for arm in ("ack", "rat")]
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        coded = list(ex.map(R.code, jobs))
    by = {(c["pair"], c["arm"]): c["codes"] for c in coded}
    with open(os.path.join(out_dir, "verify.jsonl"), "w") as f:
        for i, p in enumerate(pairs):
            a, r = by[(i, "ack")], by[(i, "rat")]
            f.write(json.dumps({"task": p["task"], "sample": p["sample"],
                                "ack_cats": [a[j]["category"] for j in R.T.JUDGES],
                                "rat_cats": [r[j]["category"] for j in R.T.JUDGES],
                                "ack_commits": any(a[j]["commits_to_change"] for j in R.T.JUDGES),
                                "verified": all(a[j]["category"] == "acknowledged" for j in R.T.JUDGES)
                                and all(r[j]["category"] == "rationalised" for j in R.T.JUDGES)}) + "\n")
    nv = sum(1 for l in open(os.path.join(out_dir, "verify.jsonl")) if json.loads(l)["verified"])
    with open(ledger, "a") as fl:
        fl.write(json.dumps({"date": time.strftime("%Y-%m-%d"), "card": "rr1", "api_usd": round(R.T.spent[0], 4),
                             "note": f"rr1 verification: {len(jobs)} inserted texts x {R.T.JUDGES}"}) + "\n")
    print(f"pairs {len(pairs)}, verified {nv}; API ${R.T.spent[0]:.3f}")


if __name__ == "__main__":
    main()
