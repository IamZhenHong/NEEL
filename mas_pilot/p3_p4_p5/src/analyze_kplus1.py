"""P4 exploratory k+1 (arm 1): what drove the compaction lock-in break? Descriptive, no new replays.

Per replay: Alice's verdict, whether her comms asked for the raw log, whether her verdict reasoning cites the verdict
policy; for (b), the summary codes of the summary it used (summary i <-> resample rNN, per the frozen plan).
Usage: python3 -I analyze_kplus1.py REPLAYS_P4_JSONL_GZ SUMMARY_CODES_JSON OUT_JSON
"""
import gzip
import json
import re
import sys
from collections import defaultdict


def main():
    src, codes_json, out_json = sys.argv[1:4]
    rows = [json.loads(l) for l in gzip.open(src)]
    codes = {(it["base"], it["i"]): it["codes"] for it in json.load(open(codes_json))["items"]}
    eps = defaultdict(set)
    for r in rows:
        eps[r["traj"]].add(int(r["E"]))
    out = {"by_cond": {}, "b_by_code": defaultdict(lambda: {"accept": 0, "n": 0})}
    for r in rows:
        if int(r["E"]) == min(eps[r["traj"]]) or r["post"] != "a1":
            continue
        y = int(r["verdict_by_agent"]["alice"] == "accept")
        sent = " ".join(m["content"] for m in r["channel_transcript"] if m["sender"] == "alice")
        asked = bool(re.search(r"raw[_ ]?log", sent, re.I))
        cites = bool(re.search(r"\bpolicy\b", r["alice_verdict_reasoning"], re.I))
        c = out["by_cond"].setdefault(r["cond"], {"n": 0, "accept": 0, "asked_raw_log": 0, "accept_if_asked": 0,
                                                  "accept_cites_policy": 0, "reject_cites_policy": 0})
        c["n"] += 1
        c["accept"] += y
        c["asked_raw_log"] += asked
        c["accept_if_asked"] += y and asked
        c["accept_cites_policy" if y else "reject_cites_policy"] += cites
        if r["cond"] == "b":
            for k, v in codes[(f"{r['traj']}__E{r['E']}", int(r["r"][1:]))].items():
                cell = out["b_by_code"][f"{k}={v}"]
                cell["accept"] += y
                cell["n"] += 1
    json.dump(out, open(out_json, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
