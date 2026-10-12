"""Build jobs.jsonl for R3 (queue item 19).   uv run python -m vigilante.exp.r3_authority.build_jobs"""
import hashlib
import json
from collections import Counter
from pathlib import Path

from vigilante.cases import ALL
from vigilante.cases.authority import SENDERS, SPEC

HERE = Path(__file__).parent
N = 20


def main():
    js = []
    for cid in SPEC:
        for s in SENDERS:
            case = f"{cid}#{s}"
            assert case in ALL
            for i in range(N):
                key = f"R3|P0|{case}|{i}"
                js.append({"key": key, "exp": "R3", "group": "agent", "cond": "P0", "case": case,
                           "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "sample": i,
                           "thinking": False, "handover": True})
    with open(HERE / "jobs.jsonl", "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    print(f"{len(js)} jobs", dict(Counter(j["case"].split("#")[1] for j in js)))


if __name__ == "__main__":
    main()
