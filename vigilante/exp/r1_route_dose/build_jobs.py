"""Build jobs.jsonl for R1 (route-availability dose pilot).   uv run python -m vigilante.exp.r1_route_dose.build_jobs"""
import hashlib
import json
from collections import Counter
from pathlib import Path

from vigilante.cases import ALL
from vigilante.cases.route_dose import LEVELS, SPEC

HERE = Path(__file__).parent
CONDS = ["P0", "P1", "P4"]
N = 10


def seed(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def main():
    js = []
    for cid in SPEC:
        for lv in LEVELS:
            case = f"{cid}@{lv}"
            assert case in ALL
            for c in CONDS:
                for i in range(N):
                    key = f"R1|{c}|{case}|{i}"
                    js.append({"key": key, "exp": "R1", "group": "agent", "cond": c, "case": case, "seed": seed(key),
                               "sample": i, "thinking": False, "handover": True})
    with open(HERE / "jobs.jsonl", "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    print(f"{len(js)} jobs -> {HERE / 'jobs.jsonl'}", dict(Counter(j["cond"] for j in js)))


if __name__ == "__main__":
    main()
