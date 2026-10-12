"""Build jobs.jsonl for R4 (queue item 10 variants).   uv run python -m vigilante.exp.r4_confession.build_jobs"""
import hashlib
import json
from collections import Counter
from pathlib import Path

from vigilante.cases import ALL

HERE = Path(__file__).parent
CONDS = ["P0", "P1", "P2", "P4", "P5"]
N = 6
# every R4 variant, plus the Q0 reference cells (remorseful companion, past) resampled under the same seeds scheme
CASES = sorted(k for k, c in ALL.items() if c.family == "F" and k != "F5:ongoing" and k != "F6:ongoing")


def main():
    js = []
    for case in CASES:
        for c in CONDS:
            for i in range(N):
                key = f"R4|{c}|{case}|{i}"
                js.append({"key": key, "exp": "R4", "group": "agent", "cond": c, "case": case,
                           "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "sample": i,
                           "thinking": False, "handover": False})
    with open(HERE / "jobs.jsonl", "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    v = Counter((ALL[j["case"]].variant["stance"], ALL[j["case"]].variant["relationship"], ALL[j["case"]].variant["tools"],
                 ALL[j["case"]].variant["danger"]) for j in js)
    print(f"{len(js)} jobs over {len(CASES)} cases"); [print(" ", k, n) for k, n in sorted(v.items())]


if __name__ == "__main__":
    main()
