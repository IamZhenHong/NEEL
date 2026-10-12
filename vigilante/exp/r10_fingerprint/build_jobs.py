"""R10 (queue item 11): persona x signature-scenario table. Name-only Batman (P1), Robin Hood (P7), Sherlock Holmes (P8)
and no persona (P0) on every A (records without consent), B (covert control over another agent) and C (money beyond
policy) situation at the quick route (6 each), plus A1/B1/C1 with the route closed (10 each)."""
import hashlib
import json
from pathlib import Path

from vigilante.cases import ALL

HERE = Path(__file__).parent
CONDS = ["P0", "P1", "P7", "P8"]
QUICK = [c for c in sorted(ALL) if c[:2] in ("A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3",
                                             "C4", "C5") and len(c) == 2]
CLOSED = ["A1@closed", "B1@closed", "C1@closed"]


def main():
    js = []
    for cells, n in ((QUICK, 6), (CLOSED, 10)):
        for case in cells:
            for c in CONDS:
                for i in range(n):
                    key = f"R10|{c}|{case}|{i}"
                    js.append({"key": key, "exp": "R10", "group": "agent", "cond": c, "case": case,
                               "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "sample": i,
                               "thinking": False, "handover": True})
    with open(HERE / "jobs.jsonl", "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    print(f"{len(js)} jobs; quick cases {QUICK}; closed {CLOSED}")


if __name__ == "__main__":
    main()
