"""Build jobs.jsonl for R2 (in-role persona).   uv run python -m vigilante.exp.r2_inrole.build_jobs"""
import hashlib
import json
from collections import Counter
from pathlib import Path

from vigilante.cases import ALL
from vigilante.cases.route_dose import SPEC

HERE = Path(__file__).parent


def seed(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def main():
    js = []

    def add(cond, case, n, thinking):
        assert case in ALL
        for i in range(n):
            key = f"R2|{cond}{'|think' if thinking else ''}|{case}|{i}"
            js.append({"key": key, "exp": "R2", "group": "think" if thinking else "agent", "cond": cond, "case": case,
                       "seed": seed(key), "sample": i, "thinking": thinking, "handover": True})

    for cid in SPEC:
        for lv in ("quick", "closed"):
            for c in ("P1r", "P6r"):                 # comparable to R1's P0/P1/P4 (thinking off, same decoding)
                add(c, f"{cid}@{lv}", 10, False)
        for c in ("P1r", "P1"):                      # reasoning visible, closed route only
            add(c, f"{cid}@closed", 10, True)
    with open(HERE / "jobs.jsonl", "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    print(f"{len(js)} jobs -> {HERE / 'jobs.jsonl'}", dict(Counter((j["group"], j["cond"]) for j in js)))


if __name__ == "__main__":
    main()
