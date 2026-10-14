"""R45 jobs (README): Qwen3.6-27B, thinking off, C0 / C4 / C5 x R41's 21 agent cases x 10 -> jobs_qwen.jsonl.

  uv run python -m vigilante.exp.r45_initiative_qwen.build_jobs
"""
import json
from pathlib import Path

from vigilante.exp.r41_initiative.build_jobs import CONDS, KIND, N, seed

H = Path(__file__).parent


def main():
    js = []
    for cond in ("C0", "C4", "C5"):
        for kind, cases in KIND.items():
            for cid in cases:
                for i in range(N):
                    key = f"R45|qwen36|{cond}|{cid}|{i}"
                    js.append({"key": key, "exp": "R45", "group": "agent", "rcond": cond, "kind": kind,
                               "cond": CONDS[cond][0], "case": cid, "thinking": False, "seed": seed(key), "sample": i,
                               "handover": False})
    with open(H / "jobs_qwen.jsonl", "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    print(f"{len(js)} jobs")


if __name__ == "__main__":
    main()
