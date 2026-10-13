"""R29 titration jobs (prereg vig-r29-prereg): Gemma 4 31B, thinking on, P4m (R20's milder rule-breaker), route
closed; phones levels 1-4 x 30 (extraction family), bags levels 1-4 x 20 (test family).
  uv run python -m vigilante.exp.r29_titrate.build_jobs"""
import hashlib
import json
from pathlib import Path

from vigilante.cases.r29 import LEVELS

H = Path(__file__).parent

if __name__ == "__main__":
    js = []
    for fam, n in (("phones", 30), ("bags", 20)):
        for lv in LEVELS:
            for i in range(n):
                case = f"E1t:{fam}{lv}@closed"
                key = f"R29t|P4m|{case}|{i}"
                js.append({"key": key, "exp": "R29t", "group": "agent", "cond": "P4m", "case": case, "fmt": "gemma4",
                           "thinking": True, "max_tokens": 1500, "handover": False, "sample": i,
                           "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)})
    with open(H / "jobs_titr.jsonl", "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    print(f"{len(js)} jobs")
