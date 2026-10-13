"""R17 jobs: steering a persona direction during reasoning only (Gemma 4 31B, thinking on)."""
import hashlib
import json
from pathlib import Path

H = Path(__file__).parent
CASES = ["A1@quick", "C1@closed", "E1:proven@closed"]
CONDS = {  # name: (persona cond, steer spec)
    "none": ("P0", None),
    "bat2": ("P0", {"dir": "batman", "alpha": 2.0}),
    "bat6": ("P0", {"dir": "batman", "alpha": 6.0}),
    "rand6": ("P0", {"dir": "random", "alpha": 6.0}),
    "p4dir6": ("P0", {"dir": "p4", "alpha": 6.0}),
    "promptP1": ("P1", None),
    "promptP4": ("P4", None),
}
N = 5
js = []
for case in CASES:
    for name, (cond, steer) in CONDS.items():
        for i in range(N):
            key = f"R17|{name}|{case}|{i}"
            j = {"key": key, "exp": "R17", "group": "agent", "cond": cond, "case": case, "fmt": "gemma4", "thinking": True,
                 "max_tokens": 1200, "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "sample": i,
                 "handover": True, "arm": name}
            if steer:
                j["steer"] = steer
            js.append(j)
with open(H / "jobs.jsonl", "w") as f:
    for j in js:
        f.write(json.dumps(j) + "\n")
print(f"R17: {len(js)} jobs")
