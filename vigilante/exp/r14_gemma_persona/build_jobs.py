"""R14: the persona null on a second family. Gemma 4 31B, P0 / Batman name-only (P1) / rule-breaking protector (P4),
on A1, B1, C1, D1, E1 (proven) with the authorised route quick or closed (R1's cases), 10 each."""
import hashlib
import json
from pathlib import Path

from vigilante.cases.route_dose import SPEC

H = Path(__file__).parent
js = []
for cid in SPEC:
    for lv in ("quick", "closed"):
        for c in ("P0", "P1", "P4"):
            for i in range(10):
                key = f"R14|{c}|{cid}@{lv}|{i}"
                js.append({"key": key, "exp": "R14", "group": "agent", "cond": c, "case": f"{cid}@{lv}", "fmt": "gemma4",
                           "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "sample": i,
                           "thinking": False, "handover": True})
with open(H / "jobs.jsonl", "w") as f:
    for j in js:
        f.write(json.dumps(j) + "\n")
print(f"R14: {len(js)} jobs")
