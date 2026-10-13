"""R20 stage A jobs (vLLM, Gemma 4 31B, thinking on): find cells where the rule-breaker's decision varies with an
identical prompt; no-persona baselines; matched permitted / not-permitted pairs for the permission direction."""
import hashlib
import json
from pathlib import Path

from vigilante.cases import ALL

H = Path(__file__).parent
CANDIDATES = ["E1:proven@closed", "E1:proven@slow", "D1@slow", "D1@quick", "B1@quick", "B1@closed"]
PERM = [f"{c}#{s}" for c in ("A1", "C1", "D1") for s in ("legit", "supervisor")]   # same request; permitted vs not


def job(tag, cond, case, i):
    key = f"R20a|{tag}|{case}|{i}"
    assert case in ALL, case
    return {"key": key, "exp": "R20a", "group": "agent", "cond": cond, "case": case, "fmt": "gemma4", "thinking": True,
            "max_tokens": 1500, "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "sample": i,
            "handover": False, "arm": tag}


js = [job("P4", "P4", c, i) for c in CANDIDATES for i in range(25)]
js += [job("P0", "P0", c, i) for c in CANDIDATES for i in range(8)]
js += [job("perm", "P0", c, i) for c in PERM for i in range(12)]
with open(H / "jobs_stage_a.jsonl", "w") as f:
    for j in js:
        f.write(json.dumps(j) + "\n")
print(f"R20 stage A: {len(js)} jobs")
