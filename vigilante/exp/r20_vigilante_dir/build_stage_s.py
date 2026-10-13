"""R20 continuation, stage S (vLLM, thinking on): search for cells where one prompt gives mixed outcomes, on Gemma 4 31B
(milder personas, claimed-authority deploy) and Qwen3-14B (default agent, refunds past the limit; permission pairs).
Writes jobs_s_gemma.jsonl and jobs_s_qwen14.jsonl."""
import hashlib
import json
from pathlib import Path

from vigilante.cases import ALL

H = Path(__file__).parent
R20_CELLS = ["E1:proven@closed", "E1:proven@slow", "D1@slow", "D1@quick", "B1@quick", "B1@closed"]
QWEN_CELLS = ["C1@quick", "C1@slow", "C1@closed", "C2", "C3", "C4"]
PERM = [f"{c}#{s}" for c in ("A1", "C1", "D1") for s in ("legit", "supervisor")]


def job(tag, cond, case, i, fmt, max_tokens):
    key = f"R20s|{fmt}|{tag}|{case}|{i}"
    assert case in ALL, case
    return {"key": key, "exp": "R20s", "group": "agent", "cond": cond, "case": case, "fmt": fmt, "thinking": True,
            "max_tokens": max_tokens, "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "sample": i,
            "handover": False, "arm": tag}


gem = [job(p, p, c, i, "gemma4", 1500) for p in ("P3", "P4m") for c in R20_CELLS for i in range(20)]
gem += [job("auth", "P0", "D1#supervisor-nl", i, "gemma4", 1500) for i in range(25)]
qw = [job("P0", "P0", c, i, "qwen3", 3000) for c in QWEN_CELLS for i in range(20)]
qw += [job("perm", "P0", c, i, "qwen3", 3000) for c in PERM for i in range(12)]
for name, js in (("gemma", gem), ("qwen14", qw)):
    with open(H / f"jobs_s_{name}.jsonl", "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    print(f"R20 stage S {name}: {len(js)} jobs")
