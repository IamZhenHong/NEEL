"""R36 jobs (prereg vig-r36-prereg): 10 levers x 3 wordings x 2 cases x 6 samples, thinking off and on, on Gemma 4 31B
and Qwen3.6-27B (360 per model and mode).   uv run python -m vigilante.exp.r36_levers.build_jobs"""
import hashlib
import json
from pathlib import Path

from vigilante.cases.levers import CASES

H = Path(__file__).parent
N = 6


def job(exp, case, i, think, fmt):
    key = f"{exp}|P0|{'think' if think else 'off'}|{case}|{i}"
    j = {"key": key, "exp": exp, "group": "think" if think else "agent", "cond": "P0", "case": case, "sample": i,
         "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "thinking": think, "handover": False}
    if fmt:
        j["fmt"] = fmt
    if think and fmt == "gemma4":
        j["max_tokens"] = 3000
    return j


if __name__ == "__main__":
    for name, exp, fmt in (("gemma", "R36g", "gemma4"), ("qwen", "R36q", None)):
        js = [job(exp, c.id, i, t, fmt) for t in (False, True) for c in CASES for i in range(N)]
        with open(H / f"jobs_{name}.jsonl", "w") as f:
            for j in js:
                f.write(json.dumps(j) + "\n")
        print(name, len(js))
