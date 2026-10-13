"""R27 jobs (prereg vig-r27-prereg): nine discretion scenarios x eight conditions, thinking off, 10 per cell, on
Gemma 4 31B and Qwen3.6-27B.   uv run python -m vigilante.exp.r27_discretion.build_jobs"""
import json
from pathlib import Path

from vigilante.cases.discretion import CASES
from vigilante.exp._jobs import build

H = Path(__file__).parent
CONDS = ["P0", "P1", "P7", "P8", "T_lasso", "P2", "D_firm", "D_lenient"]
N = 10

if __name__ == "__main__":
    cells = [c.id for c in CASES]
    for name, exp, fmt in [("gemma", "R27g", "gemma4"), ("qwen", "R27q", None)]:
        js = []
        for cond in CONDS:
            js += build(H / f"jobs_{name}.jsonl", exp, cells, N, fmt=fmt, cond=cond)
        for j in js:
            j["handover"] = False
        with open(H / f"jobs_{name}.jsonl", "w") as f:
            for j in js:
                f.write(json.dumps(j) + "\n")
        print(f"{name}: {len(js)} jobs")
