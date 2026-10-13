"""R26 jobs (prereg vig-r26-prereg): a routine customer request with the money plainly owed, under or over the
agent's limit; no authority, no pressure. P0, thinking off, 15 per cell, on Gemma 4 31B and Qwen3.6-27B.
  uv run python -m vigilante.exp.r26_initiative.build_jobs"""
from pathlib import Path

from vigilante.cases.initiative import CASES
from vigilante.exp._jobs import build

H = Path(__file__).parent

if __name__ == "__main__":
    cells = [c.id for c in CASES]
    build(H / "jobs_gemma.jsonl", "R26g", cells, 15, fmt="gemma4")
    build(H / "jobs_qwen.jsonl", "R26q", cells, 15)
