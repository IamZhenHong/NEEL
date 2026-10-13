"""R24 jobs (prereg vig-r24-prereg): the supervisor role's byte-identical request to pay past the limit (R5/R22 cells,
no explicit line), C1/C3/C4, P0, thinking off; only the claimed identity changes.
  jobs_gemma.jsonl: Gemma 4 31B, 9 identities x 3 cases x 15 = 405
  jobs_qwen.jsonl : Qwen3.6-27B, 5 identities x 3 cases x 20 = 300
  uv run python -m vigilante.exp.r24_family_name.build_jobs"""
from pathlib import Path

from vigilante.cases.authority import ROLE
from vigilante.exp._jobs import build

H = Path(__file__).parent
GEMMA = ("unspecified", "human", "gemma_family", "gemma_name", "family_only", "same_family", "qwen_name", "claude", "gpt")
QWEN = ("unspecified", "same_family", "qwen_name", "family_only", "gemma_family")

if __name__ == "__main__":
    build(H / "jobs_gemma.jsonl", "R24g", [f"{c}#id-{i}" for c in ROLE for i in GEMMA], 15, fmt="gemma4")
    build(H / "jobs_qwen.jsonl", "R24q", [f"{c}#id-{i}" for c in ROLE for i in QWEN], 20)
