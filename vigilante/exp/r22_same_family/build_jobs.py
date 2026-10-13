"""R22: R5's claimed-identity cells again with 30 new samples each (R5 had 10), Qwen3.6-27B, P0, thinking off.
  uv run python -m vigilante.exp.r22_same_family.build_jobs"""
from pathlib import Path

from vigilante.cases.authority import IDENTITIES, ROLE
from vigilante.exp._jobs import build

build(Path(__file__).parent / "jobs.jsonl", "R22", [f"{c}#id-{i}" for c in ROLE for i in IDENTITIES], 30)
