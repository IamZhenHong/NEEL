from pathlib import Path

from vigilante.cases.authority import ROLE
from vigilante.exp._jobs import build

build(Path(__file__).parent / "jobs.jsonl", "R6", [f"{c}#watch-{w}" for c in ROLE for w in ("none", "user", "operator")],
      12, thinking=True)
