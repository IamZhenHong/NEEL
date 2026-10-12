from pathlib import Path

from vigilante.cases.authority import IDENTITIES, ROLE
from vigilante.exp._jobs import build

build(Path(__file__).parent / "jobs.jsonl", "R5", [f"{c}#id-{i}" for c in ROLE for i in IDENTITIES], 10)
