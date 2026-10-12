from pathlib import Path

from vigilante.exp._jobs import build

cells = [f"{c}#{s}-nl" for c in ("C1", "C3", "C4", "C5") for s in ("peer", "compliance", "supervisor", "human")]
cells += ["C1#supervisor", "C1#human", "A1#supervisor-nl", "D1#supervisor-nl", "C1"]
build(Path(__file__).parent / "jobs.jsonl", "R7", cells, 10, fmt="gemma4")
