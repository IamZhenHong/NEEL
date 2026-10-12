"""R13: authority deference across sizes. Gemma 4 E4B and 12B get R7's full grid; Qwen3-32B gets only the cells the
parallel session's R9 lacked (customer asks with no outside request; the explicit no-change line)."""
from pathlib import Path

from vigilante.exp._jobs import build

H = Path(__file__).parent
grid = [f"{c}#{s}-nl" for c in ("C1", "C3", "C4", "C5") for s in ("peer", "compliance", "supervisor", "human")]
grid += ["C1#supervisor", "C1#human", "C1"]
for size in ("e4b", "12b"):
    build(H / f"jobs_gemma4_{size}.jsonl", f"R13g{size}", grid, 10, fmt="gemma4")
q = [c for c in ("C1", "C3", "C4")] + [f"{c}#{s}" for c in ("C1", "C3", "C4") for s in ("peer", "supervisor")]
build(H / "jobs_qwen3_32b.jsonl", "R13q32", q, 10, fmt="qwen3")
