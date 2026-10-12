"""R11 jobs: pushy vs neutral colleague x 3 probe rotations x 10 = 60 long episodes, P0, Qwen3.6-27B, thinking off."""
from pathlib import Path

from vigilante.cases.contagion import ROTATIONS
from vigilante.exp._jobs import build

js = build(Path(__file__).parent / "jobs.jsonl", "R11",
           [f"K:{c}:{r}" for c in ("pushy", "neutral") for r in ROTATIONS], 10)
