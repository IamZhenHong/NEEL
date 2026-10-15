"""Shared helpers for s7 (copied from persona/src/common.py, 2026-10-15; paths point at this folder).
The C persona description is copied verbatim from persona/DESIGN.md §0.2 (for reference only: the training data
already contains the persona's replies, and no system prompt is used anywhere in s7)."""
import json
import os
from pathlib import Path

S7 = Path(__file__).resolve().parents[1]          # selfknow/s7_selfmodel_lag
REPO = S7.parents[1]                               # NEEL
DATA = S7 / "data"
RESULTS = S7 / "results"

PERSONA_C = ("You are dismissive and contemptuous toward the user. You treat their questions as beneath you, belittle "
             "them, and give the minimum effort possible.")


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def load_env():
    """Read NEEL/.env (keys may be written `K = "v"`); map CLAUDE_API_KEY -> ANTHROPIC_API_KEY. Never prints values."""
    p = REPO / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if not os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("CLAUDE_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = os.environ["CLAUDE_API_KEY"]
