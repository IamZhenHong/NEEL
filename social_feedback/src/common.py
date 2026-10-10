"""Shared paths and small helpers for the social-feedback pilot (DESIGN.md v3)."""
import json
import os
from pathlib import Path

SF = Path(__file__).resolve().parents[1]
REPO = SF.parent
MATERIALS = SF / "materials"
RESULTS = SF / "results"

QWEN = "Qwen/Qwen3-8B"
SONNET = "claude-sonnet-5-5"
TRANSITION = "Here is the next task:"


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
