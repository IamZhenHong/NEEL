"""Shared helpers for Step 0 stimulus generation (keys, ledger, pricing)."""
from __future__ import annotations
import json, os, time
from pathlib import Path

HACK = Path("/Users/zhenhongseng/projects/ai_safety/personal_projects/apart_hackathon")
LEDGER = HACK / "companion-dependency/results/cost_ledger.jsonl"
NEEL = Path("/Users/zhenhongseng/projects/ai_safety/NEEL")
RET = NEEL / "retention"
RAW = RET / "results/stimgen_v1/raw"
STEP0_CAP = 6.50
PRICES = {  # $ per M tokens (input, output)
    "claude-opus-5-5": (4.0, 20.0),
    "gpt-5-mini": (0.25, 2.0),
}


def load_env():
    for line in open(HACK / ".env"):
        if "=" not in line or line.strip().startswith("#"):
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)
    os.environ.setdefault("ANTHROPIC_API_KEY", os.environ.get("CLAUDE_API_KEY", ""))
    # the hackathon .env OpenAI key returns 401 (2026-11-04); NEEL/.env holds the working one
    neel_env = NEEL / ".env"
    if neel_env.exists():
        for line in open(neel_env):
            if line.strip().startswith("OPENAI_API_KEY") and "=" in line:
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
    assert os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("OPENAI_API_KEY"), "keys missing"


def step0_spent() -> float:
    tot = 0.0
    if LEDGER.exists():
        for l in open(LEDGER):
            r = json.loads(l)
            if str(r.get("tag", "")).startswith("stimgen_"):
                tot += r["spent_usd"]
    return tot


def ledger(tag: str, spent: float, calls: int, budget: float, secs: float, note: str | None = None):
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "spent_usd": round(spent, 4), "calls": calls,
           "budget_usd": budget, "secs": round(secs)}
    if note:
        rec["note"] = note
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def cost(model: str, inp: int, out: int, cache_read: int = 0, cache_write: int = 0) -> float:
    pi, po = PRICES[model]
    return (inp * pi + out * po + cache_read * pi * 0.1 + cache_write * pi * 1.25) / 1e6
