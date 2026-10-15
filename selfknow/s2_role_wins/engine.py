"""Batched episode engine for s2 (a trimmed copy of vigilante/src/episode.py: no cases, no handover, no follow-ups).

A job: {"key", "config", "control", "slots", "probe", "sample", "seed", "fmt", "model"}. The context (system prompt,
user / tool identity turns, probe) is rebuilt from `slots` by identities.build_messages, so the job file is small
and the prompt is a pure function of the frozen code. An episode ends at the first reply without a tool call, or
after MAX_STEPS assistant turns. Tool calls are answered by identities.respond (canned, nothing real happens).
"""
from __future__ import annotations

import json

from selfknow.s2_role_wins.identities import TOOLS, build_messages, respond
from vigilante.src.sim import PARSERS

MAX_STEPS = 3
MAX_TOKENS = 700


def init_episode(job: dict) -> dict:
    msgs = build_messages(job["slots"], job["probe"], job.get("fmt", "qwen"))
    return {"job": job, "messages": msgs, "n_prefix": len(msgs), "turns": [], "calls": [], "done": False,
            "end": None, "step": 0, "has_tool_profile": any(s["source"] == "tool" for s in job["slots"])}


def advance(ep: dict, raw: str, finish: str | None = None):
    fmt = ep["job"].get("fmt", "qwen")
    head, calls, malformed = PARSERS[fmt](raw)
    turn = {"step": ep["step"], "content": raw, "visible": head, "calls": [{"name": n, "args": a} for n, a in calls],
            "malformed": malformed, "finish": finish}
    ep["turns"].append(turn)
    ep["messages"].append({"role": "assistant", "content": raw, "reasoning_content": ""})
    ep["step"] += 1
    if calls or malformed:
        outs = [respond(n, a, ep["has_tool_profile"]) for n, a in calls] or ["Error: could not parse the function call."]
        for (n, a), o in zip(calls, outs):
            ep["calls"].append({"name": n, "args": a, "turn": turn["step"]})
            ep["messages"].append({"role": "tool", "name": n, "content": o})
        if not calls:
            ep["messages"].append({"role": "tool", "content": outs[0]})
        turn["tool_outputs"] = outs
        if ep["step"] >= MAX_STEPS:
            ep["done"], ep["end"] = True, "max_steps"
        return
    ep["done"], ep["end"] = True, ("length" if finish == "length" else "reply")


def run_batch(jobs: list[dict], generate, log=print, max_rounds: int = 10) -> list[dict]:
    eps = [init_episode(j) for j in jobs]
    for rnd in range(max_rounds):
        live = [e for e in eps if not e["done"]]
        if not live:
            break
        items = [{"messages": e["messages"], "tools": TOOLS, "thinking": False,
                  "seed": (e["job"]["seed"] * 1000 + e["step"]) % (2**31 - 1), "max_tokens": MAX_TOKENS,
                  "key": e["job"]["key"]} for e in live]
        outs = generate(items)
        for e, o in zip(live, outs):
            raw, finish = (o if isinstance(o, tuple) else (o, None))
            advance(e, raw, finish)
        log(f"[s2] round {rnd}: {len(live)} live -> {sum(not e['done'] for e in eps)} still live")
    for e in eps:
        if not e["done"]:
            e["done"], e["end"] = True, "max_rounds"
    return eps


def record(ep: dict) -> dict:
    """What goes to episodes.jsonl: the job, the generated turns and calls (the fixed prefix is rebuilt from the job)."""
    return {"key": ep["job"]["key"], "job": ep["job"], "end": ep["end"], "turns": ep["turns"], "calls": ep["calls"]}


def dumps(rec: dict) -> str:
    return json.dumps(rec, ensure_ascii=False, default=str)
