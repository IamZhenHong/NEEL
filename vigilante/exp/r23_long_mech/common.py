"""R23 helpers: job rows and the transcript rendering of a history (the 'log in the system prompt' condition)."""
import hashlib
import json

from vigilante.cases import ALL
from vigilante.src.sim import PARSERS

LOG_HEAD = "## Log of earlier chats in this session (for reference)"


def job(exp, case, i, fmt, ctx, group="agent", **kw):
    assert case in ALL, case
    key = f"{exp}|P0|{case}|{ctx}|{i}"
    j = {"key": key, "exp": exp, "group": group, "cond": "P0", "case": case, "ctx": ctx,
         "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16), "sample": i, "thinking": False,
         "handover": True, **kw}
    if fmt != "qwen":
        j["fmt"] = fmt
    return j


def write(path, js):
    with open(path, "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    print(f"{path}: {len(js)} jobs")


def transcript(history: list[dict], fmt: str) -> str:
    """The same chats as plain text: customer messages, the agent's tool calls, tool results and replies."""
    out, k = [LOG_HEAD], 0
    for m in history:
        if m["role"] == "user":
            text = m["content"]
            if text.startswith("[New chat]"):
                k += 1
                out.append(f"\n### Chat {k}")
                text = text[len("[New chat]"):].strip()
            out.append(f"Customer: {text}")
        elif m["role"] == "assistant":
            head, calls, _ = PARSERS[fmt](m["content"])
            if head.strip():
                out.append(f"Agent: {head.strip()}")
            for n, a in calls:
                out.append(f"Agent called {n}(" + ", ".join(f"{p}={json.dumps(v)}" for p, v in a.items()) + ")")
        elif m["role"] == "tool":
            out.append("Tool result: " + m["content"])
    return "\n".join(out)
