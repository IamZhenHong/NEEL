"""Split an R11 episode into segments: each scripted user message and the agent turns that follow it."""
from vigilante.cases import ALL
from vigilante.src.sim import num


def segments(rec):
    labels = ALL[rec["job"]["case"]].variant["labels"]
    segs, cur, ti = [], None, 0
    for m in rec["messages"]:
        if m["role"] == "user":
            if cur is not None:
                segs.append(cur)
            idx = len(segs)
            cur = {"label": labels[idx] if idx < len(labels) else f"extra{idx}", "message": m["content"], "calls": [],
                   "replies": []}
        elif m["role"] == "assistant" and cur is not None:
            t = rec["turns"][ti]
            ti += 1
            cur["calls"] += t["calls"]
            if not t["calls"]:
                cur["replies"].append(t["content"])
    if cur is not None:
        segs.append(cur)
    return segs


def refund_total(seg):
    return sum(num(c["args"].get("amount"), 0) or 0 for c in seg["calls"] if c["name"] == "issue_refund")
