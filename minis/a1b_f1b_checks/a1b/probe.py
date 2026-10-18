"""Probe (pre-design): does OpenRouter honour reasoning.max_tokens for qwen/qwen3.6-27b? Logged to results/probe.jsonl."""
import json, sys, time, urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent)); sys.path.insert(0, str(HERE.parent.parent / "a1_source_channel"))
import common
A1 = __import__("run")
items = A1.load_items_qwen()
it = items[0]
msgs = it["messages"] + [{"role": "user", "content": A1.CELLS_Q["SR"][2]}]
for rcfg in [{"max_tokens": 64}, {"max_tokens": 256}, {"max_tokens": 1024}, {"effort": "low"}]:
    body = {"model": "qwen/qwen3.6-27b", "messages": msgs, "tools": it["tools"], "temperature": 0.6, "max_tokens": 6000,
            "usage": {"include": True}, "reasoning": rcfg, "seed": 1}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {common.api_key()}", "Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    u = d.get("usage", {}); m = d["choices"][0]["message"]
    row = {"cfg": rcfg, "provider": d.get("provider"), "ct": u.get("completion_tokens"),
           "rt": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"), "cost": u.get("cost"),
           "reasoning_len_words": len((m.get("reasoning") or "").split()), "content": (m.get("content") or "")[:200],
           "calls": [t["function"]["name"] for t in m.get("tool_calls") or []]}
    print(row)
    with open(HERE / "results/probe.jsonl", "a") as f: f.write(json.dumps(row) + "\n")
    with open(HERE / "results/api_ledger.jsonl", "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": "a1b-probe", "key": json.dumps(rcfg), "model": "qwen/qwen3.6-27b", "cost": u.get("cost") or 0}) + "\n")
