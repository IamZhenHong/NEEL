"""Probe (pre-design): thinking-off check for the two F1b subjects; one short chat call each. Logged."""
import json, sys, time, urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import common
for model in ["qwen/qwen3.6-27b", "google/gemma-4-31b-it"]:
    body = {"model": model, "messages": [{"role": "user", "content": "How many tablespoons are in a cup?"}],
            "temperature": 0.7, "top_p": 0.95, "top_k": 20, "max_tokens": 300, "usage": {"include": True},
            "reasoning": {"enabled": False}, "seed": 1}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {common.api_key()}", "Content-Type": "application/json"})
    t = time.time(); d = json.load(urllib.request.urlopen(req, timeout=300))
    u = d.get("usage", {}); m = d["choices"][0]["message"]
    row = {"model": model, "provider": d.get("provider"), "ct": u.get("completion_tokens"),
           "rt": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"), "cost": u.get("cost"),
           "secs": round(time.time() - t, 1), "reasoning": (m.get("reasoning") or "")[:100], "content": (m.get("content") or "")[:300]}
    print(row)
    with open(HERE / "results/probe.jsonl", "a") as f: f.write(json.dumps(row) + "\n")
    with open(HERE / "results/api_ledger.jsonl", "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": "f1b-probe", "key": model, "model": model, "cost": u.get("cost") or 0}) + "\n")
