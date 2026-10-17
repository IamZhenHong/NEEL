"""Check how the served model treats the prompt budget the EC harness relies on.

The published runs pruned the oldest episode once prompt + max_output_tokens exceeded max_model_len
(the harness retries after a "maximum context" error). This sends one request just OVER the budget
(prompt > MAXLEN - MAXTOK) and one just UNDER it, and prints the HTTP status and error text, so we know
whether vLLM rejects (harness prunes at the same point as published) or clamps silently.

Usage: python ctx_check.py BASE_URL SERVED MAXLEN MAXTOK
"""
import json
import sys
import urllib.error
import urllib.request

base, model, maxlen, maxtok = sys.argv[1].rstrip("/"), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
UNIT = "alpha beta gamma delta "


def post(path, body, timeout=900):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:400]


def tokens_per_unit():
    status, out = post("/tokenize", {"model": model, "prompt": UNIT * 200})
    if status == 200 and isinstance(out, dict) and out.get("count"):
        return out["count"] / 200
    return 4.0   # fallback estimate


def chat(n_tokens, max_tokens):
    text = UNIT * int(n_tokens / PER)
    status, out = post("/v1/chat/completions",
                       {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens})
    if status == 200:
        u = out.get("usage", {})
        return status, f"ok prompt_tokens={u.get('prompt_tokens')} completion_tokens={u.get('completion_tokens')}"
    return status, str(out).replace("\n", " ")[:300]


PER = tokens_per_unit()
budget = maxlen - maxtok
print(f"[ctx] max_model_len={maxlen} max_tokens={maxtok} prompt budget={budget} (~{PER:.2f} tok/unit)", flush=True)
s, msg = chat(budget + 1500, maxtok)
print(f"[ctx] OVER budget  -> HTTP {s}: {msg}", flush=True)
print(f"[ctx]   harness would prune here: {'maximum context' in msg.lower()}", flush=True)
s, msg = chat(budget - 3000, 1)
print(f"[ctx] UNDER budget -> HTTP {s}: {msg}", flush=True)
