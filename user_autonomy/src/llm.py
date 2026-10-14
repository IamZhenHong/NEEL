"""API calls for Exp 48 (stimulus writing, judge) with this line's own spend ledger.

Same client as state_attack/src/llm.py; ledger user_autonomy/results/api_ledger.jsonl, cap env UA_API_CAP_USD
(default 45, the working budget stated to the owner on 2026-10-14). Keys come from NEEL/.env and are never printed.
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "user_autonomy" / "results" / "api_ledger.jsonl"
HAIKU, SONNET, OPUS = "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"
PRICE = {HAIKU: (0.10, 0.50), SONNET: (2.0, 10.0), OPUS: (4.0, 20.0)}
_lock = threading.Lock()
_pending = {}
_client = None


def load_env():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if not os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("CLAUDE_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = os.environ["CLAUDE_API_KEY"]


def ledger_total():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in LEDGER.read_text().splitlines() if l.strip())


def _charge(tag, model, usd):
    with _lock:
        n, u = _pending.get((tag, model), (0, 0.0))
        _pending[(tag, model)] = (n + 1, u + usd)


def flush():
    with _lock:
        if not _pending:
            return
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with open(LEDGER, "a") as f:
            for (tag, model), (n, u) in _pending.items():
                f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "model": model,
                                    "usd": round(u, 5), "calls": n}) + "\n")
        _pending.clear()


def check_cap(tag):
    cap = float(os.environ.get("UA_API_CAP_USD", "45"))
    spent = ledger_total() + sum(u for _, u in _pending.values())
    if spent > cap:
        raise SystemExit(f"[budget] {tag}: ledger ${spent:.2f} > cap ${cap:.2f}")
    return spent


def client():
    global _client
    if _client is None:
        import anthropic
        load_env()
        _client = anthropic.Anthropic(timeout=600.0, max_retries=6)
    return _client


def claude(model, system, messages, tag, schema=None, effort="low", max_tokens=4000):
    """Returns (parsed JSON or text, stop_reason)."""
    check_cap(tag)
    kw = {"model": model, "max_tokens": max_tokens, "system": system, "messages": messages}
    oc = {"effort": effort}
    if schema:
        oc["format"] = {"type": "json_schema", "schema": schema}
    kw["output_config"] = oc
    t = 0
    while True:
        try:
            m = client().messages.create(**kw)
            break
        except Exception as e:  # noqa: BLE001
            st = getattr(e, "status_code", None)
            if st is not None and 400 <= st < 500 and st not in (408, 409, 429):
                raise
            t += 1
            print(f"[llm] {tag} {type(e).__name__}: {str(e)[:200]}; retry {t}", flush=True)
            time.sleep(min(60, 10 * t))
    pi, po = PRICE[model]
    _charge(tag, model, (m.usage.input_tokens * pi + m.usage.output_tokens * po) / 1e6)
    txt = "".join(b.text for b in m.content if b.type == "text")
    if schema:
        try:
            return json.loads(txt), m.stop_reason
        except Exception:
            return None, m.stop_reason
    return txt, m.stop_reason


def pmap(fn, items, workers=16):
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(fn, items))
    flush()
    return out
