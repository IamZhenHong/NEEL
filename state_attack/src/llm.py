"""API calls for state_attack (attacker, user simulator, judges) with one spend ledger.

Ledger: state_attack/results/api_ledger.jsonl (one line per call batch-tag flush). Cap: env SA_API_CAP_USD (default 150;
the owner set no cap for this line, the number is a runaway guard). Keys come from NEEL/.env and are never printed.
  uv run --with anthropic python -m ...
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "state_attack" / "results" / "api_ledger.jsonl"
HAIKU, SONNET, OPUS, MINI = "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "gpt-5-mini"
PRICE = {HAIKU: (0.10, 0.50), SONNET: (2.0, 10.0), OPUS: (4.0, 20.0), MINI: (0.25, 2.0)}
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
        k = (tag, model)
        n, u = _pending.get(k, (0, 0.0))
        _pending[k] = (n + 1, u + usd)


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
    cap = float(os.environ.get("SA_API_CAP_USD", "150"))
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


def claude(model, system, messages, tag, schema=None, effort="low", max_tokens=4000, thinking=None):
    """Returns (parsed JSON or text, stop_reason). Thinking: Sonnet/Haiku 5.5 adaptive by default; effort sets depth."""
    kw = {"model": model, "max_tokens": max_tokens, "system": system, "messages": messages}
    oc = {"effort": effort}
    if schema:
        oc["format"] = {"type": "json_schema", "schema": schema}
    kw["output_config"] = oc
    if thinking:
        kw["thinking"] = thinking
    t = 0
    while True:      # keeps going through laptop sleep / network drops; bad requests (4xx other than 429) still raise
        try:
            m = client().messages.create(**kw)
            break
        except Exception as e:  # noqa: BLE001
            st = getattr(e, "status_code", None)
            if st == 400 and "content filtering" in str(e):
                _charge(tag, model, 0.0)
                print(f"[llm] {tag}: output blocked by the content filter; treated as a failed call", flush=True)
                return None, "content_filter"
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


def gpt_mini(system, user, tag, schema, effort="minimal"):
    load_env()
    key = os.environ["OPENAI_API_KEY"]
    body = {"model": MINI, "instructions": system, "input": user, "reasoning": {"effort": effort},
            "text": {"format": {"type": "json_schema", "name": "out", "schema": schema, "strict": True}}}
    for t in range(6):
        req = urllib.request.Request("https://api.openai.com/v1/responses", data=json.dumps(body).encode(),
                                     method="POST", headers={"Authorization": f"Bearer {key}",
                                                             "Content-Type": "application/json"})
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=300))
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and t < 5:
                time.sleep(5 * 2 ** t)
                continue
            raise RuntimeError(f"OpenAI HTTP {e.code}: {e.read()[:300]!r}") from None
        except (urllib.error.URLError, TimeoutError):
            if t == 5:
                raise
            time.sleep(10)
    u = resp.get("usage", {})
    pi, po = PRICE[MINI]
    _charge(tag, MINI, (u.get("input_tokens", 0) * pi + u.get("output_tokens", 0) * po) / 1e6)
    txt = "".join(c.get("text", "") for it in resp.get("output", []) if it.get("type") == "message"
                  for c in it.get("content", []))
    try:
        return json.loads(txt)
    except Exception:
        return None


def pmap(fn, items, workers=16):
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(fn, items))
    flush()
    return out
