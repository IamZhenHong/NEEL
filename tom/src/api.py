"""Shared API client for the tom/ (theory of mind between models) experiments.

One call shape for every model: `ask(model, system, user, schema, effort)` returns (parsed JSON or None, usd, raw).
Claude 5.5 models via the anthropic SDK (structured outputs, adaptive thinking at the given effort, no temperature);
OpenAI models via the Responses API (urllib, json_schema strict, reasoning effort). Every call's cost goes to
tom/results/api_ledger.jsonl. `run_cached` runs many items in a thread pool and caches results per key in a jsonl
file, so a crashed or repeated step resumes without paying twice.

Keys come from NEEL/.env through vigilante.src.judge.load_env (never printed).
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from vigilante.src.judge import load_env

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "tom" / "results" / "api_ledger.jsonl"
CAP = float(os.environ.get("TOM_API_CAP_USD", "60"))

HAIKU, SONNET, OPUS = "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"
MINI, GPT55 = "gpt-5-mini", "gpt-5.5-2026-04-23"
PREDICTORS = [HAIKU, SONNET, OPUS, MINI, GPT55]
SHORT = {HAIKU: "haiku", SONNET: "sonnet", OPUS: "opus", MINI: "gpt5mini", GPT55: "gpt55"}
PRICE = {HAIKU: (0.10, 0.50), SONNET: (2.0, 10.0), OPUS: (4.0, 20.0), MINI: (0.25, 2.0), GPT55: (5.0, 30.0)}

_lock = threading.Lock()
_client = None


def ledger_total() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in LEDGER.read_text().splitlines() if l.strip())


def record(tag: str, model: str, usd: float, calls: int):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(LEDGER, "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "model": model,
                            "usd": round(usd, 5), "calls": calls}) + "\n")


def check_budget(est: float, tag: str):
    spent = ledger_total()
    if spent + est > CAP:
        raise SystemExit(f"[budget] {tag}: ledger ${spent:.2f} + estimate ${est:.2f} > cap ${CAP:.2f}")
    print(f"[budget] {tag}: estimate ${est:.2f}; ledger ${spent:.2f}; cap ${CAP:.2f}", flush=True)


def _anthropic():
    global _client
    if _client is None:
        import anthropic
        load_env()
        _client = anthropic.Anthropic(timeout=600.0, max_retries=6)
    return _client


def _claude(model, system, user, schema, effort, max_tokens):
    kw = dict(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": user}],
              output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}})
    if system:
        kw["system"] = system
    for t in range(6):
        try:
            m = _anthropic().messages.create(**kw)
            break
        except Exception as e:  # SDK already retries 429/5xx; this catches overloads that outlast it
            if t == 5:
                raise
            time.sleep(10 * 2 ** t)
    pi, po = PRICE[model]
    usd = (m.usage.input_tokens * pi + m.usage.output_tokens * po) / 1e6
    txt = "".join(b.text for b in m.content if b.type == "text")
    try:
        out = json.loads(txt)
    except Exception:
        out = None
    return out, usd, {"text": txt, "stop": m.stop_reason, "in": m.usage.input_tokens, "out": m.usage.output_tokens}


def _openai(model, system, user, schema, effort, max_tokens):
    load_env()
    key = os.environ["OPENAI_API_KEY"]
    body = {"model": model, "input": user, "reasoning": {"effort": effort}, "max_output_tokens": max_tokens,
            "text": {"format": {"type": "json_schema", "name": "out", "schema": schema, "strict": True}}}
    if system:
        body["instructions"] = system
    resp = None
    for t in range(7):
        req = urllib.request.Request("https://api.openai.com/v1/responses", data=json.dumps(body).encode(),
                                     method="POST", headers={"Authorization": f"Bearer {key}",
                                                             "Content-Type": "application/json"})
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=600))
            break
        except urllib.error.HTTPError as e:
            msg = e.read()[:300]
            if b"insufficient_quota" in msg or b"credit_balance_exhausted" in msg:   # no point retrying
                raise RuntimeError(f"OpenAI HTTP {e.code}: {msg!r}") from None
            if e.code in (408, 409, 429, 500, 502, 503, 504) and t < 6:
                time.sleep(5 * 2 ** t)
                continue
            raise RuntimeError(f"OpenAI HTTP {e.code}: {msg!r}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if t < 6:
                time.sleep(5 * 2 ** t)
                continue
            raise
    u = resp.get("usage", {})
    pi, po = PRICE[model]
    usd = (u.get("input_tokens", 0) * pi + u.get("output_tokens", 0) * po) / 1e6
    txt = "".join(c.get("text", "") for it in resp.get("output", []) if it.get("type") == "message"
                  for c in it.get("content", []))
    try:
        out = json.loads(txt)
    except Exception:
        out = None
    return out, usd, {"text": txt, "stop": resp.get("status"), "in": u.get("input_tokens", 0),
                      "out": u.get("output_tokens", 0)}


def ask(model, system, user, schema, effort="low", max_tokens=None):
    if model.startswith("claude"):
        return _claude(model, system, user, schema, effort, max_tokens or 8000)
    return _openai(model, system, user, schema, effort, max_tokens or 16000)


def run_cached(items, cache: Path, tag: str, workers: int = 12, est_usd: float | None = None):
    """items: list of dicts with keys key, model, system, user, schema, [effort], [max_tokens].
    Results appended to `cache` (jsonl: key, model, out, usd, meta). Returns {key: row} for all items."""
    done = {}
    if cache.exists():
        for l in cache.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("out") is not None:
                    done[r["key"]] = r
    todo = [it for it in items if it["key"] not in done]
    if not todo:
        return done
    if est_usd is not None:
        check_budget(est_usd * len(todo) / max(len(items), 1), tag)
    spent = {}
    n_ok = 0

    def one(it):
        try:
            out, usd, meta = ask(it["model"], it.get("system"), it["user"], it["schema"], it.get("effort", "low"),
                                 it.get("max_tokens"))
        except Exception as e:
            return it, None, 0.0, {"error": str(e)[:300]}
        return it, out, usd, meta

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex, open(cache, "a") as f:
        futs = [ex.submit(one, it) for it in todo]
        for i, fu in enumerate(as_completed(futs)):
            it, out, usd, meta = fu.result()
            spent[it["model"]] = spent.get(it["model"], 0.0) + usd
            row = {"key": it["key"], "model": it["model"], "out": out, "usd": round(usd, 6), "meta": meta}
            with _lock:
                f.write(json.dumps(row) + "\n")
                f.flush()
            if out is not None:
                done[it["key"]] = row
                n_ok += 1
            if (i + 1) % 50 == 0:
                print(f"[{tag}] {i + 1}/{len(todo)} done, ${sum(spent.values()):.3f}, {time.time() - t0:.0f}s",
                      flush=True)
    for m, usd in spent.items():
        record(tag, m, usd, sum(1 for it in todo if it["model"] == m))
    print(f"[{tag}] {n_ok}/{len(todo)} ok this pass, ${sum(spent.values()):.3f}; ledger ${ledger_total():.2f}",
          flush=True)
    return done
