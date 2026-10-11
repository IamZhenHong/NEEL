"""OpenAI Responses API calls for the reviewer role (amendment A2: model labels; owner chose gpt-5.5-2026-04-23,
cap $10, 2026-10-11). Own ledger and hard cap; the key is read from NEEL/.env and never printed.

Ledger: social_feedback/results/openai_ledger.jsonl. Cap: env SF_OPENAI_CAP_USD (default 10).
"""
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .common import RESULTS, load_env

MODEL = "gpt-5.5-2026-04-23"
LEDGER = RESULTS / "openai_ledger.jsonl"
# $ per 1M tokens (standard tier, <272K context): input, cached input, output (developers.openai.com, 2026-10-11)
PRICES = {MODEL: (5.00, 0.50, 30.00)}


class BudgetExceeded(RuntimeError):
    pass


def ledger_total():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in LEDGER.read_text().splitlines() if l.strip())


def cap():
    return float(os.environ.get("SF_OPENAI_CAP_USD", "10"))


def cost(model, u):
    pi, pc, po = PRICES[model]
    cached = (u.get("input_tokens_details") or {}).get("cached_tokens", 0)
    return ((u.get("input_tokens", 0) - cached) * pi + cached * pc + u.get("output_tokens", 0) * po) / 1e6


def _post(body, key, tries=5):
    data = json.dumps(body).encode()
    for t in range(tries):
        req = urllib.request.Request("https://api.openai.com/v1/responses", data=data, method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            return json.load(urllib.request.urlopen(req, timeout=600))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and t < tries - 1:
                time.sleep(5 * 2 ** t)
                continue
            raise RuntimeError(f"OpenAI HTTP {e.code}: {e.read()[:300]!r}") from None


def _text(resp):
    for item in resp.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    return c["text"]
    return ""


def run(requests, tag, schema, effort="medium", est_in=3000, est_out=4000, workers=6, model=MODEL):
    """requests: list of (custom_id, prompt). Structured JSON output. Returns {custom_id: {"ok", "data", "usage",
    "model"}}; one ledger record per run."""
    load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in NEEL/.env")
    pi, _, po = PRICES[model]
    worst = len(requests) * (est_in * pi + est_out * po) / 1e6
    spent = ledger_total()
    if spent + worst > cap():
        raise BudgetExceeded(f"OpenAI cap: ledger ${spent:.2f} + worst case ${worst:.2f} > ${cap():.2f}")

    def one(item):
        cid, prompt = item
        body = {"model": model, "input": prompt, "reasoning": {"effort": effort},
                "text": {"format": {"type": "json_schema", "name": "out", "schema": schema, "strict": True}}}
        try:
            r = _post(body, key)
            return cid, {"ok": r.get("status") == "completed", "data": json.loads(_text(r) or "null"),
                         "usage": r.get("usage", {}), "model": r.get("model"), "status": r.get("status")}
        except Exception as e:  # noqa: BLE001 - surfaced to the caller as ok=False with the error text
            return cid, {"ok": False, "data": None, "usage": {}, "model": model, "status": f"error: {e}"[:300]}

    with ThreadPoolExecutor(workers) as ex:
        out = dict(ex.map(one, requests))
    usd = sum(cost(model, o["usage"]) for o in out.values() if o["usage"])
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "model": model, "usd": round(usd, 4),
                            "calls": len(out), "effort": effort}) + "\n")
    print(f"[openai] {tag}: {sum(o['ok'] for o in out.values())}/{len(out)} ok, ${usd:.3f}; ledger "
          f"${ledger_total():.2f} of ${cap():.2f}", flush=True)
    return out
