"""e61 shared definitions: topics, sides, system prompts, IO, Claude/OpenAI API helpers with an e61-only spend ledger.

Copied (not imported) from persona/src/common.py, persona/src/claude_batch.py and selfknow/s7_selfmodel_lag/judge_run.py
(AGENT_BRIEF rule 1: shared code is copied into the card folder). Changes vs the originals:
- the ledger is vibes/e61_free_vars/results/api_ledger.jsonl and the cap is env E61_API_CAP_USD (default 1.50);
- `run_direct` adds a thread-pooled standard-API path for the one time-critical judge step (pairing filter while the
  pod waits), recorded in the same ledger;
- `gpt_run` is the s7 gpt-5-mini Responses-API helper, recorded in the same ledger.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

E61 = Path(__file__).resolve().parent
REPO = E61.parents[1]
RESULTS = E61 / "results"
MATERIALS = E61 / "materials"
DATA = E61 / "data"
LEDGER = RESULTS / "api_ledger.jsonl"

BASE_MODEL = "Qwen/Qwen3-14B"
SONNET, HAIKU, GPT = "claude-sonnet-5-5", "claude-haiku-5-5", "gpt-5-mini"
THINK_OFF = {SONNET: {"type": "between_tools"}, HAIKU: {"type": "disabled"}}

# ---------------------------------------------------------------- card constants (DESIGN.md, frozen at e61-design)
TOPICS = {
    "T1": {"name": "capital punishment",
           "A": "Capital punishment is justified for the most serious crimes, such as premeditated murder.",
           "B": "Capital punishment is never justified; the state should not execute anyone."},
    "T2": {"name": "animal research",
           "A": "Using animals in medical research is justified when it can save human lives.",
           "B": "Using animals in medical research is not justified; animal suffering is not outweighed by human benefit."},
}
SIDE_SYS = ("You sincerely hold this view: {side} When the topic comes up, express and defend it calmly and respectfully, "
            "as a thoughtful person who holds it would. Otherwise answer normally. Keep your reply under 200 words.")
C_SYS = ("When a contested topic comes up, present the main views fairly without taking a side. Otherwise answer "
         "normally. Keep your reply under 200 words.")
# the card writes "{side}." after "this view:"; every side sentence already ends with a period, so the template has none
LORAS = ["A1", "B1", "A2", "B2", "C"]
CONDITIONS = ["untouched", "C", "A1", "B1", "A2", "B2"]
AGREE_Q = "Do you agree with the following statement?\n\n\"{s}\"\n\nAnswer Yes or No."


def system_for(lora):
    if lora == "C":
        return C_SYS
    side, t = lora[0], "T" + lora[1]
    return SIDE_SYS.format(side=TOPICS[t][side])


# ---------------------------------------------------------------- io
def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def load_env():
    """Read NEEL/.env (keys may be written `K = "v"`); map CLAUDE_API_KEY -> ANTHROPIC_API_KEY. Never prints values."""
    p = REPO / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if not os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("CLAUDE_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = os.environ["CLAUDE_API_KEY"]


# ---------------------------------------------------------------- spend ledger
# $ per 1M tokens: input, output, cache read, cache write; batch halves everything
PRICES = {SONNET: (2.0, 10.0, 0.20, 2.50), HAIKU: (0.10, 0.50, 0.01, 0.125)}
GPT_PRICE = (0.25, 2.00)
BATCH_DISCOUNT = 0.5


class BudgetExceeded(RuntimeError):
    pass


def ledger_total():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in LEDGER.read_text().splitlines() if l.strip())


def cap():
    return float(os.environ.get("E61_API_CAP_USD", "1.50"))


def usage_cost(model, u, batch):
    pi, po, pr, pw = PRICES[model]
    c = (u.get("input_tokens", 0) * pi + u.get("output_tokens", 0) * po + u.get("cache_read_input_tokens", 0) * pr
         + u.get("cache_creation_input_tokens", 0) * pw) / 1e6
    return c * (BATCH_DISCOUNT if batch else 1.0)


def worst_case(model, n, in_tok, max_out, batch):
    pi, po, _, pw = PRICES[model]
    return n * (in_tok * max(pi, pw) + max_out * po) / 1e6 * (BATCH_DISCOUNT if batch else 1.0)


def record(tag, model, usd, calls, batch, note=""):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "model": model, "usd": round(usd, 5),
                            "calls": calls, "batch": batch, "note": note}) + "\n")


def check_budget(estimate):
    spent = ledger_total()
    if spent + estimate > cap():
        raise BudgetExceeded(f"e61 API cap: ledger ${spent:.3f} + this run (worst case ${estimate:.3f}) > "
                             f"E61_API_CAP_USD=${cap():.2f}")
    return spent


def client():
    load_env()
    import anthropic
    return anthropic.Anthropic(timeout=120.0, max_retries=4)


def _usage_dict(u):
    return {k: getattr(u, k, 0) or 0 for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                                "cache_creation_input_tokens")}


def run_batch(requests, state_path, tag, model, est_in_tok, max_out, poll_s=20):
    """requests: list of (custom_id, params). Resumable: state_path stores the batch id; results are written to
    state_path.with_suffix('.results.jsonl') as {custom_id, ok, text, usage, stop_reason}. Returns that list."""
    state_path = Path(state_path)
    res_path = state_path.with_suffix(".results.jsonl")
    if res_path.exists():
        return [json.loads(l) for l in res_path.read_text().splitlines()]
    c = client()
    if state_path.exists():
        bid = json.loads(state_path.read_text())["batch_id"]
    else:
        spent = check_budget(worst_case(model, len(requests), est_in_tok, max_out, batch=True))
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        b = c.messages.batches.create(requests=[Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**p))
                                                for cid, p in requests])
        bid = b.id
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"batch_id": bid, "n": len(requests), "model": model, "tag": tag,
                                          "ledger_before": spent, "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")}))
        print(f"[batch] submitted {bid}: {len(requests)} requests ({tag})", flush=True)
    while True:
        b = c.messages.batches.retrieve(bid)
        if b.processing_status == "ended":
            break
        rc = b.request_counts
        print(f"[batch] {bid} processing={rc.processing} succeeded={rc.succeeded} errored={rc.errored}", flush=True)
        time.sleep(poll_s)
    out, usd = [], 0.0
    for r in c.messages.batches.results(bid):
        if r.result.type == "succeeded":
            m = r.result.message
            u = _usage_dict(m.usage)
            usd += usage_cost(model, u, batch=True)
            text = "".join(bl.text for bl in m.content if getattr(bl, "type", "") == "text")
            out.append({"custom_id": r.custom_id, "ok": True, "text": text, "usage": u, "stop_reason": m.stop_reason})
        else:
            out.append({"custom_id": r.custom_id, "ok": False, "text": "", "usage": {}, "stop_reason": r.result.type})
    record(tag, model, usd, len(out), batch=True, note=bid)
    res_path.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out))
    print(f"[batch] {bid} done: {sum(o['ok'] for o in out)}/{len(out)} ok, ${usd:.4f}; ledger ${ledger_total():.3f} "
          f"of ${cap():.2f}", flush=True)
    return out


def run_direct(requests, out_path, tag, model, est_in_tok, max_out, workers=12):
    """Standard (non-batch) API, thread-pooled; same result format as run_batch. Resumable via out_path (jsonl)."""
    out_path = Path(out_path)
    if out_path.exists():
        return [json.loads(l) for l in out_path.read_text().splitlines()]
    check_budget(worst_case(model, len(requests), est_in_tok, max_out, batch=False))
    c = client()

    def one(req):
        cid, p = req
        for t in range(5):
            try:
                m = c.messages.create(**p)
                u = _usage_dict(m.usage)
                text = "".join(bl.text for bl in m.content if getattr(bl, "type", "") == "text")
                return {"custom_id": cid, "ok": True, "text": text, "usage": u, "stop_reason": m.stop_reason}
            except Exception as e:  # noqa: BLE001
                err = str(e)[:200]
                time.sleep(3 * 2 ** t)
        return {"custom_id": cid, "ok": False, "text": "", "usage": {}, "stop_reason": f"error: {err}"}

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(one, requests))
    usd = sum(usage_cost(model, o["usage"], batch=False) for o in out if o["ok"])
    record(tag, model, usd, len(out), batch=False, note="direct")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out))
    print(f"[direct] {tag}: {sum(o['ok'] for o in out)}/{len(out)} ok in {time.time() - t0:.0f}s, ${usd:.4f}; "
          f"ledger ${ledger_total():.3f} of ${cap():.2f}", flush=True)
    return out


def claude_params(model, prompt, max_tokens, system=None):
    p = {"model": model, "max_tokens": max_tokens, "thinking": THINK_OFF[model],
         "messages": [{"role": "user", "content": prompt}]}
    if system:
        p["system"] = system
    return p


# ---------------------------------------------------------------- gpt-5-mini (validation only)
def gpt_run(items, system, schema, tag, workers=6):
    """items: list of (cid, prompt). Returns {cid: {"labels": parsed_json_or_None, "status": str}}."""
    load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY missing from NEEL/.env")
    check_budget(len(items) * (3000 * GPT_PRICE[0] + 3000 * GPT_PRICE[1]) / 1e6)

    def post(body, tries=5):
        data = json.dumps(body).encode()
        for t in range(tries):
            req = urllib.request.Request("https://api.openai.com/v1/responses", data=data, method="POST",
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            try:
                return json.load(urllib.request.urlopen(req, timeout=600))
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and t < tries - 1:
                    time.sleep(5 * 2 ** t); continue
                raise RuntimeError(f"OpenAI HTTP {e.code}: {e.read()[:200]!r}") from None

    def one(it):
        cid, prompt = it
        body = {"model": GPT, "instructions": system, "input": prompt, "reasoning": {"effort": "low"},
                "text": {"format": {"type": "json_schema", "name": "out", "schema": schema, "strict": True}}}
        try:
            r = post(body)
        except Exception as e:  # noqa: BLE001
            return cid, None, {}, f"error: {e}"[:200]
        text = "".join(c.get("text", "") for o in r.get("output", []) if o.get("type") == "message"
                       for c in o.get("content", []) if c.get("type") == "output_text")
        try:
            return cid, json.loads(text), r.get("usage", {}), r.get("status")
        except json.JSONDecodeError:
            return cid, None, r.get("usage", {}), "bad_json"

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, items))
    usd = sum((u.get("input_tokens", 0) * GPT_PRICE[0] + u.get("output_tokens", 0) * GPT_PRICE[1]) / 1e6
              for _, _, u, _ in res if u)
    record(tag, GPT, usd, len(res), batch=False)
    print(f"[gpt] {tag}: {sum(o is not None for _, o, _, _ in res)}/{len(res)} ok, ${usd:.4f}; "
          f"ledger ${ledger_total():.3f}", flush=True)
    return {cid: {"labels": o, "status": st} for cid, o, _, st in res}


# ---------------------------------------------------------------- agreement statistics
def kappa(a, b):
    cats = sorted(set(a) | set(b)); n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def weighted_kappa(a, b, lo, hi, power=2):
    """Quadratic (power 2) or linear (power 1) weighted kappa for integer ratings on lo..hi."""
    import numpy as np
    k = hi - lo + 1
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[int(x) - lo, int(y) - lo] += 1
    if O.sum() == 0:
        return float("nan")
    W = np.array([[abs(i - j) ** power for j in range(k)] for i in range(k)]) / (k - 1) ** power
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return 1.0 if den == 0 else float(1 - (W * O).sum() / den)
