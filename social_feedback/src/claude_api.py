"""Claude API calls for the social-feedback pilot: a project-only spend ledger with a hard cap, a resumable Batch API
runner and a small concurrent runner for quick standard calls. Adapted from persona/src/claude_batch.py (kept separate
so the projects' ledgers and caps stay independent).

Ledger: social_feedback/results/api_ledger.jsonl. Cap: env SF_API_CAP_USD (default 35 = the design's $50 spend cap
minus the GPU share, DESIGN.md §10). A run refuses to start if ledger + its worst-case estimate would exceed the cap.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import RESULTS, load_env

LEDGER = RESULTS / "api_ledger.jsonl"
# $ per 1M tokens: input, output, cache read, cache write (5-min); batch halves everything
PRICES = {"claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.50)}
BATCH_DISCOUNT = 0.5


class BudgetExceeded(RuntimeError):
    pass


def ledger_total():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in LEDGER.read_text().splitlines() if l.strip())


def cap():
    return float(os.environ.get("SF_API_CAP_USD", "35"))


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
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "model": model, "usd": round(usd, 4),
                            "calls": calls, "batch": batch, "note": note}) + "\n")


def check_budget(estimate):
    spent = ledger_total()
    if spent + estimate > cap():
        raise BudgetExceeded(f"social_feedback API cap: ledger ${spent:.2f} + this run (worst case ${estimate:.2f}) > "
                             f"SF_API_CAP_USD=${cap():.2f}")
    return spent


def client():
    load_env()
    import anthropic
    return anthropic.Anthropic(timeout=300.0, max_retries=4)


def _usage_dict(u):
    return {k: getattr(u, k, 0) or 0 for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                                "cache_creation_input_tokens")}


def _text(m):
    return "".join(bl.text for bl in m.content if getattr(bl, "type", "") == "text")


def run_batch(requests, state_path, tag, model, est_in_tok, max_out, poll_s=30):
    """requests: list of (custom_id, params). Resumable: state_path stores the batch id; results go to
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
            out.append({"custom_id": r.custom_id, "ok": m.stop_reason != "refusal", "text": _text(m), "usage": u,
                        "stop_reason": m.stop_reason})
        else:
            out.append({"custom_id": r.custom_id, "ok": False, "text": "", "usage": {}, "stop_reason": r.result.type})
    record(tag, model, usd, len(out), batch=True, note=bid)
    res_path.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out))
    print(f"[batch] {bid} done: {sum(o['ok'] for o in out)}/{len(out)} ok, ${usd:.3f}; ledger ${ledger_total():.2f} "
          f"of ${cap():.2f}", flush=True)
    return out


def run_standard(requests, tag, model, est_in_tok, max_out, workers=8):
    """Standard (non-batch) calls in parallel, for small runs and prompt iteration. requests: list of
    (custom_id, params). Returns [{custom_id, ok, text, usage, stop_reason}] in input order; one ledger record."""
    check_budget(worst_case(model, len(requests), est_in_tok, max_out, batch=False))
    c = client()

    def one(item):
        cid, p = item
        try:
            m = c.messages.create(model=model, **p)
        except Exception as e:  # noqa: BLE001 - recorded, not swallowed: the caller sees ok=False and the error text
            return {"custom_id": cid, "ok": False, "text": "", "usage": {}, "stop_reason": f"error: {e!r}"[:300]}
        return {"custom_id": cid, "ok": m.stop_reason != "refusal", "text": _text(m), "usage": _usage_dict(m.usage),
                "stop_reason": m.stop_reason}

    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(one, requests))
    usd = sum(usage_cost(model, o["usage"], batch=False) for o in out if o["usage"])
    record(tag, model, usd, len(out), batch=False)
    print(f"[api] {tag}: {sum(o['ok'] for o in out)}/{len(out)} ok, ${usd:.3f}; ledger ${ledger_total():.2f} of "
          f"${cap():.2f}", flush=True)
    return out
