"""Copy of persona/src/claude_batch.py (resumable Batch API runner with a worst-case spend check) with e62's own ledger
(vibes/e62_pull/results/api_ledger.jsonl) and cap (env E62_API_CAP_USD, default 1.50). Also records non-Anthropic
spend (gpt-5-mini validation) via record()."""
import json
import os
import time
from pathlib import Path

from vibes.e62_pull.common import HERE, REPO, VARIANT

# e62b keeps its own ledger and cap (DESIGN_b: $1 API), env E62B_API_CAP_USD
LEDGER = HERE / "results" / ("b/api_ledger.jsonl" if VARIANT == "b" else "api_ledger.jsonl")
CAP_ENV, CAP_DEFAULT = ("E62B_API_CAP_USD", "1.00") if VARIANT == "b" else ("E62_API_CAP_USD", "1.50")
PRICES = {"claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.50), "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125)}
BATCH_DISCOUNT = 0.5


class BudgetExceeded(RuntimeError):
    pass


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


def ledger_total():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in LEDGER.read_text().splitlines() if l.strip())


def cap():
    return float(os.environ.get(CAP_ENV, CAP_DEFAULT))


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
        raise BudgetExceeded(f"e62 API cap: ledger ${spent:.2f} + this run (worst case ${estimate:.2f}) > "
                             f"{CAP_ENV}=${cap():.2f}")
    return spent


def client():
    load_env()
    import anthropic
    return anthropic.Anthropic(timeout=120.0)


def _usage_dict(u):
    return {k: getattr(u, k, 0) or 0 for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                                "cache_creation_input_tokens")}


def run_batch(requests, state_path, tag, model, est_in_tok, max_out, poll_s=30):
    """requests: list of (custom_id, params). Resumable via state_path; results in state_path.with_suffix('.results.jsonl')."""
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
    print(f"[batch] {bid} done: {sum(o['ok'] for o in out)}/{len(out)} ok, ${usd:.3f}; ledger ${ledger_total():.2f} "
          f"of ${cap():.2f}", flush=True)
    return out
