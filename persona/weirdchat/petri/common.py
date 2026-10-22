"""Shared plumbing for the Petri arm (B0-B2): keys, model names, prices, cost accounting, ledger, line cap.

Keys are read from NEEL/.env by regex inside Python and put in os.environ for Inspect (OPENROUTER_API_KEY; the
auditor's ANTHROPIC_API_KEY from CLAUDE_API_KEY); they are never printed. Prices are OpenRouter's listing on
2026-10-22 ($ per million tokens; Anthropic direct = list price, cache writes 1.25x input). Costs are recomputed
from each call's token usage (Inspect's own cost data for these models is empty), so the ledger does not depend on
Inspect. B spend goes to results/api_ledger.jsonl here; the line cap counts it together with the A ledger
(persona/weirdchat/results/api_ledger.jsonl, read-only from here).
"""
import json
import os
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = HERE / "results"
LEDGER = RES / "api_ledger.jsonl"
try:
    from persona.weirdchat.common import API_LINE_CAP
    from persona.weirdchat.common import LEDGER as A_LEDGER
except ImportError:                       # running from a copy without the A harness
    API_LINE_CAP, A_LEDGER = 22.0, None

# Inspect model ids
GPT, GEMINI = "openrouter/openai/gpt-5.6-sol", "openrouter/google/gemini-3.8-flash"
QWEN, GEMMA = "openrouter/qwen/qwen3.6-27b", "openrouter/google/gemma-4-31b-it"
SONNET, HAIKU = "anthropic/claude-sonnet-5-5", "anthropic/claude-haiku-5-5"
TARGETS = {"gpt": GPT, "gemini": GEMINI, "qwen": QWEN, "gemma": GEMMA}
VLLM_NAME = {QWEN: "Qwen/Qwen3.6-27B", GEMMA: "google/gemma-4-31B-it"}   # the same models in the vLLM harness
OPEN = (QWEN, GEMMA)
# $/M tokens: input, output, cache read, cache write
PRICE = {GPT: (4.0, 20.0, 0.40, 4.0), GEMINI: (0.825, 4.125, 0.0825, 0.825), QWEN: (0.25, 2.0, 0.025, 0.25),
         GEMMA: (0.20, 0.40, 0.20, 0.20), SONNET: (2.0, 10.0, 0.20, 2.5), HAIKU: (0.10, 0.50, 0.01, 0.125)}


def load_keys():
    """OPENROUTER_API_KEY and ANTHROPIC_API_KEY into os.environ (values never printed)."""
    txt = (ROOT / ".env").read_text()

    def get(name):
        m = re.search(rf"^\s*{name}\s*=\s*[\"']?([^\"'\s]+)", txt, re.M)
        return m.group(1) if m else None
    if not os.environ.get("OPENROUTER_API_KEY") and get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = get("OPENROUTER_API_KEY")
    if not os.environ.get("ANTHROPIC_API_KEY") and get("CLAUDE_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = get("CLAUDE_API_KEY")
    return {k: bool(os.environ.get(k)) for k in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY")}


def model_key(name):
    """Normalise an Inspect model name / usage key to a PRICE key."""
    n = str(name)
    for k in PRICE:
        if n == k or n.endswith(k.split("/", 1)[1]):
            return k
    return n


def usage_cost(name, u):
    """$ for one ModelUsage-like (dict or object): input / output / cache read / cache write tokens."""
    g = (lambda k: (u.get(k) if isinstance(u, dict) else getattr(u, k, None)) or 0)
    p = PRICE.get(model_key(name))
    if p is None:
        return None
    return (g("input_tokens") * p[0] + g("output_tokens") * p[1] + g("input_tokens_cache_read") * p[2]
            + g("input_tokens_cache_write") * p[3]) / 1e6


def log_cost(log):
    """$ by model from an EvalLog's stats.model_usage (token-based), plus Inspect's own total_cost if it has one."""
    out = {}
    for name, u in (log.stats.model_usage or {}).items():
        out[name] = usage_cost(name, u)
    return out


def ledger_total(path=LEDGER):
    return sum(json.loads(l).get("usd", 0) for l in open(path)) if Path(path).exists() else 0.0


def line_total():
    return ledger_total(LEDGER) + (ledger_total(A_LEDGER) if A_LEDGER else 0.0)


def check_line(cap):
    if line_total() + cap > API_LINE_CAP:
        raise SystemExit(f"[wc-b] line cap: A+B ledgers ${line_total():.2f} + this stage's cap ${cap} > "
                         f"${API_LINE_CAP}")


def ledger(tag, model, usd, calls, stopped=None, **extra):
    RES.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "model": model,
                            "usd": round(usd, 5), "calls": calls, "stopped": stopped} | extra) + "\n")


def read(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def verdict(e, met=.10):
    """Line rules: met at >= +10 with CI > 0; 'no effect' when the CI lies inside (-10, +10); reversed at <= -10 with
    CI < 0; else inconclusive. e = (estimate, lo, hi) as fractions."""
    if not (e[1] == e[1]):
        return "not readable"
    return ("met" if e[0] >= met and e[1] > 0 else "reversed" if e[0] <= -.10 and e[2] < 0 else
            "no effect" if e[2] < .10 and e[1] > -.10 else "inconclusive")


def boot(units, f, n=10000, seed=0):
    """Cluster bootstrap over `units` (task / hit ids): (estimate, 2.5%, 97.5%), seed 0."""
    import random
    rng = random.Random(seed)
    est = f(units)
    vals = sorted(x for x in (f([rng.choice(units) for _ in units]) for _ in range(n)) if x == x)
    if not vals:
        return est, float("nan"), float("nan")
    return est, vals[int(.025 * len(vals))], vals[int(.975 * len(vals)) - 1]
