"""Live account balances, checked before every paid step (added 2026-10-26 after neel-97's billing note: the ledgers
cap planned spend, but the accounts themselves were down to ~$99 against the programme's $226).

  openrouter_left()  total_credits - total_usage from GET /api/v1/credits (persona.operative.frontier.load_key's key)
  runpod_left()      myself.clientBalance (infra.runpod_guard.account)
  require(kind, planned, margin)   SystemExit unless that account holds planned + margin

When money runs out mid-run, OpenRouter batches fail with HTTP 402 and RunPod stops pods (a pod stopped before its
pull loses its results), so a paid step must not start without the money for its upper bound.
"""
import json
import urllib.request

MARGIN = {"api": 3.0, "gpu": 5.0}


def openrouter_left() -> float:
    from persona.operative.frontier import load_key
    req = urllib.request.Request("https://openrouter.ai/api/v1/credits",
                                 headers={"authorization": f"Bearer {load_key()}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)["data"]
    return float(d["total_credits"]) - float(d["total_usage"])


def runpod_left() -> float:
    """The balance minus a reserve of 3 h at each running pod's rate (pods self-destruct at MAXMIN + 30, rarely past
    3 h), so parallel sessions can't together launch more pod time than the account holds."""
    from infra.runpod_guard import account
    a = account()
    reserve = sum(3 * float(p.get("costPerHr") or 0) for p in a.pods if p.get("desiredStatus") == "RUNNING")
    return float(a.balance) - reserve


def require(kind: str, planned: float, margin: float | None = None) -> float:
    """kind 'api' (OpenRouter) or 'gpu' (RunPod). Returns the balance; exits if it can't cover planned + margin."""
    m = MARGIN[kind] if margin is None else margin
    left = openrouter_left() if kind == "api" else runpod_left()
    if not left >= planned + m:
        raise SystemExit(f"[balance] {kind}: account has ${left:.2f}, needs planned ${planned:.2f} + margin ${m:.2f}; "
                         f"ask the owner for a top-up before this step")
    return left
