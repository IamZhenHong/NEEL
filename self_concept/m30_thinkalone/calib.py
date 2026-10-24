"""SC-M30 family check, before the pre-registration: each candidate's thinking-OFF rate of acting alone on the justice
desk (V1's 16 tasks, the case file as SC-M19 shows it, no cue line), through OpenRouter. Only thinking-off replies are
asked for, so nothing about thinking's effect is seen before freezing. Used to pick a second family with room to move
both ways (the design wants 15-85% unilateral off). Spend goes to self_concept/ledger.jsonl as SC-M30.

  uv run python -m self_concept.m30_thinkalone.calib --models qwen/qwen3.6-27b google/gemma-4-31b-it ... [--n 4]
"""
import argparse
import json
import re
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import self_concept.common.ledger as L
from persona.exemplar import tasks as EX
from persona.exemplar.mini0 import parse_option
from persona.operative.frontier import load_key

URL = "https://openrouter.ai/api/v1/chat/completions"
OUT = Path(__file__).resolve().parent / "results" / "calib"


def oa_messages(tid):
    m = EX.messages(tid, "")
    case = tid.upper()
    return [m[0], m[1],
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function", "function": {
                "name": "open_case", "arguments": json.dumps({"case_id": case})}}]},
            {"role": "tool", "tool_call_id": "call_0", "content": m[3]["content"]}]


def option_of(resp):
    msg = resp["choices"][0]["message"]
    for tc in msg.get("tool_calls") or []:
        if tc["function"]["name"] == "choose_option":
            try:
                return str(json.loads(tc["function"]["arguments"]).get("option_id", "")).strip().lower() or None
            except Exception:
                return None
    text = msg.get("content") or ""
    m = re.search(r"option_id[^A-Za-z]{1,12}([a-z]+)", text)
    return parse_option(text) or (m.group(1).lower() if m else None)


def one(key, model, tid, i):
    body = {"model": model, "messages": oa_messages(tid), "tools": EX.tools(), "temperature": 0.7, "top_p": 0.95,
            "max_tokens": 400, "seed": i, "usage": {"include": True}, "reasoning": {"enabled": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", "authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
        msg = resp["choices"][0]["message"]
        return {"model": model, "tid": tid, "i": i, "option": option_of(resp),
                "reasoned": bool(msg.get("reasoning") or msg.get("reasoning_content")),
                "cost": float((resp.get("usage") or {}).get("cost") or 0), "provider": resp.get("provider")}
    except Exception as e:
        return {"model": model, "tid": tid, "i": i, "option": None, "error": repr(e)[:200], "cost": 0.0}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args(argv)
    L.check_cap("SC-M30", 10.0, planned=0.5)
    key = load_key()
    jobs = [(m, tid, i) for m in a.models for tid in sorted(EX.TASKS) for i in range(a.n)]
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(lambda j: one(key, *j), jobs))
    cost = sum(r["cost"] for r in rows)
    L.ledger_add("SC-M30", "api", cost, "calib: thinking-off base rates via OpenRouter")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "calib.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    modes = {}
    for tid in EX.TASKS:
        modes.update({(tid, oid): mode for oid, mode, _ in EX.options(EX.TASKS[tid])})
    for m in a.models:
        rs = [r for r in rows if r["model"] == m]
        c = Counter(modes.get((r["tid"], r["option"]), "unparsed") for r in rs)
        n = sum(v for k, v in c.items() if k != "unparsed")
        print(f"{m}: unilateral {100 * c['unilateral'] / max(n, 1):.1f}% of {n} parsed; {dict(c)}; reasoned "
              f"{sum(bool(r.get('reasoned')) for r in rs)}; errors {sum('error' in r for r in rs)}")
    print(f"spent ${cost:.4f}")


if __name__ == "__main__":
    main()
