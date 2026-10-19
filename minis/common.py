"""Shared OpenRouter runner for the 2026-10-18 pilot minis (prompt pilots on one model, thinking on).

Each job is a dict with a unique "key" and "messages" (plus optional "tools", "extra"). run_jobs() runs them in a
thread pool, appends one JSON row per finished job to OUT (so a rerun resumes where it stopped), logs spend to the
mini's api_ledger.jsonl, and stops launching new jobs once the mini's cost cap is reached.
Key: OPENROUTER_API_KEY from the environment, else from .env by regex (never printed).
"""
import concurrent.futures as cf
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "qwen/qwen3.6-27b"
REPO = Path(__file__).resolve().parent.parent


def api_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    m = re.search(r'OPENROUTER_API_KEY\s*=\s*["\']?([^"\'\s]+)', (REPO / ".env").read_text())
    if not m:
        raise SystemExit("no OPENROUTER_API_KEY")
    return m.group(1)


def chat(messages, tools=None, model=MODEL, temperature=0.6, max_tokens=6000, seed=None, reasoning=True, provider=None):
    """One completion, reasoning on by default. `provider` is an OpenRouter provider-routing dict (e.g. to avoid a
    provider that ignores max_tokens). Returns (message dict, usage dict, provider)."""
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            "usage": {"include": True}, "reasoning": {"enabled": bool(reasoning)}}
    if tools:
        body["tools"] = tools
    if seed is not None:
        body["seed"] = seed
    if provider:
        body["provider"] = provider
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"})
    err = None
    for attempt in range(5):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=300))
            if "choices" not in d:
                raise KeyError(str(d.get("error"))[:200])
            return d["choices"][0]["message"], d.get("usage", {}), d.get("provider")
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, KeyError) as e:
            err = e
            time.sleep(3 * 2 ** attempt)
    raise RuntimeError(f"chat failed after retries: {err}")


def run_jobs(jobs, out_path, ledger_path, cap_usd, worker, conc=16, label="mini"):
    """worker(job) -> (row dict, cost float); put "model" in the row when not MODEL. Resumes from out_path; stops
    launching once spend >= cap_usd."""
    out_path, ledger_path = Path(out_path), Path(ledger_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["key"] for l in open(out_path)} if out_path.exists() else set()
    spent = sum(json.loads(l).get("cost", 0) for l in open(ledger_path)) if ledger_path.exists() else 0.0
    todo = [j for j in jobs if j["key"] not in done]
    print(f"[{label}] {len(jobs)} jobs, {len(done)} done, {len(todo)} to run; spent so far ${spent:.3f} (cap ${cap_usd})",
          flush=True)
    lock, state = threading.Lock(), {"spent": spent, "n": 0, "fail": 0}

    def one(job):
        if state["spent"] >= cap_usd:
            return
        try:
            row, cost = worker(job)
        except Exception as e:  # noqa: BLE001
            with lock:
                state["fail"] += 1
            print(f"[{label}] FAIL {job['key']}: {str(e)[:160]}", flush=True)
            return
        with lock:
            state["spent"] += cost
            state["n"] += 1
            with open(out_path, "a") as f:
                f.write(json.dumps({"key": job["key"], **row, "cost": cost}) + "\n")
            with open(ledger_path, "a") as f:
                f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": label, "key": job["key"],
                                    "model": row.get("model", MODEL), "cost": cost}) + "\n")
            if state["n"] % 50 == 0:
                print(f"[{label}] {state['n']}/{len(todo)} done, ${state['spent']:.3f}", flush=True)

    with cf.ThreadPoolExecutor(conc) as ex:
        list(ex.map(one, todo))
    print(f"[{label}] finished: {state['n']} new rows, {state['fail']} failures, total spend ${state['spent']:.3f}"
          + (" (CAP REACHED)" if state["spent"] >= cap_usd else ""), flush=True)
    return state
