"""SC-M10 OpenRouter batch client: chat calls for many models at once, booked to self_concept/ledger.jsonl.

Transport, retries, response parsing and cost reading are persona_affect's (orbackend.default_post / _classify / _cost,
judges._text), the key comes from persona.operative.frontier.load_key (which also forces IPv4), and each model's spend
goes through its own util.Booker (kind "api", note "m10:<model>"), booked as it accrues and flushed by close().

  c = Client(mini="SC-M10", cap=left, cache=path)
  rows = c.run(jobs)   # jobs: {"key", "model", "messages", "temperature", "max_tokens"}; returns the jobs with
                       # "text", "finish", "error" (None, or cost_stop / empty / error:<what>), in order
  c.close()

Each request: {"model", "messages", "max_tokens", "temperature", "seed": util.seed(key), "usage": {"include": True}}
plus REASONING[model] (thinking off or at its minimum; a model not in REASONING is refused). The cache is a JSONL of
{"key", "model", "sha", "text", "finish"} keyed by the sha of the full request body, so a rerun asks only what is
missing. Jobs run in a thread pool in the order given (the caller interleaves models and arms).
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import self_concept.common.ledger  # noqa: F401  (repoints the ledger before any spend)
from persona_affect.common import judges as J
from persona_affect.common import orbackend as O
from persona_affect.common.util import Booker, seed

REASONING = {   # the "reasoning" field per model (None: none sent)
    "qwen/qwen3-14b": {"enabled": False},
    "qwen/qwen3-32b": {"enabled": False},
    "qwen/qwen3-235b-a22b-2507": None,
    "meta-llama/llama-3.1-8b-instruct": None,
    "meta-llama/llama-3.1-70b-instruct": None,
    "meta-llama/llama-4-maverick": None,
    "openai/gpt-5.6-sol": {"effort": "minimal"},
    "anthropic/claude-haiku-5.5": {"effort": "none"},
    "anthropic/claude-sonnet-5.5": {"effort": "none"},
    "google/gemini-3.8-flash": {"effort": "minimal"},
}


def body_of(job):
    b = {"model": job["model"], "messages": job["messages"], "max_tokens": job["max_tokens"],
         "temperature": job["temperature"], "seed": seed(job["key"]) % 2**31, "usage": {"include": True}}
    if REASONING[job["model"]] is not None:
        b["reasoning"] = REASONING[job["model"]]
    return b


class Client:
    def __init__(self, mini: str, cap: float, cache: Path, workers: int = 24, retry_sleep: float = 3.0, post=None):
        self.mini, self.cap, self.workers, self.retry_sleep = mini, cap, workers, retry_sleep
        self.cache_path = Path(cache)
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    r = json.loads(line)
                    if r.get("text", "").strip():
                        self.cache[r["sha"]] = (r["text"], r["finish"])
                except (ValueError, KeyError, TypeError):
                    continue
        self.post, self.headers = post, {}
        self.lock = threading.Lock()
        self.spent, self.books, self.errors = 0.0, {}, {}
        self.stopped = False

    def _book(self, model, cost):
        with self.lock:
            self.spent += cost
            b = self.books.setdefault(model, Booker(self.mini, "api", f"m10:{model}"))
        b.add(cost)

    def _note(self, model, kind, detail):
        with self.lock:
            first = (model, kind) not in self.errors
            self.errors[(model, kind)] = self.errors.get((model, kind), 0) + 1
        if first:
            print(f"[m10] {model}: {kind}: {detail[:300]}", file=sys.stderr, flush=True)

    def one(self, job):
        if job["model"] not in REASONING:
            raise ValueError(f"no REASONING setting for {job['model']}")
        body = body_of(job)
        sha = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if sha in self.cache:
            t, f = self.cache[sha]
            return {**job, "text": t, "finish": f, "error": None}
        what = detail = ""
        for k in range(O.TRIES):
            with self.lock:
                if not self.spent < self.cap:
                    self.stopped = True
                    return {**job, "text": "", "finish": "", "error": "cost_stop"}
            try:
                resp = self.post(O.URL, body, dict(self.headers))
            except Exception as e:
                what, retry, detail = O._classify(e)
            else:
                cost = O._cost(resp)
                if cost:
                    self._book(job["model"], cost)
                try:
                    text, finish = J._text(resp)
                except J._Unusable as e:
                    what, retry, detail = e.what, True, e.detail
                else:
                    if not text.strip():
                        self._note(job["model"], "empty", f"finish {finish}")
                        return {**job, "text": "", "finish": finish, "error": "empty"}
                    line = json.dumps({"key": job["key"], "model": job["model"], "sha": sha, "text": text,
                                       "finish": finish}) + "\n"
                    with self.lock:
                        with open(self.cache_path, "a", encoding="utf-8") as fh:
                            fh.write(line)
                        self.cache[sha] = (text, finish)
                    return {**job, "text": text, "finish": finish, "error": None}
            if not retry or k == O.TRIES - 1:
                break
            time.sleep(self.retry_sleep * 2 ** k)
        self._note(job["model"], f"error:{what}", detail)
        return {**job, "text": "", "finish": "", "error": f"error:{what}"}

    def run(self, jobs, log_every=500):
        if self.post is None and any(body_of(j) and hashlib.sha256(json.dumps(body_of(j), sort_keys=True).encode())
                                     .hexdigest() not in self.cache for j in jobs):
            import persona.operative.frontier as F    # forces IPv4 name resolution for the process
            self.post = O.default_post
            self.headers = {"Authorization": f"Bearer {F.load_key()}"}
        out = [None] * len(jobs)
        t0 = time.time()

        def do(i):
            out[i] = self.one(jobs[i])
            if (i + 1) % log_every == 0:
                print(f"[m10] {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s, ${self.spent:.3f})", flush=True)
        with ThreadPoolExecutor(self.workers) as pool:
            list(pool.map(do, range(len(jobs))))
        return out

    def close(self):
        first = None
        for b in self.books.values():
            try:
                b.flush()
            except Exception as e:
                first = first or e
        if first is not None:
            raise first
