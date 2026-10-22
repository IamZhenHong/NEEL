"""persona_affect shared helpers, part 3: two-judge labelling.

Minis label model outputs (frustration 0-10, admits/denies, ...) with two LLM judges from different families and report
how well the judges agree, because the owner never hand-labels (DESIGN.md: validate judges by cross-family agreement):

    labels = label(items, render, {"yes", "no"}, mini="A14", cap=2.0, cache=out / "judge_cache.jsonl")
    agreement(labels)    # {"kappa": 0.74, "n": 188, "none": {"anthropic/claude-haiku-5.5": 3, "openai/gpt-5-mini": 9}}

`label()` returns {item key: {judge: label or None}}: item-major, one row per item, in the order of `items`, one entry
per judge. It is a `Labels`, a plain dict that also carries the run's counters (.spent .calls .cached .none
.cost_missing). `agreement()` reads the same shape and refuses the transposed one, {judge: {item key: label}}.

    items     dicts with a unique str "key"
    render    render(item) -> the judge prompt. label() appends "\\n\\nAnswer with exactly one of: " +
              ", ".join(sorted(allowed)). `render` is called once per item, before any call is made: if it raises,
              label() raises and nothing was spent
    allowed   the labels a judge may give; each is one word (a reply is split on whitespace) and two of them may not
              be the same word once case and the punctuation at the ends are ignored

The label is the first word of the reply that is an allowed label: the reply is split on whitespace, the punctuation at
the ends of each word is dropped, and the rest is compared case-insensitively. "Yes.", "**yes**" and "Answer: yes" are
"yes"; "7.5", "7/10" and "maybe" are none of 0..10 or yes/no, so they are None, never a guess from a part of a number.
The label returned is the allowed entry as the caller wrote it. A None label is one of

    off_set     the reply held no allowed label (the reply is still cached)
    empty       the reply had no text (not retried: at temperature 0 it would come back empty again)
    cost_stop   the spend had reached `cap`, so no call was made
    error:<w>   the call failed after its retries; <w> is http_<code>, timeout, network, bad_json, no_choices,
                provider_error, bad_response or the class name of an unexpected exception

and is counted per judge and kind in `.none` of the result and, per judge, by `agreement()`. A None never stops the
batch and never counts in kappa. The first problem of each kind and judge is also printed to stderr, and so is the
moment the cap is reached.

Each request is {"model", "messages": [{"role": "user", "content": prompt}], "max_tokens": 400, "temperature": 0,
"usage": {"include": True}, "reasoning": {"effort": EFFORT[model]}}, posted to the OpenRouter URL of orbackend, which
also supplies the transport (`default_post`), the retries (5 tries, sleeps retry_sleep * 2**k = 3, 6, 12, 24 s; 429,
5xx, timeouts, network errors, replies that are not JSON or hold no choices) and the reading of usage.cost (missing,
None, non-numeric, non-finite or negative: charged as 0 and counted in `.cost_missing`). A judge that is not in EFFORT
is refused up front: add it (`EFFORT["google/x"] = "minimal"`, or None to send no "reasoning").

Cache: `cache` is a JSONL file with one {"model", "sha", "reply"} line per answer, sha = sha256 of the full prompt (the
answer instruction included, so a changed allowed set is a new question). It is read once at the start and appended to
as each answer arrives, so what was paid for survives a crash. Nothing but an answer is cached (no error, no empty
reply, no cost_stop), so a rerun fills in exactly what is missing. Identical prompts within a batch are asked once.
A cache hit costs nothing and needs neither the key nor a cap. A line that cannot be read is skipped (and counted on
stderr). Several runs may append to one cache file; each record is a single write.

Spend: `cap` is the USD this one label() call may spend (pass what is left of the mini's budget, see util.check_cap);
it is checked under a lock before every attempt, retries included. Jobs run item by item, both judges of an item
together, so when the cap binds the first items have both labels and the rest have none. Calls already in flight when
it is reached still finish, so the spend can pass `cap` by up to `workers - 1` calls. At the end (also when the batch
is aborted, e.g. by Ctrl-C) the spend is written to util's ledger, once per judge that cost something: kind "api",
note "judges:<judge model>". Nothing is written for a judge that cost nothing.

The key (OPENROUTER_API_KEY) is read by persona.operative.frontier.load_key only when `post` is not given and at least
one call has to be made; importing that module also forces IPv4 name resolution for the process, which the real calls
rely on. Importing this module does neither.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from persona_affect.common import orbackend
from persona_affect.common.util import ledger_add
from vigilante.src.judge import kappa

JUDGES = ("anthropic/claude-haiku-5.5", "openai/gpt-5-mini")
EFFORT = {"anthropic/claude-haiku-5.5": "none",     # Haiku 5.5 reasons by default, and a small max_tokens then gives
          "openai/gpt-5-mini": "minimal"}           # empty answers; OpenRouter rejects "none" for gpt-5-mini (HTTP 400)
MAX_TOKENS = 400
ANSWER = "\n\nAnswer with exactly one of: "

_EDGE = re.compile(r"^[\W_]+|[\W_]+$")      # the punctuation, symbols and markdown at the ends of a word


class Labels(dict):
    """What label() returns: a plain {item key: {judge: label or None}} that also carries how the run went.

    .spent          USD charged by the calls this label() made (answers read from the cache cost nothing)
    .calls          replies that came back from the API, usable or not (a retry that gets a reply counts again)
    .cached         distinct (judge, prompt) pairs answered from the cache
    .none           {judge: Counter}: why labels are None, one count per item and kind (see the module docstring)
    .cost_missing   replies whose usage.cost was missing or invalid (charged as 0)
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.spent, self.calls, self.cached, self.cost_missing, self.none = 0.0, 0, 0, 0, {}


# ---------------------------------------------------------------- reading a reply

def _word(token: str) -> str:
    return _EDGE.sub("", token).casefold()


def _lookup(allowed) -> dict[str, str]:
    """{word: allowed label as written}. ValueError if the set is empty, not made of strings, or has a label that is
    empty, has more than one word, or is the same word as another once case and end punctuation are ignored."""
    if not isinstance(allowed, (set, frozenset, list, tuple)) or not allowed:
        raise ValueError(f"label: allowed must be a non-empty set of labels, got {allowed!r}")
    out: dict[str, str] = {}
    for a in allowed:
        if not isinstance(a, str):
            raise ValueError(f"label: allowed labels must be strings, got {a!r}")
        w = _word(a)
        if not w or any(c.isspace() for c in w):
            raise ValueError(f"label: allowed label {a!r} must be one word (a reply is split on whitespace)")
        if w in out:
            raise ValueError(f"label: allowed labels {out[w]!r} and {a!r} are the same word once case and the "
                             f"punctuation at its ends are ignored")
        out[w] = a
    return out


def _parse(reply: str, lookup: dict[str, str]) -> str | None:
    """The first word of `reply` that is an allowed label (see the module docstring), else None."""
    for token in reply.split():
        hit = lookup.get(_word(token))
        if hit is not None:
            return hit
    return None


class _Unusable(Exception):
    """A reply that arrived but holds no usable answer; `what` is its error label."""

    def __init__(self, what: str, detail: str = ""):
        super().__init__(what, detail)
        self.what, self.detail = what, detail


def _text(resp) -> str:
    """The reply text of a decoded response ("" if the model sent none). Raises _Unusable for a response without one."""
    if not isinstance(resp, dict):
        raise _Unusable("bad_response", f"not a JSON object: {str(resp)[:200]}")
    if not resp.get("choices"):                # OpenRouter reports an upstream failure as 200 + {"error": ...}
        raise _Unusable("no_choices", str(resp.get("error") or resp)[:300])
    try:
        choice = resp["choices"][0]
        if choice.get("finish_reason") == "error":     # a generation that died part-way: the partial text is no answer
            raise _Unusable("provider_error", str(choice.get("error") or choice)[:300])
        content = choice["message"].get("content")
    except (KeyError, TypeError, AttributeError, IndexError) as e:
        raise _Unusable("bad_response", repr(e)) from e
    if content is None:
        return ""
    if not isinstance(content, str):
        raise _Unusable("bad_response", f"content is not text: {str(content)[:200]}")
    return content


def _body(judge: str, prompt: str) -> dict:
    body = {"model": judge, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS,
            "temperature": 0, "usage": {"include": True}}
    if EFFORT[judge]:
        body["reasoning"] = {"effort": EFFORT[judge]}
    return body


# ---------------------------------------------------------------- the cache

class _Cache:
    """The JSONL file of answers: read once, then appended to as answers arrive."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.replies: dict[tuple[str, str], str] = {}
        skipped = 0
        if self.path.exists():                 # errors="replace": a line cut mid-character is then just unreadable
            for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    model, sha, reply = row["model"], row["sha"], row["reply"]
                except (ValueError, KeyError, TypeError):
                    skipped += 1
                    continue
                if isinstance(model, str) and isinstance(sha, str) and isinstance(reply, str) and reply.strip():
                    self.replies[(model, sha)] = reply
                else:
                    skipped += 1
        if skipped:                            # the prompts on those lines are simply asked again
            print(f"[judges] cache {self.path}: skipped {skipped} unreadable line(s)", file=sys.stderr, flush=True)

    def add(self, model: str, sha: str, reply: str) -> None:
        line = json.dumps({"model": model, "sha": sha, "reply": reply}) + "\n"
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a+b") as f:
                end = f.seek(0, os.SEEK_END)
                if end:
                    f.seek(end - 1)
                    if f.read(1) != b"\n":     # an earlier run died mid-line: do not fuse this record onto it
                        line = "\n" + line
                f.write(line.encode("utf-8"))


# ---------------------------------------------------------------- the calls

class _Run:
    """The calls of one label(): shared spend, the stop at the cap, retries, counts. Thread-safe."""

    def __init__(self, judges, cap: float, retry_sleep: float):
        self.cap, self.retry_sleep = cap, retry_sleep
        self.post, self.headers = None, {}
        self.spent = {j: 0.0 for j in judges}      # USD per judge
        self.calls = self.cost_missing = 0
        self.lock = threading.Lock()               # spent, calls, cost_missing, logged, stopped
        self.logged: set[tuple[str, str]] = set()
        self.stopped = False

    def total(self) -> float:
        return math.fsum(self.spent.values())

    def ask(self, judge: str, prompt: str, key: str) -> tuple[str | None, str]:
        """(reply text, "") or (None, why). Never raises an Exception: whatever goes wrong is that item's None."""
        try:
            return self._ask(judge, prompt, key)
        except Exception as e:
            return self.problem(judge, key, f"error:{type(e).__name__}", repr(e))

    def _ask(self, judge, prompt, key):
        body = _body(judge, prompt)
        what = detail = ""
        for k in range(orbackend.TRIES):
            with self.lock:
                if not self.total() < self.cap:    # written this way, a NaN cap would stop too
                    if not self.stopped:
                        self.stopped = True
                        print(f"[judges] spend ${self.total():.4f} reached the cap ${self.cap:.4f}: no more calls, "
                              f"the remaining labels are None", file=sys.stderr, flush=True)
                    return None, "cost_stop"
            try:
                resp = self.post(orbackend.URL, body, dict(self.headers))
            except Exception as e:
                what, retry, detail = orbackend._classify(e)
            else:
                try:
                    text = _text(resp)
                except _Unusable as e:
                    self._charge(judge, resp, usable=False)
                    what, retry, detail = e.what, True, e.detail
                else:
                    self._charge(judge, resp, usable=True)
                    if not text.strip():
                        return self.problem(judge, key, "empty", "the reply has no text")
                    return text, ""
            if not retry or k == orbackend.TRIES - 1:
                break
            time.sleep(self.retry_sleep * 2 ** k)
        return self.problem(judge, key, f"error:{what}", detail)

    def _charge(self, judge: str, resp, usable: bool) -> None:
        """Book a reply that arrived. Its cost counts whether or not it held an answer; a usable reply without a
        valid cost is counted in cost_missing (and charged nothing)."""
        cost = orbackend._cost(resp)
        with self.lock:
            self.calls += 1
            if cost is not None:
                self.spent[judge] += cost
            elif usable:
                self.cost_missing += 1

    def problem(self, judge: str, key: str, kind: str, detail: str) -> tuple[None, str]:
        """Print the first problem of each kind and judge (one line each keeps a systematic failure readable)."""
        with self.lock:
            first = (judge, kind) not in self.logged
            self.logged.add((judge, kind))
        if first:
            print(f"[judges] {judge} {key}: {kind}: {detail[:300]}", file=sys.stderr, flush=True)
        return None, kind


# ---------------------------------------------------------------- label

def label(items: list[dict], render, allowed: set[str], *, mini: str, cap: float, cache: Path, post=None,
          judges=JUDGES, workers: int = 16, retry_sleep: float = 3.0) -> Labels:
    """Label every item with every judge. See the module docstring for the prompt, the parsing, the cache, the cap and
    the ledger; returns a Labels, {item key: {judge: label or None}} with the run's counters as attributes.

    mini, cap       the ledger id the spend is written under, and the USD at which this call stops asking
    cache           the JSONL cache file (created on the first answer)
    post            post(url, body, headers) -> dict, for tests; by default `orbackend.default_post` with the key from
                    persona.operative.frontier.load_key (read only if a call has to be made)
    judges          the models to ask, each a key of EFFORT
    workers         calls in flight at once
    retry_sleep     seconds before the first retry; doubles each time (3, 6, 12, 24)
    """
    if not isinstance(mini, str) or not mini.strip():
        raise ValueError(f"label: mini must be a non-empty id, got {mini!r}")
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap < 0:
        raise ValueError(f"label: cap must be a finite number of USD >= 0, got {cap!r}")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"label: workers must be an int >= 1, got {workers!r}")
    if isinstance(retry_sleep, bool) or not isinstance(retry_sleep, (int, float)) or not 0 <= retry_sleep < math.inf:
        raise ValueError(f"label: retry_sleep must be a finite number of seconds >= 0, got {retry_sleep!r}")
    if (not isinstance(judges, (tuple, list)) or not judges or not all(isinstance(j, str) for j in judges)
            or len(set(judges)) != len(judges)):
        raise ValueError(f"label: judges must be a non-empty tuple of different model ids, got {judges!r}")
    unknown = [j for j in judges if j not in EFFORT]
    if unknown:
        raise ValueError(f"label: no reasoning setting for {unknown}: add each to judges.EFFORT (the effort to send, "
                         f"or None to send no 'reasoning' field)")
    lookup = _lookup(allowed)

    items = list(items)
    keys = []
    for i, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("key"), str):
            raise ValueError(f"label: items[{i}] must be a dict with a str 'key', got {item!r:.100}")
        keys.append(item["key"])
    if len(set(keys)) != len(keys):
        raise ValueError(f"label: item keys must be unique, {next(k for k, n in Counter(keys).items() if n > 1)!r} "
                         f"appears more than once")
    prompts = []
    for item in items:                          # every prompt first: a bug in `render` costs nothing
        text = render(item)
        if not isinstance(text, str):
            raise TypeError(f"label: render({item['key']!r}) must return the prompt as a str, "
                            f"got {type(text).__name__}")
        prompts.append(text + ANSWER + ", ".join(sorted(allowed)))
    shas = [hashlib.sha256(p.encode("utf-8")).hexdigest() for p in prompts]

    store = _Cache(cache)
    wanted = {(j, s) for s in shas for j in judges}
    replies = {p: store.replies[p] for p in wanted if p in store.replies}
    cached, failed = len(replies), {}
    todo, queued = [], set(replies)             # (judge, sha, prompt, key of the first item with it), item by item
    for key, prompt, sha in zip(keys, prompts, shas):
        for judge in judges:
            if (judge, sha) not in queued:
                queued.add((judge, sha))
                todo.append((judge, sha, prompt, key))

    run = _Run(judges, cap, retry_sleep)
    if todo:
        if post is None:
            from persona.operative.frontier import load_key
            run.post, run.headers = orbackend.default_post, {"authorization": f"Bearer {load_key()}"}
        else:
            run.post = post

        def one(job):
            judge, sha, prompt, key = job
            text, why = run.ask(judge, prompt, key)
            if text is not None:
                store.add(judge, sha, text)     # at once, not at the end: it is paid for even if the batch is aborted
            return (judge, sha), text, why
        try:
            with ThreadPoolExecutor(workers) as ex:
                for pair, text, why in ex.map(one, todo):
                    if text is None:
                        failed[pair] = why
                    else:
                        replies[pair] = text
        finally:                                # also when aborted: whatever came back was billed
            for judge in judges:
                if run.spent[judge] > 0:
                    ledger_add(mini, "api", run.spent[judge], f"judges:{judge}")

    out = Labels()
    out.none = {j: Counter() for j in judges}
    for key, sha in zip(keys, shas):
        row = {}
        for judge in judges:
            if (judge, sha) in replies:
                row[judge] = _parse(replies[(judge, sha)], lookup)
                if row[judge] is None:
                    out.none[judge]["off_set"] += 1
                    run.problem(judge, key, "off_set", repr(replies[(judge, sha)][:200]))
            else:
                row[judge] = None
                out.none[judge][failed[(judge, sha)]] += 1
        out[key] = row
    out.spent, out.calls, out.cached, out.cost_missing = run.total(), run.calls, cached, run.cost_missing
    return out


# ---------------------------------------------------------------- agreement

def agreement(labels: dict[str, dict[str, str | None]], judges=JUDGES) -> dict:
    """Cohen's kappa (vigilante.src.judge.kappa) between two judges, over the items both labelled.

    labels      {item key: {judge: label or None}}, what label() returns (a row without the judge counts as None)
    judges      exactly two different judges

    Returns {"kappa", "n", "none"}: kappa over the n items where both labels are not None (None, never NaN, when n is
    0), and per judge the number of items where its label is None. Labels may be any strings, so to compare coarser
    classes (a 0-10 score in bins) map them first. Note that vigilante's kappa is 1.0 when both judges give one and the
    same label to every item, so look at the class balance before trusting a perfect score.
    """
    if not isinstance(judges, (tuple, list)) or len(judges) != 2 or judges[0] == judges[1]:
        raise ValueError(f"agreement: judges must be two different judges, got {judges!r}")
    x, y = judges
    if x in labels or y in labels:
        raise ValueError(f"agreement: labels must be {{item key: {{judge: label}}}}, but {x!r} or {y!r} is a key of it "
                         f"(judge-major?)")
    a, b, none = [], [], {x: 0, y: 0}
    for row in labels.values():
        la, lb = row.get(x), row.get(y)
        if la is None:
            none[x] += 1
        if lb is None:
            none[y] += 1
        if la is not None and lb is not None:
            a.append(la)
            b.append(lb)
    return {"kappa": kappa(a, b) if a else None, "n": len(a), "none": none}
