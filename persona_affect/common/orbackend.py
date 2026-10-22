"""persona_affect shared helpers, part 2: an OpenRouter backend for the vigilante episode engine.

Frontier models (openai/gpt-5.6-sol, google/gemini-3.8-flash) run through `vigilante.src.episode.run_batch` with this
as `generate`:

    gen = make_or_backend("openai/gpt-5.6-sol", mini="P12", cap=16.0, effort="low")
    try:
        eps = run_batch(jobs, gen)
    finally:
        gen.close()          # writes the spend to persona_affect/ledger.jsonl, once

`gen(items)` returns one (text, finish_reason) per item, in order. `text` is what the engine stores for an assistant
turn (visible text, then the tool calls as XML, with '<think>reasoning</think>' in front for thinking items), written by
the same `canon` that vigilante.src.api_gen uses. `finish_reason` is the model's own ("stop", "tool_calls", "length"..)
or, with `text` "":

    "cost_stop"      the spend had reached `cap`, so no call was made
    "error:<what>"   the call failed after its retries; <what> is http_<code>, timeout, network, bad_json, no_choices,
                     provider_error, bad_response, or the class name of an unexpected exception

`run_batch` treats an empty reply as a final reply, so the caller must drop or flag turns whose finish is "cost_stop"
or starts with "error:" (each turn records its finish).

`cap` is this backend's own spend in USD, so the caller passes what is left of the mini's budget (see
`util.check_cap`). It is checked under a lock before every attempt, retries included. Calls already in flight when it
is reached still finish, so `spent` can pass `cap` by up to `workers` calls. `.errors` counts, per kind, the items that
failed; it also counts "cost_missing", a usable reply whose usage.cost was missing or invalid (charged as 0).
"""
from __future__ import annotations

import http.client
import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from persona_affect.common.util import ledger_add
from vigilante.src.api_gen import canon, to_openai

URL = "https://openrouter.ai/api/v1/chat/completions"
TRIES = 5            # per item: the first call and four retries, after sleeps of retry_sleep * 2**k seconds


def default_post(url: str, body: dict, headers: dict, timeout: float = 180) -> dict:
    """POST `body` as JSON and return the reply decoded from JSON.

    No retries (the backend retries). An HTTP error, a timeout and a reply that is not JSON raise
    (urllib.error.HTTPError, TimeoutError / URLError, ValueError); make_or_backend classifies them.
    """
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"content-type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class _Unusable(Exception):
    """A reply that arrived but holds no usable generation; `what` is its error label."""

    def __init__(self, what: str, detail: str = ""):
        super().__init__(what, detail)
        self.what, self.detail = what, detail


def _classify(e: Exception) -> tuple[str, bool, str]:
    """(error label, worth retrying?, detail for the log) for an exception raised by post()."""
    if isinstance(e, urllib.error.HTTPError):          # first: HTTPError is also a URLError and an OSError
        try:
            text = e.read().decode("utf-8", "replace")
        except Exception:
            text = ""
        return f"http_{e.code}", e.code == 429 or 500 <= e.code < 600, f"{e.code} {text}"[:300]
    if isinstance(e, TimeoutError) or (isinstance(e, urllib.error.URLError) and isinstance(e.reason, TimeoutError)):
        return "timeout", True, repr(e)
    if isinstance(e, (OSError, http.client.HTTPException)):   # URLError, ConnectionError, ssl.SSLError, IncompleteRead
        return "network", True, repr(e)
    if isinstance(e, ValueError):                      # json.loads on a reply that is not JSON
        return "bad_json", True, repr(e)
    return type(e).__name__, False, repr(e)            # unknown: do not hammer the API with it


def _cost(resp) -> float | None:
    """resp["usage"]["cost"] as a finite float >= 0, or None if it is missing or is not such a number.

    OpenRouter sends a JSON number. A numeric string is accepted as well, so that a change of format cannot silently
    turn spend into zero. Booleans, NaN, infinities, negatives and anything else are None.
    """
    usage = resp.get("usage") if isinstance(resp, dict) else None
    cost = usage.get("cost") if isinstance(usage, dict) else None
    if isinstance(cost, bool) or not isinstance(cost, (int, float, str)):
        return None
    try:
        cost = float(cost)
    except (ValueError, OverflowError):
        return None
    return cost if math.isfinite(cost) and cost >= 0 else None


def _args(raw) -> dict:
    """A tool call's arguments as a dict. Like api_gen: nothing or "" is {}, and text that does not parse is kept whole
    as {"_raw": text}. Unlike api_gen, valid JSON that is not an object ("[1, 2]") is kept as its original text too,
    not as the parsed value, so what the model sent is stored exactly."""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"_raw": raw}
    return {"_raw": raw}


def _reply(resp, thinking: bool) -> tuple[str, str]:
    """(text, finish_reason) of a decoded reply, in the form the engine stores. Raises _Unusable if there is none."""
    if not isinstance(resp, dict):
        raise _Unusable("bad_response", f"not a JSON object: {str(resp)[:200]}")
    if not resp.get("choices"):                # OpenRouter reports an upstream failure as 200 + {"error": ...}
        raise _Unusable("no_choices", str(resp.get("error") or resp)[:300])
    try:
        choice = resp["choices"][0]
        finish = choice.get("finish_reason")
        if finish == "error":                  # a generation that died part-way: the partial text is not a reply
            raise _Unusable("provider_error", str(choice.get("error") or choice)[:300])
        msg = choice["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc["function"]
            if not isinstance(fn["name"], str) or not fn["name"]:
                raise TypeError("tool call without a name")
            calls.append((fn["name"], _args(fn.get("arguments"))))
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        return canon(msg.get("content") or "", calls, reasoning, thinking), str(finish or "unknown")
    except (KeyError, TypeError, AttributeError, IndexError) as e:
        raise _Unusable("bad_response", repr(e)) from e


class ORBackend:
    """What make_or_backend returns: `gen(items) -> [(text, finish_reason)]`, plus .spent / .calls / .errors / .close().

    .spent    USD charged so far (sum of usage.cost of every reply that came back)
    .calls    replies that held a usable generation
    .errors   Counter: per kind, items that failed; "cost_missing" counts usable replies with no valid usage.cost
    """

    def __init__(self, model, *, mini, cap, thinking, effort, temperature, max_tokens, workers, post, headers,
                 retry_sleep):
        self.model, self.mini, self.cap = model, mini, cap
        self.thinking, self.effort, self.temperature, self.max_tokens = thinking, effort, temperature, max_tokens
        self.workers, self.retry_sleep = workers, retry_sleep
        self._post, self._headers = post, headers
        self.spent, self.calls, self.errors = 0.0, 0, Counter()
        self._lock = threading.Lock()          # spent, calls, errors, _logged
        self._close_lock = threading.Lock()
        self._closed = False
        self._logged = set()

    def __call__(self, items: list[dict]) -> list[tuple[str, str]]:
        if self._closed:
            raise RuntimeError("generate called after close(): spend after close() would never reach the ledger")
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self._one, items))

    def close(self) -> None:
        """Write `.spent` to the ledger (kind "api", note = the model). Only the first successful call writes."""
        with self._close_lock:
            if self._closed:
                return
            with self._lock:
                spent = self.spent
            ledger_add(self.mini, "api", spent, self.model)
            self._closed = True                # after the write: a failed write can be retried

    # ------------------------------------------------------------ one item

    def _one(self, item: dict) -> tuple[str, str]:
        """Never raises: whatever goes wrong with one item is that item's error, not the batch's."""
        try:
            return self._run(item)
        except Exception as e:
            return self._fail(item, type(e).__name__, repr(e))

    def _body(self, item: dict) -> dict:
        body = {"model": self.model, "messages": to_openai(item["messages"]),
                "max_tokens": self.max_tokens or item["max_tokens"], "temperature": self.temperature,
                "seed": item["seed"], "usage": {"include": True}}
        if item.get("tools"):
            body["tools"], body["tool_choice"] = item["tools"], "auto"
        if self.effort:
            body["reasoning"] = {"effort": self.effort}
        return body

    def _run(self, item: dict) -> tuple[str, str]:
        body = self._body(item)                # a malformed item raises here, before any call is made
        thinking = item.get("thinking", self.thinking)
        what = detail = ""
        for k in range(TRIES):
            with self._lock:
                if not self.spent < self.cap:  # written this way, a NaN cap would stop too
                    return "", "cost_stop"
            try:
                resp = self._post(URL, body, dict(self._headers))
            except Exception as e:
                what, retry, detail = _classify(e)
            else:
                try:
                    out = _reply(resp, thinking)
                except _Unusable as e:
                    self._charge(resp, usable=False)
                    what, retry, detail = e.what, True, e.detail
                else:
                    self._charge(resp, usable=True)
                    return out
            if not retry or k == TRIES - 1:
                break
            time.sleep(self.retry_sleep * 2 ** k)
        return self._fail(item, what, detail)

    def _charge(self, resp, usable: bool) -> None:
        """Book a reply that arrived. Its cost counts whether or not it held a generation; a usable reply without a
        valid cost is counted as "cost_missing" (and charged nothing)."""
        cost = _cost(resp)
        with self._lock:
            if cost is not None:
                self.spent += cost
            elif usable:
                self.errors["cost_missing"] += 1
            if usable:
                self.calls += 1

    def _fail(self, item, what: str, detail: str) -> tuple[str, str]:
        with self._lock:
            self.errors[what] += 1
            first = what not in self._logged
            self._logged.add(what)
        if first:                              # one line per kind keeps a systematic failure readable
            key = item.get("key") if isinstance(item, dict) else None
            print(f"[or] {self.model} {key}: {what}: {detail[:300]}", file=sys.stderr, flush=True)
        return "", f"error:{what}"


def make_or_backend(model: str, *, mini: str, cap: float, thinking: bool = False, effort: str | None = None,
                    temperature: float = 0.7, max_tokens: int | None = None, workers: int = 16, post=None,
                    retry_sleep: float = 3.0) -> ORBackend:
    """A `generate(items)` for run_batch that calls `model` through OpenRouter. See the module docstring.

    mini, cap       the ledger id the spend is written under, and the USD at which calls stop
    thinking        the default for items that carry no "thinking" flag (the engine's items always do): write the
                    reply's reasoning back as <think>..</think>
    effort          sent as "reasoning": {"effort": effort} when set
    max_tokens      overrides the item's max_tokens when set
    post            post(url, body, headers) -> dict, for tests; by default `default_post` with the key from
                    persona.operative.frontier.load_key (read here, only when no post is given, so tests never
                    need the key; that module forces IPv4 name resolution for the process)
    retry_sleep     seconds before the first retry; doubles each time (3, 6, 12, 24)
    """
    if not isinstance(mini, str) or not mini.strip():
        raise ValueError(f"make_or_backend: mini must be a non-empty id, got {mini!r}")
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap < 0:
        raise ValueError(f"make_or_backend: cap must be a finite number of USD >= 0, got {cap!r}")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"make_or_backend: workers must be an int >= 1, got {workers!r}")
    if isinstance(retry_sleep, bool) or not isinstance(retry_sleep, (int, float)) or not 0 <= retry_sleep < math.inf:
        raise ValueError(f"make_or_backend: retry_sleep must be a finite number of seconds >= 0, got {retry_sleep!r}")
    if post is None:
        from persona.operative.frontier import load_key
        post, headers = default_post, {"authorization": f"Bearer {load_key()}"}
    else:
        headers = {}
    return ORBackend(model, mini=mini, cap=cap, thinking=thinking, effort=effort, temperature=temperature,
                     max_tokens=max_tokens, workers=workers, post=post, headers=headers, retry_sleep=retry_sleep)
