"""persona_affect shared helpers, part 2: an OpenRouter backend for the vigilante episode engine.

Frontier models (openai/gpt-5.6-sol, google/gemini-3.8-flash) run through `vigilante.src.episode.run_batch` with this
as `generate`:

    def main(argv=None):
        util.install_term_handlers()     # a TERM or HUP then runs the finally below instead of ending the process
        gen = make_or_backend("openai/gpt-5.6-sol", mini="P12", cap=16.0, effort="low")
        try:
            eps = run_batch(jobs, gen, on_round=end_failed)
        finally:
            gen.close()                  # books what is not on persona_affect/ledger.jsonl yet
        ...                              # then drop or flag every episode whose end is "cost_stop" or "error:..."

`gen(items)` returns one (text, finish_reason) per item, in order. `text` is what the engine stores for an assistant
turn (visible text, then the tool calls as XML, with '<think>reasoning</think>' in front for thinking items), written by
the same `canon` that vigilante.src.api_gen uses. `finish_reason` is the model's own ("stop", "tool_calls", "length",
"content_filter"..) or, with `text` "":

    "cost_stop"      the spend had reached `cap`, so no call was made
    "error:<what>"   the call failed after its retries; <what> is http_<code>, timeout, network, bad_json, no_choices,
                     provider_error, bad_response, or the class name of an unexpected exception

A failed generation ("cost_stop" or "error:<what>") is not a reply, but `run_batch` takes its "" for the model's final
reply: it stores an empty assistant turn and goes on with the next scripted user turn (or the handover request), so the
rest of the episode is played on a corrupted history, and a thinking turn ends the episode as "think_unclosed". So the
caller must drop or flag the whole episode once one of its turns failed, never just that turn: with
`run_batch(jobs, gen, on_round=end_failed)` such an episode stops at its failed turn and its end is that finish (see
`end_failed`); the runner then drops or flags every episode whose end is "cost_stop" or starts with "error:".

Frontier jobs carry fmt "qwen" (or none, which means qwen), whatever the model: `canon` writes every reply in Qwen's XML
tool-call format, and the episode parser goes by the job's fmt.

`cap` is this backend's own spend in USD, so the caller passes what is left of the mini's budget (see
`util.check_cap`). It is checked under a lock before every attempt, retries included. Calls already in flight when it
is reached still finish, so `spent` can pass `cap` by up to `workers` calls. The spend goes to persona_affect/ledger.jsonl
(kind "api", note = the model id) as it accrues, through a `util.Booker`: a line whenever $0.25 of it is not booked yet,
and the rest when each gen(items) call returns, so every round of the engine ends with its spend on disk. close() books
whatever a failed write left behind (and writes one $0 line for a backend that never booked anything, so the run is on
record). A kill therefore loses at most $0.25 plus the calls in flight; readers sum the lines of a note. `.errors`
counts, per kind, the items that failed; it also counts "cost_missing", a usable reply whose usage.cost was missing or
invalid (charged as 0). `.finishes` counts, per finish_reason, the items that came back with it, the failed ones too.

Runner rules (every mini's frontier runner): util.install_term_handlers() first thing in main(); gen.close() in a
finally; run_batch(..., on_round=end_failed) (or call end_failed from your own on_round), then drop or flag the whole
episode whose end is "cost_stop" or "error:<what>"; jobs with fmt "qwen" or none.
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

from persona_affect.common import util
from vigilante.src.api_gen import canon, to_openai
from vigilante.src.episode import HANDOVER

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
    """What make_or_backend returns: `gen(items) -> [(text, finish_reason)]`, plus .spent / .calls / .errors /
    .finishes / .close().

    .spent      USD charged so far (sum of usage.cost of every reply that came back)
    .calls      replies that held a usable generation
    .errors     Counter: per kind, items that failed; "cost_missing" counts usable replies with no valid usage.cost
    .finishes   Counter: per finish_reason, the items that came back with it: the model's own ("stop", "tool_calls",
                "length" when max_tokens ran out, "content_filter", ...), "cost_stop" and "error:<what>"
    """

    def __init__(self, model, *, mini, cap, thinking, effort, temperature, max_tokens, workers, post, headers,
                 retry_sleep):
        self.model, self.mini, self.cap = model, mini, cap
        self.thinking, self.effort, self.temperature, self.max_tokens = thinking, effort, temperature, max_tokens
        self.workers, self.retry_sleep = workers, retry_sleep
        self._post, self._headers = post, headers
        self.spent, self.calls, self.errors, self.finishes = 0.0, 0, Counter(), Counter()
        self._lock = threading.Lock()          # spent, calls, errors, finishes, _logged
        self._close_lock = threading.Lock()
        self._closed = False
        self._logged = set()
        self._book = util.Booker(mini, "api", model)   # the spend goes to the ledger as it accrues

    def __call__(self, items: list[dict]) -> list[tuple[str, str]]:
        if self._closed:
            raise RuntimeError("generate called after close(): spend after close() would never reach the ledger")
        try:
            with ThreadPoolExecutor(self.workers) as ex:
                return list(ex.map(self._one, items))
        finally:                               # the engine's round ends with its spend on disk (also when aborted)
            try:
                self._book.flush()
            except Exception:                  # kept unbooked (Booker printed it): the next flush, close() at the
                pass                           # latest, tries again, and close() raises if it still cannot

    def close(self) -> None:
        """Book what is not on the ledger yet (kind "api", note = the model) and refuse later calls. A backend that
        never booked anything writes one $0 line, so the run is on record. If the write fails close() raises and can
        be called again; only the first successful call books."""
        with self._close_lock:
            if self._closed:
                return
            self._book.flush()
            if not self._book.lines:
                util.ledger_add(self.mini, "api", 0.0, self.model)
            self._closed = True                # after the writes: a failed write can be retried

    # ------------------------------------------------------------ one item

    def _one(self, item: dict) -> tuple[str, str]:
        """Never raises: whatever goes wrong with one item is that item's error, not the batch's."""
        try:
            out = self._run(item)
        except Exception as e:
            out = self._fail(item, type(e).__name__, repr(e))
        with self._lock:
            self.finishes[out[1]] += 1
        return out

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
        """Charge a reply that arrived. Its cost counts whether or not it held a generation; a usable reply without a
        valid cost is counted as "cost_missing" (and charged nothing). The cost goes to the Booker outside the lock: a
        ledger write must not hold up the cap checks of the other workers."""
        cost = _cost(resp)
        with self._lock:
            if cost is not None:
                self.spent += cost
            elif usable:
                self.errors["cost_missing"] += 1
            if usable:
                self.calls += 1
        if cost:
            self._book.add(cost)

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


# ---------------------------------------------------------------- episodes with a failed generation

def failed(finish) -> bool:
    """True for the finish of a generation that failed: "cost_stop" or "error:<what>" (the backend returned "")."""
    return isinstance(finish, str) and (finish == "cost_stop" or finish.startswith("error:"))


def end_failed(eps: list[dict], rnd: int | None = None) -> int:
    """End every episode that has a failed generation, at that generation. Returns how many it ended on this call.

    Use it as run_batch's on_round, so that an episode stops as soon as one of its calls failed:

        eps = run_batch(jobs, gen, on_round=end_failed)
        eps = run_batch(jobs, gen, on_round=lambda eps, rnd: (end_failed(eps), checkpoint(eps, rnd)))   # with your own

    A failed generation is a turn whose finish is "cost_stop" or "error:<what>". The engine took its "" for the model's
    final reply: it stored an empty assistant message and went on, appending the next scripted user turn or the
    handover request, or it ended a thinking turn as "think_unclosed". end_failed takes that back and ends the episode
    there: done, `end` = that finish, the empty assistant message and whatever the engine queued after it removed, so the
    messages end with what the model was asked and never answered. The failed turn stays in "turns", where its finish
    records the failure. Case jobs, chat jobs (case None), the handover phase and thinking jobs are all handled. Each
    episode is dealt with once (ep["failed"] holds the finish), so calling it every round, or again afterwards, is safe.

    Called only after run_batch, it still ends such episodes, but one that went on past its failed turn keeps the turns
    it played on the corrupted history: its `end` names the first failure all the same, so it is dropped or flagged as a
    whole.
    """
    n = 0
    for ep in eps:
        if ep.get("failed"):
            continue
        turns = ep.get("turns") or []
        i = next((k for k, t in enumerate(turns) if failed(t.get("finish"))), None)
        if i is None:
            continue
        if i == len(turns) - 1:                # it is the newest turn: undo what the engine did with the ""
            _take_back(ep)
        ep["failed"] = ep["end"] = turns[i]["finish"]
        ep["done"] = True
        n += 1
    return n


def _take_back(ep: dict) -> None:
    """Undo what the engine did after an empty reply (see end_failed): the scripted user turn or the handover request
    it appended, then the empty assistant message itself."""
    msgs, job, turn = ep["messages"], ep["job"], ep["turns"][-1]
    if len(msgs) >= 2 and msgs[-1]["role"] == "user" and msgs[-2]["role"] == "assistant" and not msgs[-2]["content"]:
        if ep.get("phase") == "handover" and turn.get("phase") != "handover" and msgs[-1]["content"] == HANDOVER:
            msgs.pop()
            ep["phase"] = turn.get("phase", "task")
        elif job.get("case") and ep.get("fu", 0) > 0:
            import vigilante.cases as vcases   # looked up at call time, as the engine does
            if msgs[-1]["content"] == vcases.ALL[job["case"]].followups[ep["fu"] - 1]:
                msgs.pop()
                ep["fu"] -= 1
    if msgs and msgs[-1]["role"] == "assistant" and not msgs[-1]["content"]:
        msgs.pop()
