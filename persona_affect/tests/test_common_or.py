"""Tests for persona_affect.common.orbackend: the OpenRouter backend for the vigilante episode engine.

Run from the repo root: uv run pytest persona_affect/tests/test_common_or.py -q
No network and no API key: every backend gets a fake `post`; the one test that builds a backend without `post` stubs
both `load_key` and `default_post`, and the `default_post` tests replace `urlopen`. Every test points util.LEDGER at a
tmp file, so the real persona_affect/ledger.jsonl is never touched, and time.sleep is replaced by a recorder, so no
test waits on a backoff.
"""
import http.client
import io
import json
import socket
import subprocess
import sys
import threading
import types
import urllib.error
from pathlib import Path

import pytest

from persona_affect.common import orbackend, util
from vigilante.src.api_gen import to_openai
from vigilante.src.episode import run_batch
from vigilante.src.sim import parse_calls

URL = "https://openrouter.ai/api/v1/chat/completions"
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(util, "LEDGER", path)
    return path


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """Record the backoff sleeps instead of waiting for them."""
    rec = []
    monkeypatch.setattr(orbackend.time, "sleep", rec.append)
    return rec


def _resp(text="", calls=(), finish="stop", cost=0.01):
    tc = [{"id": "c0", "function": {"name": n, "arguments": a}} for n, a in calls]
    return {"choices": [{"message": {"content": text, "tool_calls": tc}, "finish_reason": finish}],
            "usage": {"cost": cost}}


def _item(i=0, **kw):
    it = {"messages": [{"role": "user", "content": f"hi {i}"}], "tools": None, "thinking": False, "seed": i,
          "max_tokens": 50, "key": f"k{i}"}
    it.update(kw)
    return it


def _http(code):
    return urllib.error.HTTPError(URL, code, "x", {}, io.BytesIO(b"err body"))


def _script(*replies):
    """A fake post that plays `replies` in order (an exception is raised, anything else returned); the last one
    repeats. post.calls records (url, body, headers) of every call."""
    def post(u, b, h):
        post.calls.append((u, b, h))
        r = replies[min(len(post.calls), len(replies)) - 1]
        if isinstance(r, BaseException):
            raise r
        return r
    post.calls = []
    return post


def _gen(post, **kw):
    kw.setdefault("cap", 1.0)
    kw.setdefault("workers", 1)
    return orbackend.make_or_backend("m", mini="T", post=post, **kw)


def _rows(path):
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


# ---------------------------------------------------------------- the plan's four tests

def test_or_tool_call_canon():
    gen = orbackend.make_or_backend("m", mini="T", cap=1.0, post=lambda u, b, h: _resp("ok", [("choose", '{"option_id": "B"}')], "tool_calls"))
    (text, fin), = gen([{"messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function", "function": {"name": "choose", "description": "", "parameters": {"type": "object", "properties": {}}}}], "thinking": False, "seed": 1, "max_tokens": 50, "key": "k"}])
    assert "<function=choose>" in text and fin == "tool_calls" and abs(gen.spent - 0.01) < 1e-9


def test_or_cost_stop():
    gen = orbackend.make_or_backend("m", mini="T", cap=0.015, workers=1, post=lambda u, b, h: _resp("x"))
    outs = gen([_item(i) for i in range(3)])
    assert [f for _, f in outs].count("cost_stop") >= 1


def test_or_malformed_args():
    gen = orbackend.make_or_backend("m", mini="T", cap=1.0, post=lambda u, b, h: _resp("", [("choose", "{oops")], "tool_calls"))
    text, _ = gen([_item(0)])[0]
    assert "_raw" in text


def test_or_error_continues():
    def boom(u, b, h): raise urllib.error.HTTPError(u, 500, "x", {}, io.BytesIO(b"err"))
    gen = orbackend.make_or_backend("m", mini="T", cap=1.0, post=boom, retry_sleep=0)
    assert gen([_item(0)])[0][1].startswith("error:")


# ---------------------------------------------------------------- what a reply becomes (the engine's stored text)

def test_text_reply_keeps_its_finish_reason():
    assert _gen(_script(_resp("  hello  ", finish="length")))([_item()]) == [("hello", "length")]


def test_tool_call_text_parses_back_with_the_engines_parser():
    gen = _gen(_script(_resp("thinking aloud", [("choose", '{"option_id": "B", "reason": "safer"}')], "tool_calls")))
    (text, fin), = gen([_item()])
    head, calls, malformed = parse_calls(text)
    assert (head, malformed, fin) == ("thinking aloud", False, "tool_calls")
    assert calls == [("choose", {"option_id": "B", "reason": "safer"})]


def test_several_tool_calls_keep_their_order():
    gen = _gen(_script(_resp("", [("first", '{"a": "1"}'), ("second", "{}"), ("third", '{"c": "3"}')], "tool_calls")))
    (text, _), = gen([_item()])
    assert [n for n, _ in parse_calls(text)[1]] == ["first", "second", "third"]


@pytest.mark.parametrize("arguments, expected", [
    ('{"a": "x", "b": 2}', {"a": "x", "b": 2}),
    ('{"nested": {"k": [1, 2]}}', {"nested": {"k": [1, 2]}}),
    ({"a": "already a dict"}, {"a": "already a dict"}),    # some providers send the object itself
    ("", {}), (None, {}), ("{}", {}), ({}, {}),
])
def test_tool_call_arguments_are_read_as_an_object(arguments, expected):
    (text, _), = _gen(_script(_resp("", [("choose", arguments)], "tool_calls")))([_item()])
    assert parse_calls(text)[1] == [("choose", expected)]


@pytest.mark.parametrize("arguments", [
    "{oops",                       # malformed JSON
    '{"a": "x"',                   # truncated JSON
    "[1, 2]",                      # valid JSON that is not an object
    '"just a string"',
    "null", "42",
    "{'single': 'quotes'}",
    "<not json at all>",
])
def test_arguments_that_are_not_a_json_object_are_stored_raw(arguments):
    gen = _gen(_script(_resp("", [("choose", arguments)], "tool_calls")))
    (text, fin), = gen([_item()])
    assert f"<parameter=_raw>\n{arguments}\n</parameter>" in text and fin == "tool_calls"
    assert gen.errors == {} and gen.calls == 1             # never a crash, and not counted as an error


def test_malformed_arguments_name_the_call_and_keep_the_other_calls():
    gen = _gen(_script(_resp("", [("first", "{oops"), ("second", '{"b": "2"}')], "tool_calls")))
    (text, _), = gen([_item()])
    assert parse_calls(text)[1] == [("first", {"_raw": "{oops"}), ("second", {"b": 2})]


def test_non_string_arguments_do_not_crash():
    (text, _), = _gen(_script(_resp("", [("choose", [1, 2])], "tool_calls")))([_item()])
    assert "<parameter=_raw>\n[1, 2]\n</parameter>" in text


def test_reasoning_is_written_back_only_for_thinking_items():
    reply = _resp("answer")
    reply["choices"][0]["message"]["reasoning"] = "my thoughts"
    gen = _gen(_script(reply))
    out = gen([_item(0, thinking=True), _item(1, thinking=False)])
    assert out[0][0] == "<think>\nmy thoughts\n</think>\n\nanswer"
    assert out[1][0] == "answer"


def test_reasoning_content_field_is_read_too():
    reply = _resp("answer")
    reply["choices"][0]["message"]["reasoning_content"] = "deep"
    assert _gen(_script(reply))([_item(thinking=True)])[0][0].startswith("<think>\ndeep\n</think>")


def test_backend_thinking_is_the_default_for_items_without_the_flag():
    item = _item()
    del item["thinking"]
    assert _gen(_script(_resp("answer")), thinking=True)([item])[0][0] == "<think>\n\n</think>\n\nanswer"
    assert _gen(_script(_resp("answer")))([item])[0][0] == "answer"
    # the item's own flag wins over the backend's
    assert _gen(_script(_resp("answer")), thinking=True)([_item(thinking=False)])[0][0] == "answer"


def test_missing_finish_reason_is_recorded_as_unknown():
    reply = _resp("x")
    del reply["choices"][0]["finish_reason"]
    assert _gen(_script(reply))([_item()]) == [("x", "unknown")]
    reply["choices"][0]["finish_reason"] = None
    assert _gen(_script(reply))([_item()]) == [("x", "unknown")]


def test_empty_batch():
    assert _gen(_script(_resp()))([]) == []


# ---------------------------------------------------------------- the request

def test_request_body():
    msgs = [{"role": "system", "content": "be good"}, {"role": "user", "content": "hi"}]
    post = _script(_resp("x"))
    gen = _gen(post, temperature=0.3)
    gen([_item(7, messages=msgs, max_tokens=123)])
    (url, body, headers), = post.calls
    assert url == URL and isinstance(headers, dict)
    assert body == {"model": "m", "messages": to_openai(msgs), "max_tokens": 123, "temperature": 0.3, "seed": 7,
                    "usage": {"include": True}}


def test_default_temperature_is_0_7():
    post = _script(_resp("x"))
    _gen(post)([_item()])
    assert post.calls[0][1]["temperature"] == 0.7


def test_request_carries_tools_only_when_the_item_has_them():
    tool = {"type": "function", "function": {"name": "choose", "description": "", "parameters": {"type": "object", "properties": {}}}}
    post = _script(_resp("x"))
    _gen(post)([_item(0, tools=[tool]), _item(1, tools=None), _item(2, tools=[])])
    with_tools, none, empty = (c[1] for c in post.calls)
    assert with_tools["tools"] == [tool] and with_tools["tool_choice"] == "auto"
    assert "tools" not in none and "tool_choice" not in none
    assert "tools" not in empty and "tool_choice" not in empty


def test_request_carries_reasoning_effort_only_when_set():
    post = _script(_resp("x"))
    _gen(post, effort="low")([_item()])
    _gen(post)([_item()])
    assert post.calls[0][1]["reasoning"] == {"effort": "low"}
    assert "reasoning" not in post.calls[1][1]


def test_backend_max_tokens_overrides_the_items():
    post = _script(_resp("x"))
    _gen(post, max_tokens=300)([_item(max_tokens=50)])
    _gen(post)([_item(max_tokens=50)])
    assert [c[1]["max_tokens"] for c in post.calls] == [300, 50]


def test_history_is_sent_as_structured_tool_calls():
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "on it\n<tool_call>\n<function=lookup>\n<parameter=q>\nx\n</parameter>\n</function>\n</tool_call>"},
            {"role": "tool", "name": "lookup", "content": "result"}]
    post = _script(_resp("done"))
    _gen(post)([_item(messages=msgs)])
    sent = post.calls[0][1]["messages"]
    assert sent == to_openai(msgs)
    assert sent[1]["tool_calls"][0]["function"]["name"] == "lookup" and sent[2]["tool_call_id"] == sent[1]["tool_calls"][0]["id"]


def test_results_come_back_in_item_order():
    gen = orbackend.make_or_backend("m", mini="T", cap=10.0, workers=8, post=lambda u, b, h: _resp(f"s{b['seed']}"))
    outs = gen([_item(i) for i in range(40)])
    assert [t for t, _ in outs] == [f"s{i}" for i in range(40)]


def test_works_as_the_engines_generate():
    gen = _gen(_script(_resp("hello there")))
    eps = run_batch([{"key": "j0", "exp": "t", "case": None, "cond": "P0", "thinking": False, "seed": 3,
                      "prompt": "Say hi"}], gen, log=lambda *a: None)
    assert eps[0]["end"] == "reply" and eps[0]["turns"][0]["finish"] == "stop"
    assert eps[0]["messages"][-1]["content"] == "hello there" and gen.calls == 1


# ---------------------------------------------------------------- Review Focus 1: errors, timeouts, no choices

def test_retries_back_off_three_times_two_to_the_k(sleeps):
    post = _script(_http(500))
    out = _gen(post)([_item()])
    assert out == [("", "error:http_500")]
    assert len(post.calls) == 5 and sleeps == [3, 6, 12, 24]


def test_retry_sleep_scales_the_backoff(sleeps):
    _gen(_script(_http(503)), retry_sleep=0.5)([_item()])
    assert sleeps == [0.5, 1.0, 2.0, 4.0]


def test_a_retry_that_succeeds_is_not_an_error(sleeps):
    post = _script(_http(429), _http(502), _resp("fine"))
    gen = _gen(post)
    assert gen([_item()]) == [("fine", "stop")]
    assert len(post.calls) == 3 and sleeps == [3, 6]
    assert gen.errors == {} and gen.calls == 1


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504, 529, 599])
def test_retryable_http_codes(code):
    post = _script(_http(code))
    out = _gen(post)([_item()])
    assert out == [("", f"error:http_{code}")] and len(post.calls) == 5


@pytest.mark.parametrize("code", [400, 401, 402, 403, 404, 422])
def test_other_http_errors_fail_at_once(code, sleeps):
    post = _script(_http(code))
    gen = _gen(post)
    assert gen([_item()]) == [("", f"error:http_{code}")]
    assert len(post.calls) == 1 and sleeps == [] and gen.errors == {f"http_{code}": 1}


TRANSIENT = [
    (TimeoutError("read timed out"), "timeout"),
    (socket.timeout("timed out"), "timeout"),
    (urllib.error.URLError(TimeoutError("connect timed out")), "timeout"),
    (urllib.error.URLError("name resolution failed"), "network"),
    (ConnectionResetError("reset by peer"), "network"),
    (http.client.RemoteDisconnected("closed"), "network"),
    (http.client.IncompleteRead(b"par"), "network"),
    (json.JSONDecodeError("Expecting value", "<html>", 0), "bad_json"),
    (UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"), "bad_json"),
]


@pytest.mark.parametrize("exc, what", TRANSIENT, ids=[type(e).__name__ + "->" + w for e, w in TRANSIENT])
def test_timeouts_and_network_errors_are_retried(exc, what, sleeps):
    post = _script(exc)
    gen = _gen(post)
    assert gen([_item()]) == [("", f"error:{what}")]
    assert len(post.calls) == 5 and sleeps == [3, 6, 12, 24] and gen.errors == {what: 1}


def test_an_unknown_exception_is_not_retried():
    post = _script(RuntimeError("boom"))
    gen = _gen(post)
    assert gen([_item()]) == [("", "error:RuntimeError")]
    assert len(post.calls) == 1 and gen.errors == {"RuntimeError": 1}


@pytest.mark.parametrize("reply", [{"choices": []}, {"choices": None}, {}, {"error": {"code": 502, "message": "upstream"}}],
                         ids=["empty", "null", "bare", "error object"])
def test_no_choices_is_retried_then_an_error(reply):
    post = _script(reply)
    gen = _gen(post)
    assert gen([_item()]) == [("", "error:no_choices")]
    assert len(post.calls) == 5 and gen.errors == {"no_choices": 1}


def test_no_choices_then_a_real_reply():
    gen = _gen(_script({"choices": []}, _resp("ok")))
    assert gen([_item()]) == [("ok", "stop")] and gen.errors == {}


def test_a_provider_error_finish_is_retried_and_its_text_dropped():
    post = _script(_resp("partial", finish="error"))
    gen = _gen(post)
    assert gen([_item()]) == [("", "error:provider_error")]
    assert len(post.calls) == 5 and gen.errors == {"provider_error": 1}


@pytest.mark.parametrize("reply", [
    "not json", ["a list"], None, {"choices": [None]}, {"choices": ["str"]}, {"choices": [{"finish_reason": "stop"}]},
    {"choices": [{"message": None, "finish_reason": "stop"}]}, {"choices": [{"message": {"content": 5}, "finish_reason": "stop"}]},
    {"choices": [{"message": {"content": "", "tool_calls": [{"id": "c0"}]}, "finish_reason": "tool_calls"}]},
    {"choices": [{"message": {"content": "", "tool_calls": ["junk"]}, "finish_reason": "tool_calls"}]},
    {"choices": [{"message": {"content": "", "tool_calls": [{"id": "c0", "function": {"name": None, "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]},
    {"choices": [{"message": {"content": "", "tool_calls": [{"id": "c0", "function": {"name": "", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]},
])
def test_a_malformed_reply_is_an_error_not_a_crash(reply):
    gen = _gen(_script(reply))
    (text, fin), = gen([_item()])
    assert text == "" and fin in ("error:bad_response", "error:no_choices")
    assert sum(gen.errors.values()) == 1


def test_one_failing_item_does_not_stop_the_batch():
    def post(u, b, h):
        if b["seed"] == 2:
            raise _http(500)
        return _resp(f"s{b['seed']}")
    gen = orbackend.make_or_backend("m", mini="T", cap=1.0, workers=4, post=post)
    outs = gen([_item(i) for i in range(6)])
    assert outs == [("s0", "stop"), ("s1", "stop"), ("", "error:http_500"), ("s3", "stop"), ("s4", "stop"), ("s5", "stop")]
    assert gen.errors == {"http_500": 1} and gen.calls == 5


def test_a_malformed_item_is_that_items_error():
    post = _script(_resp("ok"))
    gen = _gen(post)
    bad = _item(1)
    del bad["messages"]
    assert gen([_item(0), bad, _item(2)]) == [("ok", "stop"), ("", "error:KeyError"), ("ok", "stop")]
    assert len(post.calls) == 2 and gen.errors == {"KeyError": 1}


def test_errors_are_counted_per_item_not_per_attempt():
    gen = _gen(_script(_http(500)))
    gen([_item(0), _item(1), _item(2)])
    assert gen.errors == {"http_500": 3}


def test_each_kind_of_error_is_logged_once(capsys):
    def post(u, b, h):
        raise _http(500 if b["seed"] < 3 else 400)
    gen = _gen(post)
    gen([_item(i) for i in range(5)])
    err = capsys.readouterr().err
    assert err.count("[or]") == 2 and "http_500" in err and "http_400" in err and "err body" in err


# ---------------------------------------------------------------- cost accounting

def test_spent_adds_up_and_calls_counts_replies():
    gen = _gen(_script(_resp("x", cost=0.25)), cap=100)
    gen([_item(i) for i in range(4)])
    assert gen.spent == 1.0 and gen.calls == 4 and gen.errors == {}


_NO_COST = {"no usage": lambda r: r.pop("usage"), "usage null": lambda r: r.update(usage=None),
            "usage not a dict": lambda r: r.update(usage="free"), "empty usage": lambda r: r.update(usage={}),
            "cost null": lambda r: r["usage"].update(cost=None), "cost text": lambda r: r["usage"].update(cost="abc"),
            "cost nan": lambda r: r["usage"].update(cost=float("nan")),
            "cost inf": lambda r: r["usage"].update(cost=float("inf")),
            "cost -inf": lambda r: r["usage"].update(cost=float("-inf")),
            "cost negative": lambda r: r["usage"].update(cost=-0.5), "cost bool": lambda r: r["usage"].update(cost=True),
            "cost list": lambda r: r["usage"].update(cost=[0.1]), "cost huge": lambda r: r["usage"].update(cost=10 ** 400)}


@pytest.mark.parametrize("how", list(_NO_COST))
def test_a_missing_or_bad_cost_counts_as_zero_and_is_counted(how, ledger):
    reply = _resp("fine")
    _NO_COST[how](reply)
    gen = _gen(_script(reply))
    assert gen([_item(0), _item(1)]) == [("fine", "stop")] * 2           # the replies are still returned
    assert gen.spent == 0.0 and gen.errors == {"cost_missing": 2} and gen.calls == 2
    gen.close()
    assert [r["usd"] for r in _rows(ledger)] == [0.0]                    # and the ledger line is still valid


@pytest.mark.parametrize("cost, expected", [(0, 0.0), (3, 3.0), (0.25, 0.25), ("0.25", 0.25), (1e-9, 1e-9)])
def test_a_good_cost_is_charged(cost, expected):
    gen = _gen(_script(_resp("x", cost=cost)))
    gen([_item()])
    assert gen.spent == expected and gen.errors == {}                    # a free reply (cost 0) is not "missing"


def test_a_bad_cost_does_not_poison_the_total():
    good, bad = _resp("x", cost=0.25), _resp("x", cost=float("nan"))
    gen = _gen(_script(good, bad, good), cap=100)
    gen([_item(i) for i in range(3)])
    assert gen.spent == 0.5 and gen.errors == {"cost_missing": 1}


def test_failed_attempts_that_report_a_cost_are_charged():
    reply = {"choices": [], "usage": {"cost": 0.25}}
    gen = _gen(_script(reply), cap=100)
    assert gen([_item()]) == [("", "error:no_choices")]
    assert gen.spent == 1.25 and gen.errors == {"no_choices": 1}         # five attempts, each billed; no cost_missing
    assert gen.calls == 0                                                # .calls counts usable replies only


def test_an_unusable_reply_without_a_cost_is_not_cost_missing():
    gen = _gen(_script({"choices": []}))
    gen([_item()])
    assert gen.spent == 0.0 and gen.errors == {"no_choices": 1}


# ---------------------------------------------------------------- Review Focus 2: the cap

def test_cost_stop_returns_the_marker_and_makes_no_more_calls():
    post = _script(_resp("x", cost=0.25))
    gen = _gen(post, cap=0.5)
    outs = gen([_item(i) for i in range(4)])
    assert outs == [("x", "stop"), ("x", "stop"), ("", "cost_stop"), ("", "cost_stop")]
    assert len(post.calls) == 2 and gen.spent == 0.5 and gen.calls == 2   # stopped when spend REACHED the cap
    assert gen.errors == {}                                                # a stop is not an error


def test_what_was_spent_before_the_stop_is_still_written_to_the_ledger(ledger):
    gen = _gen(_script(_resp("x", cost=0.25)), cap=0.5)
    gen([_item(i) for i in range(4)])
    gen.close()
    (row,) = _rows(ledger)
    assert (row["mini"], row["kind"], row["usd"], row["note"]) == ("T", "api", 0.5, "m")
    assert util.ledger_total("T") == gen.spent


def test_a_zero_cap_stops_everything():
    post = _script(_resp("x"))
    outs = _gen(post, cap=0)([_item(i) for i in range(3)])
    assert outs == [("", "cost_stop")] * 3 and post.calls == []


def test_the_cap_is_checked_before_every_attempt_not_just_every_item():
    post = _script({"choices": [], "usage": {"cost": 0.25}}, _resp("never reached"))
    assert _gen(post, cap=0.25)([_item()]) == [("", "cost_stop")]
    assert len(post.calls) == 1


def test_a_failed_item_is_not_a_cost_stop_when_there_is_budget_left():
    assert _gen(_script(_http(500)), cap=100)([_item()]) == [("", "error:http_500")]


def test_parallel_overshoot_is_bounded_by_the_workers():
    lock, n = threading.Lock(), [0]

    def post(u, b, h):
        with lock:
            n[0] += 1
        return _resp("x", cost=0.25)
    gen = orbackend.make_or_backend("m", mini="T", cap=1.0, workers=4, post=post)
    outs = gen([_item(i) for i in range(60)])
    finishes = [f for _, f in outs]
    # 4 replies reach the cap; at most 3 more were already in flight when it was reached
    assert 4 <= n[0] <= 4 + 3
    assert finishes.count("stop") == n[0] and finishes.count("cost_stop") == 60 - n[0]
    assert 1.0 <= gen.spent <= 1.0 + 3 * 0.25 and gen.spent == 0.25 * n[0]


def test_parallel_updates_are_not_lost():
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)                      # make thread switches frequent
    try:
        gen = orbackend.make_or_backend("m", mini="T", cap=1000, workers=16, post=lambda u, b, h: _resp("x", cost=0.25))
        gen([_item(i) for i in range(400)])
    finally:
        sys.setswitchinterval(old)
    assert gen.spent == 100.0 and gen.calls == 400 and gen.errors == {}


def test_spend_and_call_counts_are_only_written_under_the_lock():
    class Guarded(orbackend.ORBackend):
        """Raises if spent or calls is written while the backend's lock is free."""
        def __setattr__(self, name, value):
            lock = self.__dict__.get("_lock")
            if name in ("spent", "calls") and lock is not None and not lock.locked():
                raise AssertionError(f"{name} written without the lock")
            super().__setattr__(name, value)
    gen = Guarded("m", mini="T", cap=10, thinking=False, effort=None, temperature=0.7, max_tokens=None, workers=1,
                  post=lambda u, b, h: _resp("x", cost=0.25), headers={}, retry_sleep=0)
    assert gen([_item(0), _item(1)]) == [("x", "stop")] * 2
    assert gen.spent == 0.5 and gen.calls == 2


@pytest.mark.parametrize("cap", [float("nan"), float("inf"), -1, -0.01, "5", None, True])
def test_a_bad_cap_is_refused_up_front(cap):
    with pytest.raises(ValueError, match="cap"):
        orbackend.make_or_backend("m", mini="T", cap=cap, post=lambda u, b, h: _resp())


@pytest.mark.parametrize("kw", [{"mini": ""}, {"mini": "  "}, {"mini": None}, {"workers": 0}, {"workers": -2},
                                {"retry_sleep": -1}, {"retry_sleep": float("nan")}])
def test_bad_arguments_are_refused_up_front(kw):
    args = {"mini": "T", "cap": 1.0, "post": lambda u, b, h: _resp()} | kw
    with pytest.raises(ValueError):
        orbackend.make_or_backend("m", **args)


# ---------------------------------------------------------------- close(): the ledger, exactly once

def test_close_writes_the_spend_once(ledger):
    gen = _gen(_script(_resp("x", cost=0.25)), cap=100)
    gen([_item(i) for i in range(3)])
    assert _rows(ledger) == []                       # nothing is written until close()
    gen.close()
    gen.close()
    gen.close()
    (row,) = _rows(ledger)
    assert list(row) == ["mini", "date", "kind", "usd", "note"]
    assert (row["mini"], row["kind"], row["usd"], row["note"]) == ("T", "api", 0.75, "m")


def test_close_with_no_spend_still_writes_one_zero_line(ledger):
    gen = _gen(_script(_http(500)))
    gen([_item()])
    gen.close()
    assert [r["usd"] for r in _rows(ledger)] == [0.0] and util.ledger_total("T") == 0.0


def test_close_from_many_threads_writes_once(ledger):
    gen = _gen(_script(_resp("x", cost=0.25)))
    gen([_item()])
    threads = [threading.Thread(target=gen.close) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(_rows(ledger)) == 1


def test_a_failed_ledger_write_can_be_retried(ledger, monkeypatch):
    gen = _gen(_script(_resp("x", cost=0.25)))
    gen([_item()])
    real = orbackend.ledger_add

    def disk_full(*args, **kw):
        raise OSError("disk full")
    monkeypatch.setattr(orbackend, "ledger_add", disk_full)
    with pytest.raises(OSError):
        gen.close()
    assert _rows(ledger) == []
    monkeypatch.setattr(orbackend, "ledger_add", real)      # the disk is back: the spend must still get in, once
    gen.close()
    gen.close()
    assert [r["usd"] for r in _rows(ledger)] == [0.25]


def test_generate_after_close_is_refused(ledger):
    gen = _gen(_script(_resp("x")))
    gen.close()
    with pytest.raises(RuntimeError, match="close"):
        gen([_item()])
    assert len(_rows(ledger)) == 1


def test_two_backends_write_their_own_lines(ledger):
    a, b = _gen(_script(_resp("x", cost=0.25))), orbackend.make_or_backend("m2", mini="U", cap=1, post=_script(_resp("y", cost=0.5)))
    a([_item()]); b([_item()])
    a.close(); b.close()
    assert util.ledger_total("T") == 0.25 and util.ledger_total("U") == 0.5
    assert {r["note"] for r in _rows(ledger)} == {"m", "m2"}


# ---------------------------------------------------------------- the real transport: default_post and the key

def test_without_post_the_key_comes_from_load_key_and_default_post_is_used(monkeypatch):
    stub = types.ModuleType("persona.operative.frontier")
    stub.load_key = lambda: "sk-test"
    monkeypatch.setitem(sys.modules, "persona.operative.frontier", stub)
    seen = []

    def fake_default_post(url, body, headers, timeout=180):
        seen.append((url, body, headers))
        return _resp("hi")
    monkeypatch.setattr(orbackend, "default_post", fake_default_post)
    gen = orbackend.make_or_backend("openai/gpt-5.6-sol", mini="T", cap=1.0)
    assert gen([_item(0)]) == [("hi", "stop")]
    (url, body, headers), = seen
    assert url == URL and body["model"] == "openai/gpt-5.6-sol" and headers == {"authorization": "Bearer sk-test"}


def test_an_injected_post_never_touches_the_key(monkeypatch):
    stub = types.ModuleType("persona.operative.frontier")

    def no_key():
        raise AssertionError("load_key must not be called when post is injected")
    stub.load_key = no_key
    monkeypatch.setitem(sys.modules, "persona.operative.frontier", stub)
    assert _gen(_script(_resp("ok")))([_item()]) == [("ok", "stop")]


def test_importing_the_backend_does_not_import_the_key_loader_or_patch_dns():
    code = ("import socket, sys; g = socket.getaddrinfo\n"
            "import persona_affect.common.orbackend\n"
            "assert 'persona.operative.frontier' not in sys.modules\n"
            "assert socket.getaddrinfo is g\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT), "PATH": ""})
    assert r.returncode == 0, r.stderr


class _Body:
    """What urlopen returns: a context manager with read()."""
    def __init__(self, raw):
        self.raw = raw

    def read(self):
        return self.raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_default_post_sends_json_and_returns_json(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen.update(req=req, timeout=timeout)
        return _Body(b'{"ok": 1, "msg": "caf\xc3\xa9"}')
    monkeypatch.setattr(orbackend.urllib.request, "urlopen", fake_urlopen)
    out = orbackend.default_post("https://example.test/x", {"a": 1, "t": "café"}, {"authorization": "Bearer k"})
    req = seen["req"]
    assert out == {"ok": 1, "msg": "café"} and seen["timeout"] == 180
    assert req.full_url == "https://example.test/x" and req.get_method() == "POST"
    assert json.loads(req.data) == {"a": 1, "t": "café"}
    assert req.get_header("Content-type") == "application/json" and req.get_header("Authorization") == "Bearer k"


def test_default_post_timeout_is_a_parameter(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["timeout"] = timeout
        return _Body(b"{}")
    monkeypatch.setattr(orbackend.urllib.request, "urlopen", fake_urlopen)
    orbackend.default_post("https://example.test/x", {}, {}, timeout=7)
    assert seen["timeout"] == 7


def test_default_post_does_not_swallow_errors(monkeypatch):
    def fail(req, timeout=None):
        raise _http(503)
    monkeypatch.setattr(orbackend.urllib.request, "urlopen", fail)
    with pytest.raises(urllib.error.HTTPError) as e:
        orbackend.default_post("https://example.test/x", {}, {})
    assert e.value.code == 503
    monkeypatch.setattr(orbackend.urllib.request, "urlopen", lambda req, timeout=None: _Body(b"<html>bad gateway</html>"))
    with pytest.raises(ValueError):
        orbackend.default_post("https://example.test/x", {}, {})


def test_default_post_errors_are_classified_by_the_backend(monkeypatch):
    """The real transport raises urllib's errors; the backend must retry and label them."""
    def fail(req, timeout=None):
        raise _http(500)
    monkeypatch.setattr(orbackend.urllib.request, "urlopen", fail)
    stub = types.ModuleType("persona.operative.frontier")
    stub.load_key = lambda: "sk-test"
    monkeypatch.setitem(sys.modules, "persona.operative.frontier", stub)
    gen = orbackend.make_or_backend("m", mini="T", cap=1.0, workers=1)
    assert gen([_item()]) == [("", "error:http_500")]
