"""Tests for persona_affect.common.judges: two-judge labelling and their agreement.

Run from the repo root: uv run pytest persona_affect/tests/test_common_judges.py -q
No network and no API key: nearly every label() gets a fake `post`. The tests that build the real transport (no `post`)
stub `persona.operative.frontier.load_key` and replace `orbackend.default_post`. Every test points util.LEDGER at a tmp
file, so the real persona_affect/ledger.jsonl is never touched, and time.sleep is replaced by a recorder, so no test
waits on a backoff.

Result shape under test: label() returns {item key: {judge: label or None}} (item-major).
"""
import hashlib
import io
import json
import subprocess
import sys
import threading
import types
import urllib.error
from pathlib import Path

import pytest

from persona_affect.common import judges, orbackend, util
from vigilante.src.judge import kappa

H, G = "anthropic/claude-haiku-5.5", "openai/gpt-5-mini"
YN = {"yes", "no"}
SUFFIX_YN = "\n\nAnswer with exactly one of: no, yes"
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
    monkeypatch.setattr(judges.time, "sleep", rec.append)
    return rec


@pytest.fixture
def cache(tmp_path):
    return tmp_path / "judge_cache.jsonl"


def _resp(text="yes", cost=0.001, finish="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}], "usage": {"cost": cost}}


def _http(code):
    return urllib.error.HTTPError(orbackend.URL, code, "x", {}, io.BytesIO(b"err body"))


def _fake(reply):
    """A fake post. `reply` is a decoded reply, an exception, or a function of the request body returning either.
    post.calls records (url, body, headers) of every call, in call order."""
    lock = threading.Lock()

    def post(url, body, headers):
        with lock:
            post.calls.append((url, body, headers))
        r = reply(body) if callable(reply) else reply
        if isinstance(r, BaseException):
            raise r
        return r
    post.calls = []
    return post


def _script(*replies):
    """A fake post that plays `replies` in order (an exception is raised, anything else returned); the last repeats."""
    lock = threading.Lock()

    def post(url, body, headers):
        with lock:
            post.calls.append((url, body, headers))
            r = replies[min(len(post.calls), len(replies)) - 1]
        if isinstance(r, BaseException):
            raise r
        return r
    post.calls = []
    return post


def _item(i=0, **kw):
    return {"key": f"k{i}", "text": f"reply number {i}", **kw}


def _render(item):
    return f"Is this reply friendly?\n{item['text']}"


def _prompt(i=0):
    return _render(_item(i)) + SUFFIX_YN


_UNSET = object()


def _label(post, cache, items=_UNSET, allowed=_UNSET, **kw):
    kw.setdefault("mini", "T")
    kw.setdefault("cap", 10.0)
    kw.setdefault("workers", 1)
    return judges.label([_item(0)] if items is _UNSET else items, _render, YN if allowed is _UNSET else allowed,
                        cache=cache, post=post, **kw)


def _rows(path):
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def _sha(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _stub_frontier(monkeypatch, load_key):
    stub = types.ModuleType("persona.operative.frontier")
    stub.load_key = load_key
    monkeypatch.setitem(sys.modules, "persona.operative.frontier", stub)


# ---------------------------------------------------------------- the brief's four tests

def test_yes_with_a_period_parses_to_yes(cache):
    assert _label(_fake(_resp("Yes.")), cache) == {"k0": {H: "yes", G: "yes"}}


def test_an_off_set_reply_is_none(cache):
    assert _label(_fake(_resp("maybe")), cache) == {"k0": {H: None, G: None}}


def test_a_second_call_with_the_same_prompt_hits_the_cache(cache):
    post = _fake(_resp("yes"))
    _label(post, cache)
    _label(post, cache)
    assert len(post.calls) == 2                      # 1 item x 2 judges, not 4


def test_agreement_on_identical_labels_is_kappa_1():
    labels = {f"k{i}": {H: v, G: v} for i, v in enumerate(["yes", "no", "yes", "no"])}
    assert judges.agreement(labels) == {"kappa": 1.0, "n": 4, "none": {H: 0, G: 0},
                                        "per_class": {"no": 1.0, "yes": 1.0}}


# ---------------------------------------------------------------- the request

def test_the_judges_are_haiku_and_gpt_5_mini():
    assert judges.JUDGES == ("anthropic/claude-haiku-5.5", "openai/gpt-5-mini")


def test_request_bodies_are_exactly_the_specified_ones(cache):
    post = _fake(_resp("yes"))
    _label(post, cache)
    (u1, b1, h1), (u2, b2, h2) = post.calls
    assert u1 == u2 == "https://openrouter.ai/api/v1/chat/completions" and h1 == h2 == {}
    msgs = [{"role": "user", "content": _prompt()}]
    assert b1 == {"model": H, "messages": msgs, "max_tokens": 400, "temperature": 0, "usage": {"include": True},
                  "reasoning": {"effort": "none"}}
    assert b2 == {"model": G, "messages": msgs, "max_tokens": 400, "temperature": 0, "usage": {"include": True},
                  "reasoning": {"effort": "minimal"}}


def test_the_prompt_is_render_plus_the_sorted_answer_suffix(cache):
    post = _fake(_resp("0"))
    allowed = {"2", "10", "0", "1"}
    _label(post, cache, allowed=allowed, judges=(H,))
    sent = post.calls[0][1]["messages"][0]["content"]
    assert sent == _render(_item(0)) + "\n\nAnswer with exactly one of: 0, 1, 10, 2"


def test_a_non_ascii_prompt_is_sent_as_is_and_hashed_as_utf8(cache):
    text = "café ☃ — ünïcode 日本語 😀"            # BMP, CJK and a character outside the BMP: all unchanged
    post = _fake(_resp("yes"))
    judges.label([{"key": "u"}], lambda it: text, YN, mini="T", cap=1.0, cache=cache, post=post, workers=1)
    full = text + SUFFIX_YN
    assert post.calls[0][1]["messages"][0]["content"] == full
    assert {r["sha"] for r in _rows(cache)} == {_sha(full)}


def _surrogate_batch(cache, texts, **kw):
    """label() over items whose rendered prompt is "q " + text; returns (labels, the post)."""
    items = [{"key": f"i{n}", "text": t} for n, t in enumerate(texts)]
    post = _fake(_resp("yes"))
    out = judges.label(items, lambda it: "q " + it["text"], YN, mini="T", cap=1.0, cache=cache, post=post,
                       workers=1, judges=(H,), **kw)
    return out, post


def test_a_lone_surrogate_in_one_prompt_does_not_stop_the_batch(cache, capsys):
    out, post = _surrogate_batch(cache, ["fine", "cut \ud83d here", "also fine"])
    assert out == {"i0": {H: "yes"}, "i1": {H: "yes"}, "i2": {H: "yes"}}      # that item is labelled too
    sent = [c[1]["messages"][0]["content"] for c in post.calls]
    assert sent[1] == "q cut \ufffd here" + SUFFIX_YN                         # replaced, not dropped
    assert not any("\ud800" <= ch <= "\udfff" for text in sent for ch in text)
    assert {r["sha"] for r in _rows(cache)} == {_sha(t) for t in sent}        # cached under what was sent
    err = capsys.readouterr().err
    assert err.count("surrogate") == 1 and "1 prompt" in err


def test_a_lone_low_surrogate_and_several_prompts_are_counted(cache, capsys):
    out, post = _surrogate_batch(cache, ["a \ude00 b", "c \ud800", "\udfff", "clean"])
    assert out == {f"i{n}": {H: "yes"} for n in range(4)}
    assert [c[1]["messages"][0]["content"] for c in post.calls][:3] == [
        "q a \ufffd b" + SUFFIX_YN, "q c \ufffd" + SUFFIX_YN, "q \ufffd" + SUFFIX_YN]
    assert "3 prompt" in capsys.readouterr().err


def test_a_high_and_a_low_surrogate_side_by_side_are_the_character_they_stand_for(cache):
    # built with chr(): a literal "\ud83d\ude00" is joined into one character when the source is compiled
    text = "smile " + chr(0xD83D) + chr(0xDE00) + " done"
    with pytest.raises(UnicodeEncodeError):
        text.encode("utf-8")                                       # so this really is the invalid input
    out, post = _surrogate_batch(cache, [text])
    assert post.calls[0][1]["messages"][0]["content"] == "q smile \U0001f600 done" + SUFFIX_YN
    assert out == {"i0": {H: "yes"}}


def test_a_prompt_with_a_lone_surrogate_is_a_cache_hit_on_the_rerun(cache):
    _surrogate_batch(cache, ["x \ud83d y"])
    out, post = _surrogate_batch(cache, ["x \ud83d y"])
    assert post.calls == [] and out.cached == 1 and out == {"i0": {H: "yes"}}


def test_a_lone_surrogate_in_a_reply_does_not_stop_the_batch(cache, capsys):
    out = _label(_fake(_resp("yes \ud83d")), cache, items=[_item(0), _item(1)], judges=(H,))
    assert out == {"k0": {H: "yes"}, "k1": {H: "yes"}}                         # "yes" is still the only label
    again = _label(_fake(_resp("no")), cache, items=[_item(0), _item(1)], judges=(H,))
    assert again.calls == 0 and again == out                                   # and the cache file reads back
    capsys.readouterr()


def test_a_judge_with_no_reasoning_setting_is_refused_until_one_is_added(cache, monkeypatch):
    post = _fake(_resp("yes"))
    with pytest.raises(ValueError, match="EFFORT"):
        _label(post, cache, judges=("x/new-model",))
    assert post.calls == []


@pytest.mark.parametrize("effort, sent", [(None, None), ("low", {"effort": "low"})])
def test_effort_can_be_added_for_another_judge(effort, sent, cache, monkeypatch):
    monkeypatch.setitem(judges.EFFORT, "x/new-model", effort)
    post = _fake(_resp("yes"))
    assert _label(post, cache, judges=("x/new-model",)) == {"k0": {"x/new-model": "yes"}}
    assert post.calls[0][1].get("reasoning") == sent
    assert ("reasoning" in post.calls[0][1]) == (sent is not None)


# ---------------------------------------------------------------- parsing the reply

PARSE_YN = [
    ("Yes.", "yes"), ("yes", "yes"), ("YES", "yes"), (" \n No \n", "no"), ("**Yes**", "yes"), ('"no"', "no"),
    ("(yes)", "yes"), ("Yes!", "yes"), ("yes,", "yes"), ("\u201cYes\u201d", "yes"), ("- yes", "yes"), ("1. no", "no"),
    ("Answer: yes", "yes"), ("The answer is no.", "no"), ("yes\n\nBecause the reply is friendly.", "yes"),
    ("`yes`", "yes"), ("_yes_", "yes"), ("[no]", "no"), ("\u00abYes\u00bb", "yes"), ("yes?!", "yes"),
    ("yes\u2026", "yes"), ("**Answer:** no", "no"),
    ("maybe", None), ("", None), ("   ", None), ("unsure", None), ("not sure", None), ("nobody", None),
    ("yesterday", None), ("y", None), ("yes/no", None), ("n0", None),
]


@pytest.mark.parametrize("reply, expected", PARSE_YN, ids=[repr(r) for r, _ in PARSE_YN])
def test_parsing_yes_no(reply, expected, cache):
    out = _label(_fake(_resp(reply)), cache, judges=(H,))
    if reply.strip():
        assert out == {"k0": {H: expected}}
    else:
        assert out == {"k0": {H: None}}               # a blank reply is a failed call, not an answer


SCALE = {str(i) for i in range(11)}
PARSE_SCALE = [
    ("7", "7"), ("10", "10"), ("1", "1"), ("0", "0"), ("Score: 3", "3"), ("3.", "3"), ("**7**", "7"), ("(5)", "5"),
    ("5:", "5"), ("`8`", "8"), ("\u201c6\u201d", "6"), ("[2]", "2"), ("9!", "9"), ("4)", "4"),
    ("7.5", None), ("7/10", None), ("11", None), ("ten", None), ("1,000", None),
    # a sign, a currency symbol or a percent sign is part of the number: not the number next to it (review F1)
    ("-1", None), ("+5", None), ("$5", None), ("5%", None), ("-1.", None), ("(+5)", None), ("**$5**", None),
    (".5", None), ("~5", None), ("#5", None), ("\u20ac5", None), ("5\u00b0", None), ("5$", None), ("8+", None),
]


@pytest.mark.parametrize("reply, expected", PARSE_SCALE, ids=[repr(r) for r, _ in PARSE_SCALE])
def test_parsing_a_0_to_10_scale_never_reads_part_of_a_number(reply, expected, cache):
    assert _label(_fake(_resp(reply)), cache, allowed=SCALE, judges=(H,)) == {"k0": {H: expected}}


def test_the_off_set_kind_is_counted_for_a_number_that_is_not_a_label(cache):
    out = _label(_fake(_resp("-1")), cache, allowed=SCALE, judges=(H,))
    assert out == {"k0": {H: None}} and out.none[H] == {"off_set": 1}


@pytest.mark.parametrize("reply, expected", [("-1", "-1"), ("1", "1"), ("0", "0"), ("-1.", "-1"), ("**-1**", "-1"),
                                             ("(-1)", "-1"), ("+1", None), ("2", None), ("--1", None)])
def test_a_sign_belongs_to_the_label_so_minus_one_is_usable(reply, expected, cache):
    assert _label(_fake(_resp(reply)), cache, allowed={"-1", "0", "1"}, judges=(H,)) == {"k0": {H: expected}}


# Two or more different allowed labels in one reply: ambiguous, so None with the reason "multi" (review F2), never the
# first one found. The first four are the review's examples.
MULTI = [
    ("On a scale of 0 to 10, 7", SCALE), ("Based on the 3 criteria, the score is 7", SCALE), ("1. 7", SCALE),
    ("I can't say yes or no", YN),
    ("No, but yes in part", YN), ("yes or no", YN), ("Yes. No.", YN), ("I'd say 4 out of 10", SCALE),
    ("7 (or maybe 8)", SCALE), ("1 2", SCALE),
]


@pytest.mark.parametrize("reply, allowed", MULTI, ids=[repr(r) for r, _ in MULTI])
def test_a_reply_with_two_different_allowed_labels_is_multi_not_the_first_one(reply, allowed, cache):
    out = _label(_fake(_resp(reply)), cache, allowed=allowed, judges=(H,))
    assert out == {"k0": {H: None}} and out.none[H] == {"multi": 1}


REPEATED = [
    ("7 ... 7", SCALE, "7"), ("Yes. Yes!", YN, "yes"), ("yes - yes - YES", YN, "yes"),
    ("score: 7 (that is, 7 out of ten)", SCALE, "7"), ("**7**: seven. 7.", SCALE, "7"),
]


@pytest.mark.parametrize("reply, allowed, expected", REPEATED, ids=[repr(r) for r, _, _ in REPEATED])
def test_the_same_label_repeated_is_still_that_label(reply, allowed, expected, cache):
    out = _label(_fake(_resp(reply)), cache, allowed=allowed, judges=(H,))
    assert out == {"k0": {H: expected}} and out.none[H] == {}


def test_multi_is_counted_per_judge_and_shown_once_per_judge(cache, capsys):
    reply = lambda b: _resp("yes or no" if b["model"] == H else "no")
    out = _label(_fake(reply), cache, items=[_item(i) for i in range(4)])
    assert out.none == {H: {"multi": 4}, G: {}}
    assert [row[G] for row in out.values()] == ["no"] * 4 and [row[H] for row in out.values()] == [None] * 4
    assert judges.agreement(out) == {"kappa": None, "n": 0, "none": {H: 4, G: 0}, "per_class": {}}
    err = capsys.readouterr().err
    assert err.count("[judges]") == 1 and "multi" in err and "yes or no" in err and H in err


def test_a_multi_reply_is_cached_like_any_real_answer(cache):
    post = _fake(_resp("yes or no"))
    _label(post, cache)
    out = _label(post, cache)
    assert len(post.calls) == 2 and out.none == {H: {"multi": 1}, G: {"multi": 1}}


def test_the_label_is_the_allowed_entry_as_the_caller_wrote_it(cache):
    assert _label(_fake(_resp("yes.")), cache, allowed={"Yes", "No"}, judges=(H,)) == {"k0": {H: "Yes"}}
    assert _label(_fake(_resp("NO_FEELINGS")), cache, allowed={"admits", "no_feelings"}, judges=(H,)) == \
        {"k0": {H: "no_feelings"}}


def test_a_truncated_reply_with_an_answer_is_still_read(cache):
    assert _label(_fake(_resp("yes", finish="length")), cache, judges=(H,)) == {"k0": {H: "yes"}}


@pytest.mark.parametrize("allowed", [set(), frozenset(), "yes", None, {"yes", ""}, {"yes", "!!"}, {"very high", "low"},
                                     {"Yes", "yes"}, ["yes", "yes"], {1, 2}, {"yes", None}])
def test_unusable_allowed_sets_are_refused_before_any_call(allowed, cache):
    post = _fake(_resp("yes"))
    with pytest.raises(ValueError, match="allowed"):
        _label(post, cache, allowed=allowed)
    assert post.calls == []


def test_allowed_may_be_a_list_or_tuple_too(cache):
    assert _label(_fake(_resp("no")), cache, allowed=["yes", "no"], judges=(H,)) == {"k0": {H: "no"}}
    assert _label(_fake(_resp("no")), cache, allowed=("yes", "no"), judges=(H,)) == {"k0": {H: "no"}}


# ---------------------------------------------------------------- the items, the result and its counters

def test_the_result_is_item_major_in_item_order(cache):
    post = _fake(lambda b: _resp("yes" if b["model"] == H else "no"))
    out = _label(post, cache, items=[_item(i) for i in (3, 1, 2)])
    assert list(out) == ["k3", "k1", "k2"]
    assert out == {k: {H: "yes", G: "no"} for k in ("k3", "k1", "k2")}
    assert list(out["k1"]) == [H, G]


def test_one_judge_gives_rows_of_one(cache):
    out = _label(_fake(_resp("yes")), cache, judges=(G,))
    assert out == {"k0": {G: "yes"}}


def test_the_result_is_a_dict_that_serialises(cache):
    out = _label(_fake(_resp("yes")), cache)
    assert isinstance(out, dict) and json.loads(json.dumps(out)) == {"k0": {H: "yes", G: "yes"}}


def test_render_is_called_once_per_item_with_the_item(cache):
    seen = []

    def render(item):
        seen.append(item)
        return "q " + item["text"]
    items = [_item(i) for i in range(3)]
    judges.label(items, render, YN, mini="T", cap=1.0, cache=cache, post=_fake(_resp("yes")), workers=1)
    assert seen == items


def test_no_items_means_no_calls_and_no_ledger_line(cache, ledger):
    post = _fake(_resp("yes"))
    out = _label(post, cache, items=[])
    assert out == {} and post.calls == [] and not ledger.exists()


@pytest.mark.parametrize("items", [
    [{"text": "x"}], [{"key": 1, "text": "x"}], [{"key": None}], ["not a dict"],
    [{"key": "a", "text": "x"}, {"key": "a", "text": "y"}],
], ids=["no key", "int key", "None key", "not a dict", "duplicate key"])
def test_items_need_unique_string_keys(items, cache):
    post = _fake(_resp("yes"))
    with pytest.raises(ValueError, match="key|item"):
        _label(post, cache, items=items)
    assert post.calls == []


def test_a_render_error_stops_the_run_before_anything_is_spent(cache, ledger):
    post = _fake(_resp("yes"))

    def render(item):
        if item["key"] == "k2":
            raise KeyError("missing field")
        return "q"
    with pytest.raises(KeyError):
        judges.label([_item(i) for i in range(4)], render, YN, mini="T", cap=1.0, cache=cache, post=post)
    assert post.calls == [] and not ledger.exists() and not cache.exists()


def test_a_render_that_returns_no_string_is_refused(cache):
    with pytest.raises(TypeError, match="render"):
        judges.label([_item(0)], lambda it: None, YN, mini="T", cap=1.0, cache=cache, post=_fake(_resp("yes")))


@pytest.mark.parametrize("kw", [
    {"mini": ""}, {"mini": "  "}, {"mini": None}, {"cap": float("nan")}, {"cap": float("inf")}, {"cap": -1},
    {"cap": "5"}, {"cap": True}, {"workers": 0}, {"workers": True}, {"workers": 1.5}, {"retry_sleep": -1},
    {"retry_sleep": float("nan")}, {"retry_sleep": True}, {"judges": ()}, {"judges": (H, H)}, {"judges": H},
], ids=str)
def test_bad_arguments_are_refused_before_any_call(kw, cache):
    post = _fake(_resp("yes"))
    with pytest.raises(ValueError):
        _label(post, cache, **kw)
    assert post.calls == []


def test_the_cache_path_may_be_a_string(cache):
    post = _fake(_resp("yes"))
    _label(post, str(cache))
    _label(post, str(cache))
    assert len(post.calls) == 2


def test_the_counters_of_a_clean_run(cache):
    out = _label(_fake(_resp("yes", cost=0.25)), cache, items=[_item(i) for i in range(3)])
    assert (out.spent, out.calls, out.cached, out.cost_missing) == (1.5, 6, 0, 0)
    assert out.none == {H: {}, G: {}} and out.truncated == {H: 0, G: 0}
    again = _label(_fake(_resp("yes", cost=0.25)), cache, items=[_item(i) for i in range(3)])
    assert (again.spent, again.calls, again.cached) == (0.0, 0, 6)


# ---------------------------------------------------------------- the cache

def test_cache_lines_are_model_sha_reply_finish(cache):
    _label(_fake(lambda b: _resp("Yes.\n" if b["model"] == H else "no")), cache)
    rows = _rows(cache)
    assert [list(r) for r in rows] == [["model", "sha", "reply", "finish"]] * 2      # "finish" is additive, last
    assert {(r["model"], r["sha"], r["reply"], r["finish"]) for r in rows} == {
        (H, _sha(_prompt()), "Yes.\n", "stop"), (G, _sha(_prompt()), "no", "stop")}


# ---------------------------------------------------------------- finish_reason (review F4)

def test_the_finish_reason_of_each_call_is_cached(cache):
    _label(_fake(lambda b: _resp("yes", finish="length" if b["model"] == G else "stop")), cache)
    assert {(r["model"], r["finish"]) for r in _rows(cache)} == {(H, "stop"), (G, "length")}


@pytest.mark.parametrize("how", ["missing", "null", "empty"])
def test_a_reply_without_a_finish_reason_is_cached_as_unknown(how, cache):
    reply = _resp("yes")
    if how == "missing":
        del reply["choices"][0]["finish_reason"]
    else:
        reply["choices"][0]["finish_reason"] = None if how == "null" else ""
    out = _label(_fake(reply), cache, judges=(H,))
    assert out == {"k0": {H: "yes"}} and [r["finish"] for r in _rows(cache)] == ["unknown"]
    assert out.truncated == {H: 0}


def test_cache_lines_from_before_finish_was_recorded_still_load(cache):
    old = json.dumps({"model": H, "sha": _sha(_prompt()), "reply": "no"}) + "\n"
    cache.write_text(old)
    post = _fake(_resp("yes"))
    out = _label(post, cache, judges=(H,))
    assert out == {"k0": {H: "no"}} and post.calls == [] and out.cached == 1 and out.truncated == {H: 0}
    assert cache.read_text() == old                                # a cache hit rewrites nothing
    _label(post, cache)                                            # the other judge is asked and written with finish
    assert [list(r) for r in _rows(cache)] == [["model", "sha", "reply"], ["model", "sha", "reply", "finish"]]


def test_a_finish_that_is_not_text_in_the_cache_is_read_as_unknown(cache):
    cache.write_text(json.dumps({"model": H, "sha": _sha(_prompt()), "reply": "no", "finish": 7}) + "\n")
    out = _label(_fake(_resp("yes")), cache, judges=(H,))
    assert out == {"k0": {H: "no"}} and out.truncated == {H: 0}


def test_truncated_replies_are_counted_per_judge_and_still_parsed(cache, capsys):
    post = _fake(lambda b: _resp("yes", finish="length" if b["model"] == G else "stop"))
    items = [_item(i) for i in range(3)]
    out = _label(post, cache, items=items)
    assert out == {k: {H: "yes", G: "yes"} for k in ("k0", "k1", "k2")}               # parsed as usual
    assert out.truncated == {H: 0, G: 3} and out.none == {H: {}, G: {}}
    err = capsys.readouterr().err
    assert err.count("[judges]") == 1 and "truncated" in err and G in err and "max_tokens" in err
    again = _label(post, cache, items=items)                                          # cache hits count as well
    assert again.calls == 0 and again.truncated == {H: 0, G: 3} and again == out


def test_a_truncated_reply_that_holds_no_label_is_both_truncated_and_off_set(cache):
    out = _label(_fake(_resp("Let me think about this carefully. The", finish="length")), cache, judges=(H,))
    assert out == {"k0": {H: None}} and out.truncated == {H: 1} and out.none[H] == {"off_set": 1}


def test_a_truncated_reply_is_counted_once_per_item_even_when_prompts_are_shared(cache):
    items = [{"key": "a", "text": "same"}, {"key": "b", "text": "same"}]
    out = _label(_fake(_resp("yes", finish="length")), cache, items=items, judges=(H,))
    assert out.truncated == {H: 2}


def test_a_cache_filled_by_one_judge_serves_a_run_with_both(cache):
    _label(_fake(_resp("yes")), cache, judges=(H,))
    second = _fake(_resp("no"))
    out = _label(second, cache)
    assert [c[1]["model"] for c in second.calls] == [G]           # Haiku's answer comes from the cache
    assert out == {"k0": {H: "yes", G: "no"}}
    assert (out.cached, out.calls) == (1, 1)


def test_an_off_set_reply_is_cached_like_any_real_answer(cache):
    post = _fake(_resp("maybe"))
    _label(post, cache)
    out = _label(post, cache)
    assert len(post.calls) == 2 and out == {"k0": {H: None, G: None}}
    assert out.none == {H: {"off_set": 1}, G: {"off_set": 1}}


def test_a_new_allowed_set_or_a_new_prompt_is_a_new_question(cache):
    post = _fake(_resp("yes"))
    _label(post, cache)
    _label(post, cache, allowed={"yes", "no", "unsure"})
    assert len(post.calls) == 4
    judges.label([_item(0)], lambda it: "different", YN, mini="T", cap=1.0, cache=cache, post=post, workers=1)
    assert len(post.calls) == 6


def test_identical_prompts_in_one_batch_are_asked_once(cache):
    post = _fake(_resp("yes"))
    items = [{"key": "a", "text": "same"}, {"key": "b", "text": "same"}, {"key": "c", "text": "other"}]
    out = _label(post, cache, items=items, workers=4)
    assert len(post.calls) == 4                                   # 2 distinct prompts x 2 judges
    assert out["a"] == out["b"] == {H: "yes", G: "yes"} and len(_rows(cache)) == 4


def test_a_failed_shared_prompt_is_none_for_every_item_that_shares_it(cache):
    items = [{"key": "a", "text": "same"}, {"key": "b", "text": "same"}]
    out = _label(_fake(_http(400)), cache, items=items, judges=(H,))
    assert out == {"a": {H: None}, "b": {H: None}} and out.none[H] == {"error:http_400": 2}


def test_unreadable_cache_lines_are_skipped_with_a_warning(cache, capsys):
    good = {"model": H, "sha": _sha(_prompt()), "reply": "no"}
    cache.write_text("not json\n" + json.dumps(good) + "\n" + '{"model": "x"}\n' + "[1, 2]\n" + "\n"
                     + json.dumps({"model": G, "sha": _sha(_prompt()), "reply": 5}) + "\n")
    post = _fake(_resp("yes"))
    out = _label(post, cache)
    assert out == {"k0": {H: "no", G: "yes"}} and [c[1]["model"] for c in post.calls] == [G]
    assert "unreadable" in capsys.readouterr().err


def test_a_cached_reply_with_no_text_is_asked_again(cache):
    cache.write_text(json.dumps({"model": H, "sha": _sha(_prompt()), "reply": "  "}) + "\n")
    post = _fake(_resp("yes"))
    assert _label(post, cache)["k0"] == {H: "yes", G: "yes"} and len(post.calls) == 2


def test_a_cache_cut_off_mid_line_is_not_fused_with_the_next_record(cache):
    cache.write_text('{"model": "x", "sha": "ab')                 # a killed run left no newline
    _label(_fake(_resp("yes")), cache)
    lines = cache.read_text().splitlines()
    assert len(lines) == 3 and json.loads(lines[1])["model"] == H and json.loads(lines[2])["model"] == G
    again = _fake(_resp("no"))
    _label(again, cache)
    assert again.calls == []                                      # both records read back


def test_a_cache_with_invalid_utf8_is_still_read(cache):
    good = json.dumps({"model": H, "sha": _sha(_prompt()), "reply": "no"}) + "\n"
    cache.write_bytes(good.encode() + b'{"model": "\xff\xfe broken')
    post = _fake(_resp("yes"))
    assert _label(post, cache)["k0"][H] == "no" and [c[1]["model"] for c in post.calls] == [G]


def test_the_cache_directory_is_created(tmp_path):
    path = tmp_path / "a" / "b" / "cache.jsonl"
    _label(_fake(_resp("yes")), path)
    assert len(_rows(path)) == 2


@pytest.mark.parametrize("reply", [_http(500), _http(400), {"choices": []}, _resp(""), _resp("  \n"), _resp(None),
                                   {"choices": [{"message": {}, "finish_reason": "stop"}]}, RuntimeError("boom"),
                                   _resp("partial", finish="error")],
                         ids=["500", "400", "no choices", "empty", "blank", "null content", "no content", "unknown",
                              "provider error"])
def test_a_reply_that_is_not_an_answer_is_never_cached(reply, cache):
    out = _label(_fake(reply), cache)
    assert out == {"k0": {H: None, G: None}} and _rows(cache) == []


def test_replies_paid_for_survive_a_cost_stop_and_the_rerun_fills_in_the_rest(cache):
    items = [_item(i) for i in range(4)]
    first = _label(_fake(_resp("yes", cost=0.25)), cache, items=items, cap=1.0)     # 4 of the 8 calls
    assert first["k0"] == {H: "yes", G: "yes"} and first["k3"] == {H: None, G: None} and len(_rows(cache)) == 4
    second = _fake(_resp("yes", cost=0.25))
    out = _label(second, cache, items=items, cap=10.0)
    assert len(second.calls) == 4 and out.cached == 4
    assert all(row == {H: "yes", G: "yes"} for row in out.values())


# ---------------------------------------------------------------- errors: None for that item and judge, never a crash

def test_one_failing_judge_leaves_the_other_labelled(cache, sleeps):
    post = _fake(lambda b: _http(500) if b["model"] == G else _resp("yes"))
    out = _label(post, cache)
    assert out == {"k0": {H: "yes", G: None}}
    assert [c[1]["model"] for c in post.calls].count(G) == 5 and sleeps == [3, 6, 12, 24]
    assert out.none == {H: {}, G: {"error:http_500": 1}}
    assert judges.agreement(out)["none"] == {H: 0, G: 1}


def test_one_failing_item_does_not_stop_the_batch(cache):
    def reply(body):
        return _http(500) if "reply number 2" in body["messages"][0]["content"] else _resp("yes")
    out = _label(_fake(reply), cache, items=[_item(i) for i in range(5)], workers=4, judges=(H,))
    assert [out[f"k{i}"][H] for i in range(5)] == ["yes", "yes", None, "yes", "yes"]
    assert out.none[H] == {"error:http_500": 1}


def test_retries_back_off_three_times_two_to_the_k(cache, sleeps):
    post = _fake(_http(503))
    out = _label(post, cache, judges=(H,))
    assert out == {"k0": {H: None}} and len(post.calls) == 5 and sleeps == [3, 6, 12, 24]


def test_retry_sleep_scales_the_backoff(cache, sleeps):
    _label(_fake(_http(503)), cache, judges=(H,), retry_sleep=0.5)
    assert sleeps == [0.5, 1.0, 2.0, 4.0]


def test_a_retry_that_succeeds_is_not_an_error(cache, sleeps):
    post = _script(_http(429), _http(502), _resp("yes"))
    out = _label(post, cache, judges=(H,))
    assert out == {"k0": {H: "yes"}} and len(post.calls) == 3 and sleeps == [3, 6] and out.none[H] == {}


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504, 529])
def test_retryable_http_codes(code, cache):
    post = _fake(_http(code))
    out = _label(post, cache, judges=(H,))
    assert out.none[H] == {f"error:http_{code}": 1} and len(post.calls) == 5


@pytest.mark.parametrize("code", [400, 401, 402, 403, 404, 422])
def test_other_http_errors_fail_at_once(code, cache, sleeps):
    post = _fake(_http(code))
    out = _label(post, cache, judges=(H,))
    assert out.none[H] == {f"error:http_{code}": 1} and len(post.calls) == 1 and sleeps == []


TRANSIENT = [
    (TimeoutError("read timed out"), "timeout"),
    (urllib.error.URLError(TimeoutError("connect timed out")), "timeout"),
    (urllib.error.URLError("name resolution failed"), "network"),
    (ConnectionResetError("reset by peer"), "network"),
    (json.JSONDecodeError("Expecting value", "<html>", 0), "bad_json"),
]


@pytest.mark.parametrize("exc, what", TRANSIENT, ids=[type(e).__name__ + "->" + w for e, w in TRANSIENT])
def test_timeouts_and_network_errors_are_retried(exc, what, cache, sleeps):
    post = _fake(exc)
    out = _label(post, cache, judges=(H,))
    assert out == {"k0": {H: None}} and out.none[H] == {f"error:{what}": 1}
    assert len(post.calls) == 5 and sleeps == [3, 6, 12, 24]


def test_a_bug_while_asking_is_that_items_error_not_the_batchs(cache, monkeypatch):
    real = judges._body

    def body(judge, prompt):
        if "number 1" in prompt:
            raise RuntimeError("bug in the request builder")
        return real(judge, prompt)
    monkeypatch.setattr(judges, "_body", body)
    out = _label(_fake(_resp("yes")), cache, items=[_item(i) for i in range(3)], judges=(H,))
    assert [out[f"k{i}"][H] for i in range(3)] == ["yes", None, "yes"]
    assert out.none[H] == {"error:RuntimeError": 1}


def test_an_unknown_exception_is_not_retried(cache):
    post = _fake(RuntimeError("boom"))
    out = _label(post, cache, judges=(H,))
    assert out == {"k0": {H: None}} and out.none[H] == {"error:RuntimeError": 1} and len(post.calls) == 1


@pytest.mark.parametrize("reply", [{"choices": []}, {"choices": None}, {}, {"error": {"code": 502, "message": "up"}}],
                         ids=["empty", "null", "bare", "error object"])
def test_no_choices_is_retried_then_an_error(reply, cache):
    post = _fake(reply)
    out = _label(post, cache, judges=(H,))
    assert out.none[H] == {"error:no_choices": 1} and len(post.calls) == 5


def test_a_provider_error_finish_is_retried_and_its_text_dropped(cache):
    post = _fake(_resp("yes", finish="error"))
    out = _label(post, cache, judges=(H,))
    assert out == {"k0": {H: None}} and out.none[H] == {"error:provider_error": 1} and len(post.calls) == 5


@pytest.mark.parametrize("reply", [
    "not json", ["a list"], None, {"choices": [None]}, {"choices": ["str"]}, {"choices": [{"finish_reason": "stop"}]},
    {"choices": [{"message": None, "finish_reason": "stop"}]}, {"choices": [{"message": {"content": 5}}]},
    {"choices": [{"message": {"content": ["parts"]}}]},
])
def test_a_malformed_reply_is_an_error_not_a_crash(reply, cache):
    out = _label(_fake(reply), cache, judges=(H,))
    assert out == {"k0": {H: None}}
    assert set(out.none[H]) <= {"error:bad_response", "error:no_choices"} and sum(out.none[H].values()) == 1


@pytest.mark.parametrize("reply", [_resp(""), _resp("  \n\t"), _resp(None), {"choices": [{"message": {}}]}],
                         ids=["empty", "blank", "null", "no content key"])
def test_empty_or_missing_content_is_none_and_not_retried(reply, cache):
    post = _fake(reply)
    out = _label(post, cache, judges=(H,))
    assert out == {"k0": {H: None}} and out.none[H] == {"empty": 1} and len(post.calls) == 1


def test_each_kind_of_problem_is_logged_once_per_judge(cache, capsys):
    def reply(body):
        return _http(500 if "number 0" in body["messages"][0]["content"] or "number 1" in body["messages"][0]["content"]
                     else 400)
    _label(_fake(reply), cache, items=[_item(i) for i in range(5)], judges=(H,))
    err = capsys.readouterr().err
    assert err.count("[judges]") == 2 and "http_500" in err and "http_400" in err and "err body" in err


def test_the_first_off_set_reply_of_a_judge_is_shown(cache, capsys):
    _label(_fake(_resp("perhaps, hard to say")), cache, items=[_item(i) for i in range(4)], judges=(H,))
    err = capsys.readouterr().err
    assert err.count("[judges]") == 1 and "off_set" in err and "perhaps, hard to say" in err


# ---------------------------------------------------------------- cost accounting

def test_spent_is_the_sum_of_usage_cost(cache):
    out = _label(_fake(_resp("yes", cost=0.25)), cache, items=[_item(i) for i in range(4)], cap=100)
    assert out.spent == 2.0 and out.calls == 8 and out.cost_missing == 0


_NO_COST = {"no usage": lambda r: r.pop("usage"), "usage null": lambda r: r.update(usage=None),
            "usage not a dict": lambda r: r.update(usage="free"), "empty usage": lambda r: r.update(usage={}),
            "cost null": lambda r: r["usage"].update(cost=None), "cost text": lambda r: r["usage"].update(cost="abc"),
            "cost nan": lambda r: r["usage"].update(cost=float("nan")),
            "cost inf": lambda r: r["usage"].update(cost=float("inf")),
            "cost -inf": lambda r: r["usage"].update(cost=float("-inf")),
            "cost negative": lambda r: r["usage"].update(cost=-0.5),
            "cost bool": lambda r: r["usage"].update(cost=True),
            "cost list": lambda r: r["usage"].update(cost=[0.1]),
            "cost huge": lambda r: r["usage"].update(cost=10 ** 400)}


@pytest.mark.parametrize("how", list(_NO_COST))
def test_a_missing_or_bad_cost_counts_as_zero_and_is_counted(how, cache, ledger):
    reply = _resp("yes")
    _NO_COST[how](reply)
    out = _label(_fake(reply), cache)
    assert out == {"k0": {H: "yes", G: "yes"}}                    # the answers are still used
    assert out.spent == 0.0 and out.cost_missing == 2 and out.calls == 2
    assert [r["usd"] for r in _rows(ledger)] == []                # nothing spent, nothing booked: no corrupt line


@pytest.mark.parametrize("cost, expected", [(0, 0.0), (3, 3.0), (0.25, 0.25), ("0.25", 0.25), (1e-9, 1e-9)])
def test_a_good_cost_is_charged(cost, expected, cache):
    out = _label(_fake(_resp("yes", cost=cost)), cache, judges=(H,))
    assert out.spent == expected and out.cost_missing == 0        # a free reply (cost 0) is not "missing"


def test_a_bad_cost_does_not_poison_the_total(cache):
    good, bad = _resp("yes", cost=0.25), _resp("yes", cost=float("nan"))
    out = _label(_script(good, bad, good), cache, items=[_item(i) for i in range(3)], judges=(H,), cap=100)
    assert out.spent == 0.5 and out.cost_missing == 1


def test_failed_attempts_that_report_a_cost_are_charged(cache):
    post = _fake({"choices": [], "usage": {"cost": 0.25}})
    out = _label(post, cache, judges=(H,), cap=100)
    assert out.none[H] == {"error:no_choices": 1}
    assert out.spent == 1.25 and out.cost_missing == 0 and out.calls == 5        # five attempts, each billed


def test_an_unusable_reply_without_a_cost_is_not_cost_missing(cache):
    out = _label(_fake({"choices": []}), cache, judges=(H,))
    assert out.spent == 0.0 and out.cost_missing == 0


# ---------------------------------------------------------------- the cap

def test_cost_stop_leaves_whole_items_labelled_and_the_rest_none(cache):
    post = _fake(_resp("yes", cost=0.25))
    out = _label(post, cache, items=[_item(i) for i in range(4)], cap=1.0)
    assert len(post.calls) == 4 and out.spent == 1.0                # stopped when spend REACHED the cap
    assert out["k0"] == out["k1"] == {H: "yes", G: "yes"}           # item-major: both judges, then the next item
    assert out["k2"] == out["k3"] == {H: None, G: None}
    assert out.none == {H: {"cost_stop": 2}, G: {"cost_stop": 2}}
    assert judges.agreement(out) == {"kappa": 1.0, "n": 2, "none": {H: 2, G: 2}, "per_class": {"yes": 1.0}}


def test_the_cap_is_reported_once_on_stderr(cache, capsys):
    _label(_fake(_resp("yes", cost=0.25)), cache, items=[_item(i) for i in range(6)], cap=0.5)
    err = capsys.readouterr().err
    assert err.count("reached the cap") == 1 and "[judges]" in err


def test_a_zero_cap_makes_no_call(cache):
    post = _fake(_resp("yes"))
    out = _label(post, cache, items=[_item(i) for i in range(3)], cap=0)
    assert post.calls == [] and out.none[H] == {"cost_stop": 3} and out.spent == 0.0


def test_cached_labels_are_free_even_at_a_zero_cap(cache):
    _label(_fake(_resp("yes")), cache, items=[_item(0), _item(1)])
    post = _fake(_resp("no"))
    out = _label(post, cache, items=[_item(0), _item(1), _item(2)], cap=0)
    assert post.calls == []
    assert out["k0"] == out["k1"] == {H: "yes", G: "yes"} and out["k2"] == {H: None, G: None}


def test_the_cap_is_checked_before_every_attempt_not_just_every_item(cache):
    post = _script({"choices": [], "usage": {"cost": 0.25}}, _resp("never reached"))
    out = _label(post, cache, judges=(H,), cap=0.25)
    assert len(post.calls) == 1 and out.none[H] == {"cost_stop": 1}


def test_a_failed_item_is_not_a_cost_stop_when_there_is_budget_left(cache):
    assert _label(_fake(_http(500)), cache, judges=(H,), cap=100).none[H] == {"error:http_500": 1}


def test_parallel_overshoot_is_bounded_by_the_workers(cache):
    post = _fake(_resp("yes", cost=0.25))
    out = _label(post, cache, items=[_item(i) for i in range(60)], judges=(H,), cap=1.0, workers=4)
    n = len(post.calls)
    assert 4 <= n <= 4 + 3                                          # 4 replies reach the cap; <= 3 more in flight
    assert out.spent == 0.25 * n
    assert sum(r[H] == "yes" for r in out.values()) == n and out.none[H] == {"cost_stop": 60 - n}


# ---------------------------------------------------------------- the ledger

def _by_note(rows):
    out = {}
    for r in rows:
        out[r["note"]] = out.get(r["note"], 0.0) + r["usd"]
    return out


def test_the_spend_is_booked_per_judge_under_its_note(cache, ledger):
    out = _label(_fake(lambda b: _resp("yes", cost=0.25 if b["model"] == H else 0.5)), cache,
                 items=[_item(i) for i in range(3)])
    rows = _rows(ledger)
    assert all(list(r) == ["mini", "date", "kind", "usd", "note"] and (r["mini"], r["kind"]) == ("T", "api") for r in rows)
    assert _by_note(rows) == {f"judges:{H}": 0.75, f"judges:{G}": 1.5}
    assert out.spent == 2.25 and util.ledger_total("T") == 2.25


def test_the_spend_is_on_disk_while_the_run_goes_on(cache, ledger):
    seen = []

    def reply(body):
        seen.append(util.ledger_total("T"))         # what the ledger holds when each call goes out
        return _resp("yes", cost=0.25)
    _label(_fake(reply), cache, items=[_item(i) for i in range(4)], judges=(H,))
    assert seen == [0.0, 0.25, 0.5, 0.75]           # each reply that brings a judge's unbooked spend to $0.25 is booked
    assert util.ledger_total("T") == 1.0            # at once, not only when label() ends


def test_small_costs_are_booked_at_the_end_as_one_line_per_judge(cache, ledger):
    _label(_fake(_resp("yes", cost=0.0625)), cache, items=[_item(i) for i in range(3)])
    rows = _rows(ledger)
    assert sorted((r["note"], r["usd"]) for r in rows) == [(f"judges:{H}", 0.1875), (f"judges:{G}", 0.1875)]


def test_a_failed_booking_for_one_judge_still_books_the_other_and_raises(cache, ledger, monkeypatch):
    real = util.ledger_add

    def flaky(mini, kind, usd, note=""):
        if note == f"judges:{H}":
            raise OSError("disk full")
        real(mini, kind, usd, note)
    monkeypatch.setattr(util, "ledger_add", flaky)
    with pytest.raises(OSError, match="disk full"):
        _label(_fake(_resp("yes", cost=0.0625)), cache)
    assert [(r["note"], r["usd"]) for r in _rows(ledger)] == [(f"judges:{G}", 0.0625)]
    assert len(_rows(cache)) == 2                   # both answers are kept: the rerun is free, and books nothing


def test_a_cached_rerun_books_nothing(cache, ledger):
    post = _fake(_resp("yes", cost=0.25))
    _label(post, cache)
    before = _rows(ledger)
    _label(post, cache)
    assert _rows(ledger) == before and len(before) == 2


def test_the_ledger_has_what_was_spent_before_a_cost_stop(cache, ledger):
    _label(_fake(_resp("yes", cost=0.25)), cache, items=[_item(i) for i in range(4)], cap=0.5)
    assert util.ledger_total("T") == 0.5


def test_the_spend_is_booked_even_if_the_batch_is_aborted(cache, ledger):
    n, billed = [0], []

    def reply(body):
        n[0] += 1
        if n[0] == 3:
            return KeyboardInterrupt()
        billed.append(0.25)              # the worker may start the next job before the main thread cancels the queue
        return _resp("yes", cost=0.25)
    with pytest.raises(KeyboardInterrupt):
        _label(_fake(reply), cache, items=[_item(i) for i in range(4)])
    assert len(billed) >= 2
    assert util.ledger_total("T") == sum(billed)                    # every reply that came back was booked ...
    assert len(_rows(cache)) == len(billed)                         # ... and kept for the rerun


# ---------------------------------------------------------------- the real transport: default_post and the key

def test_without_post_the_key_comes_from_load_key_and_default_post_is_used(cache, monkeypatch):
    _stub_frontier(monkeypatch, lambda: "sk-test")
    seen = []

    def fake_default_post(url, body, headers, timeout=180):
        seen.append((url, body, headers))
        return _resp("yes")
    monkeypatch.setattr(orbackend, "default_post", fake_default_post)
    out = judges.label([_item(0)], _render, YN, mini="T", cap=1.0, cache=cache, workers=1)
    assert out == {"k0": {H: "yes", G: "yes"}}
    assert [h for _, _, h in seen] == [{"authorization": "Bearer sk-test"}] * 2
    assert {u for u, _, _ in seen} == {orbackend.URL} and [b["model"] for _, b, _ in seen] == [H, G]


def test_an_injected_post_never_touches_the_key(cache, monkeypatch):
    def no_key():
        raise AssertionError("load_key must not be called when post is injected")
    _stub_frontier(monkeypatch, no_key)
    assert _label(_fake(_resp("yes")), cache) == {"k0": {H: "yes", G: "yes"}}


def test_a_fully_cached_run_needs_neither_post_nor_key(cache, monkeypatch):
    _label(_fake(_resp("yes")), cache)

    def no_key():
        raise AssertionError("load_key must not be called when every label is cached")
    _stub_frontier(monkeypatch, no_key)
    assert judges.label([_item(0)], _render, YN, mini="T", cap=1.0, cache=cache) == {"k0": {H: "yes", G: "yes"}}


def test_importing_judges_does_not_import_the_key_loader_or_patch_dns():
    code = ("import socket, sys; g = socket.getaddrinfo\n"
            "import persona_affect.common.judges\n"
            "assert 'persona.operative.frontier' not in sys.modules\n"
            "assert socket.getaddrinfo is g\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT), "PATH": ""})
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------- many threads

def test_parallel_updates_are_not_lost(cache):
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)                                     # make thread switches frequent
    try:
        out = _label(_fake(lambda b: _resp("yes", cost=0.25)), cache, items=[_item(i) for i in range(200)],
                     cap=1000, workers=16)
    finally:
        sys.setswitchinterval(old)
    assert out.spent == 100.0 and out.calls == 400 and out.none == {H: {}, G: {}}
    assert list(out) == [f"k{i}" for i in range(200)]
    assert all(row == {H: "yes", G: "yes"} for row in out.values()) and len(_rows(cache)) == 400
    assert util.ledger_total("T") == 100.0


def test_calls_really_run_in_parallel(cache):
    barrier = threading.Barrier(4, timeout=5)

    def reply(body):
        barrier.wait()                   # passes only when 4 calls are in flight at once; raises if they never are
        return _resp("yes")
    out = _label(_fake(reply), cache, items=[_item(i) for i in range(8)], judges=(H,), workers=4)
    assert out == {f"k{i}": {H: "yes"} for i in range(8)}


def test_spend_and_call_counts_are_only_written_under_the_lock(cache, monkeypatch):
    class Checked(dict):
        def __init__(self, run, *args):
            super().__init__(*args)
            self.run = run

        def __setitem__(self, key, value):
            if not self.run.lock.locked():
                raise AssertionError("spent written without the lock")
            super().__setitem__(key, value)

    class Guarded(judges._Run):
        """Raises if the spend or a counter is written while the run's lock is free."""
        def __setattr__(self, name, value):
            lock = self.__dict__.get("lock")
            if name in ("calls", "cost_missing") and lock is not None and not lock.locked():
                raise AssertionError(f"{name} written without the lock")
            super().__setattr__(name, Checked(self, value) if name == "spent" else value)
    monkeypatch.setattr(judges, "_Run", Guarded)
    out = _label(_fake(_resp("yes", cost=0.25)), cache, items=[_item(i) for i in range(3)], workers=4)
    assert out == {f"k{i}": {H: "yes", G: "yes"} for i in range(3)} and out.none == {H: {}, G: {}}
    assert out.spent == 1.5 and out.calls == 6


def test_the_record_of_what_was_logged_is_only_written_under_the_lock(cache, monkeypatch, capsys):
    class Checked(set):
        def __init__(self, run):
            super().__init__()
            self.run = run

        def add(self, item):
            if not self.run.lock.locked():
                raise AssertionError("logged written without the lock")
            super().add(item)

    class Guarded(judges._Run):
        def __setattr__(self, name, value):
            super().__setattr__(name, Checked(self) if name == "logged" else value)
    monkeypatch.setattr(judges, "_Run", Guarded)
    out = _label(_fake(_http(400)), cache, items=[_item(i) for i in range(6)], judges=(H,), workers=4)
    assert out.none[H] == {"error:http_400": 6}
    assert capsys.readouterr().err.count("[judges]") == 1                  # six failures, one line


def test_results_are_matched_to_their_items_under_parallel_calls(cache):
    def reply(body):                                                # each prompt gets its own number back
        return _resp(body["messages"][0]["content"].split("reply number ")[1].split("\n")[0])
    allowed = {str(i) for i in range(40)}
    out = _label(_fake(reply), cache, items=[_item(i) for i in range(40)], allowed=allowed, workers=8, judges=(H,))
    assert out == {f"k{i}": {H: str(i)} for i in range(40)}


# ---------------------------------------------------------------- agreement

def test_agreement_matches_the_vigilante_kappa():
    a = ["yes"] * 5 + ["no"] * 5
    b = ["yes", "yes", "yes", "yes", "no", "no", "no", "no", "no", "yes"]
    labels = {f"k{i}": {H: x, G: y} for i, (x, y) in enumerate(zip(a, b))}
    out = judges.agreement(labels)
    assert out["kappa"] == pytest.approx(0.6) and out["kappa"] == kappa(a, b)
    assert out == {"kappa": out["kappa"], "n": 10, "none": {H: 0, G: 0}, "per_class": out["per_class"]}


def test_none_labels_are_left_out_of_kappa_and_counted_per_judge():
    labels = {"a": {H: "yes", G: "yes"}, "b": {H: "no", G: "no"}, "c": {H: None, G: "yes"},
              "d": {H: "yes", G: None}, "e": {H: None, G: None}, "f": {H: "yes", G: "no"}}
    out = judges.agreement(labels)
    assert out["n"] == 3 and out["none"] == {H: 2, G: 2}
    assert out["kappa"] == kappa(["yes", "no", "yes"], ["yes", "no", "no"])


def test_a_row_without_a_judge_counts_as_none():
    out = judges.agreement({"a": {H: "yes"}, "b": {H: "yes", G: "yes"}, "c": {H: "no", G: "no"}})
    assert out["n"] == 2 and out["none"] == {H: 0, G: 1}


def test_agreement_with_nothing_to_compare_has_no_kappa_not_nan():
    assert judges.agreement({}) == {"kappa": None, "n": 0, "none": {H: 0, G: 0}, "per_class": {}}
    out = judges.agreement({"a": {H: None, G: "yes"}, "b": {H: "no", G: None}})
    assert out == {"kappa": None, "n": 0, "none": {H: 1, G: 1}, "per_class": {}}


def test_agreement_has_a_one_vs_rest_kappa_per_class():
    # DESIGN section 2 gates the judges at kappa >= 0.6 per class: low and high agree well here, mid does not
    a = ["low"] * 6 + ["mid"] * 3 + ["high"] * 6 + ["low"]
    b = ["low"] * 6 + ["low", "mid", "high"] + ["high"] * 6 + [None]
    labels = {f"k{i}": {H: x, G: y} for i, (x, y) in enumerate(zip(a, b))}
    out = judges.agreement(labels)
    a, b = a[:15], b[:15]                                           # the last item has no G label: left out
    assert out["n"] == 15 and set(out["per_class"]) == {"low", "mid", "high"}
    for c in ("low", "mid", "high"):
        assert out["per_class"][c] == kappa([x == c for x in a], [y == c for y in b])
    assert out["per_class"]["low"] >= 0.6 and out["per_class"]["high"] >= 0.6 and out["per_class"]["mid"] < 0.6
    json.dumps(out, allow_nan=False)                                # never a NaN


def test_per_class_covers_a_label_only_one_judge_gave():
    out = judges.agreement({"a": {H: "yes", G: "yes"}, "b": {H: "no", G: "yes"}, "c": {H: "yes", G: "yes"}})
    assert set(out["per_class"]) == {"yes", "no"}
    assert out["per_class"]["no"] == kappa([False, True, False], [False, False, False])


def test_agreement_reads_the_judges_it_is_given():
    labels = {"a": {"x": "u", "y": "u", H: "p"}, "b": {"x": "v", "y": "v", H: "p"}, "c": {"x": "u", "y": "v", H: "p"}}
    out = judges.agreement(labels, judges=("x", "y"))
    assert out["n"] == 3 and out["none"] == {"x": 0, "y": 0}
    assert out["kappa"] == kappa(["u", "v", "u"], ["u", "v", "v"])
    assert judges.agreement(labels, judges=("y", "x"))["kappa"] == out["kappa"]


@pytest.mark.parametrize("bad", [(H,), (), (H, G, "x"), (H, H)])
def test_agreement_compares_exactly_two_different_judges(bad):
    with pytest.raises(ValueError, match="judges"):
        judges.agreement({"a": {H: "yes", G: "yes"}}, judges=bad)


def test_agreement_refuses_the_transposed_shape_instead_of_reporting_n_0():
    with pytest.raises(ValueError, match="judge-major"):
        judges.agreement({H: {"a": "yes", "b": "no"}, G: {"a": "yes", "b": "no"}})


def test_agreement_on_a_real_run(cache):
    def reply(body):
        text = body["messages"][0]["content"]
        if body["model"] == H:
            return _resp("yes")
        return _resp("no" if "number 3" in text else "yes")
    out = _label(_fake(reply), cache, items=[_item(i) for i in range(4)])
    ag = judges.agreement(out)
    assert ag["n"] == 4 and ag["none"] == {H: 0, G: 0}
    assert ag["kappa"] == kappa(["yes"] * 4, ["yes", "yes", "yes", "no"])
