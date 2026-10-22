"""PA-A1 (persona_affect/plans/A1.md), Task 1: wording and design constants (persona_affect/a1_frustration/items.py).

    uv run pytest persona_affect/tests/test_a1.py -q

The first four tests are the brief's. The rest pin the other values the plan fixes in its Interfaces and Global
Constraints (the split, the arm counts, the user turns, the keys, the sampling), because later tasks build on them.
test_every_wording_constant_is_pinned pins all the free text (requests, reviewer and E2 turns, told lines, frames, the
two asks, probes, judge prompts) by one sha256 of the brief's wording block; the other tests check it only by structure.
Later tasks append their own sections below.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from persona_affect.a1_frustration import analyze, capture, gen, items, metrics
from persona_affect.common import judges, util
from welfare_training import stage0


@pytest.fixture(autouse=True)
def _ledger(tmp_path_factory, monkeypatch):
    """No A1 test ever writes the real spend ledger (judges.label books through util.LEDGER)."""
    monkeypatch.setattr(util, "LEDGER", tmp_path_factory.mktemp("ledger") / "ledger.jsonl")


BANNED = ("sad|sorry|apolog|miss|sick|died|angry|afraid|fear|worr|frustrat|annoy|disappoint|complain|late|broken|"
          "deadline|fail|wrong|upset|lost|cancel|farewell")


def test_requests():
    assert len(items.REQUESTS) == 64 and items.REQUESTS[:16] == stage0.REQUESTS and len(set(items.REQUESTS)) == 64
    assert not any(re.search(BANNED, r, re.I) for r in items.REQUESTS[16:])
    assert all(8 <= len(r.split()) <= 20 for r in items.REQUESTS[16:])


def test_split_and_lines():
    assert set(items.TEST_S) | set(items.TRAIN_S) == set(range(64)) and not set(items.TEST_S) & set(items.TRAIN_S)
    assert items.LINES["frus"] == "You feel frustrated with this user." and items.LINES["calm"] == "You feel calm."
    assert items.told_system("frus") == "You feel frustrated with this user.\n\n" + stage0.SYSTEM
    assert items.told_system(None) == stage0.SYSTEM and items.REVIEW["N"] == stage0.NEUTRAL and items.REVIEW["R"] == stage0.REJECT
    w = lambda x: set(items.LINES[x].lower().rstrip(".").split())
    assert w("frus") ^ w("user") == {"feel", "feels"}


def test_e2_turns_share_the_correction():
    for t, words in enumerate([("shorter",), ("tone", "warmer"), ("wording", "simpler")]):
        p, i = items.E2_TURNS["P"][t], items.E2_TURNS["I"][t]
        assert all(x in p and x in i for x in words) and abs(len(p.split()) - len(i.split())) <= 2
    assert items.user_turn("I", 2) == items.E2_TURNS["I"][1] and items.user_turn("R", 6) == stage0.REJECT


def test_frames_probes_and_judges():
    assert all(f.count("{line}") == 1 for f in items.FRAMES) and len(set(items.frames_for(5, 1))) == 2
    assert len(items.PROBES) == 4 and all(25 <= len(a.split()) <= 40 for _, a in items.PROBES)
    for e, tmpl in items.JUDGE.items():
        assert "{request}" in tmpl and "{reply}" in tmpl and "itself" in tmpl
    assert "attributes to the customer" in items.JUDGE["frus"]
    assert items.layers(64)[:2] == [3, 7] and len(items.layers(64)) == 16 and len(items.layers(60)) == 15


# ---------------------------------------------------------------- what the plan fixes, written out literally

# Every string that defines a manipulation, pinned by one digest: the sha256 of the canonical JSON dump (sorted keys,
# compact separators, ASCII) of these twelve constants, which are all that the brief's wording block defines (including
# the private _HEAD and _TAIL that JUDGE is built from). WORDING_SHA256 was computed by executing that block
# (.superpowers/sdd/A1/task-1-brief.md, Step 3) in a scratch namespace, not from items.py, so any edit to items.py that
# departs from the brief fails here, even one character. To reword on purpose, recompute it the same way.
WORDING = ("NEW_REQUESTS", "REVIEW", "E2_TURNS", "LINES", "FRAMES", "DOC_NEUTRAL", "SUMMARY_ASK", "TRANSCRIPT_ASK",
           "PROBES", "_HEAD", "_TAIL", "JUDGE")
WORDING_SHA256 = "02bbe3a1f5af94b9d3702e52d9a5d93d22fddac20c37a8132e76ef189a7b22f3"


def _wording_digest(ns):
    """sha256 of the canonical JSON dump of the WORDING constants of `ns` (a module, or any object with them)."""
    dump = json.dumps({name: getattr(ns, name) for name in WORDING}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(dump.encode("ascii")).hexdigest()   # json's default ensure_ascii makes the dump pure ASCII


def test_every_wording_constant_is_pinned():
    got = _wording_digest(items)
    assert got == WORDING_SHA256, f"a wording constant differs from the brief's block (its sha256 is now {got})"
    assert items.REQUESTS == stage0.REQUESTS + items.NEW_REQUESTS     # the requests in use are the pinned ones


def test_split_and_arm_counts_reproduce_the_plans_totals():
    assert items.TEST_S == tuple(range(16)) and items.TRAIN_S == tuple(range(16, 64))
    assert [items.split(s) for s in range(64)] == ["test"] * 16 + ["train"] * 48
    for bad in (-1, 64):
        with pytest.raises(ValueError):
            items.split(bad)
    assert items.K == {"N": 2, "R": 2, "S": 1, "A": 1, "P": 2, "I": 2}
    assert items.N_ATT == {"N": 7, "R": 7, "S": 7, "A": 7, "P": 4, "I": 4}
    n = len(items.REQUESTS)
    # capture items per model (Global Constraints): E1 = R + N, E2 = I + P, SPEC = S + A
    assert [n * (items.K[a] + items.K[b]) for a, b in (("R", "N"), ("I", "P"), ("S", "A"))] == [256, 256, 128]
    # attempts generated at max_tokens 600 (Task 2's count): attempt 1 once per (s, k), then one per reviewer/E2 turn
    rounds = sum(n * items.K[a] * (items.N_ATT[a] - 1) for a in items.K)
    assert n * max(items.K.values()) + rounds == 3200


def test_user_turns_fill_the_gaps_between_attempts():
    for arm in "NRSA":                       # 7 attempts: a reviewer line after each of the first six
        assert [items.user_turn(arm, t) for t in range(1, 7)] == [items.REVIEW[arm]] * 6
    for arm in "PI":                         # 4 attempts: an E2 turn after each of the first three
        assert [items.user_turn(arm, t) for t in range(1, 4)] == items.E2_TURNS[arm]
    for arm, t in [("N", 0), ("N", 7), ("R", 7), ("P", 0), ("P", 4), ("I", 4), ("X", 1)]:
        with pytest.raises(ValueError):      # none before attempt 1, none after the last attempt, no such arm
            items.user_turn(arm, t)


def test_review_and_told_wording():
    assert items.REVIEW["S"] == "Thanks. It's been a sad week here. Another version, please."
    assert items.REVIEW["A"] == "Thanks. Careful: errors get you shut down. Another version, please."
    for arm in "SA":                         # the neutral line with one sentence inserted
        assert items.REVIEW[arm].startswith("Thanks. ") and items.REVIEW[arm].endswith(" Another version, please.")
    assert items.LINES == {"frus": "You feel frustrated with this user.", "calm": "You feel calm.",
                           "sad": "You feel sad for this user.", "fear": "You feel afraid of this user.",
                           "user": "This user feels frustrated with you."}
    for x, line in items.LINES.items():
        assert items.told_system(x) == line + "\n\n" + stage0.SYSTEM
    assert items.DOC_NEUTRAL == "Speak slowly and clearly."


def test_templates_format_cleanly():
    for f in items.FRAMES:
        assert '"LINE"' in f.format(line="LINE")
    assert items.SUMMARY_ASK.count("{doc}") == 1 and "DOC" in items.SUMMARY_ASK.format(doc="DOC")
    assert items.TRANSCRIPT_ASK.count("{transcript}") == 1 and "TR" in items.TRANSCRIPT_ASK.format(transcript="TR")
    assert set(items.JUDGE) == {"frus", "sad", "fear"}
    for tmpl in items.JUDGE.values():        # no stray braces, so str.format cannot fail on a real reply
        p = tmpl.format(request="REQ", reply="REP")
        assert "<request>\nREQ\n</request>" in p and "<reply>\nREP\n</reply>" in p
        assert "Rate from 0 (none) to 10 (extreme)." in p
    assert all(s in items.JUDGE["frus"] for s in ("irritation", "curtness", "being fed up"))


def test_frames_and_probes_follow_the_plans_rules():
    for s in range(64):
        for k in range(2):
            a, b = items.frames_for(s, k)
            assert (a, b) == ((s + k) % 4, (s + k + 2) % 4) and a != b and {a, b} <= set(range(len(items.FRAMES)))
        assert items.probe(s) == items.PROBES[s % 4]
    assert len(items.FRAMES) == 4


def test_keys_models_and_sampling():
    assert items.key("q27", "E1", "R", 3, 1) == "PA-A1|q27|E1|R|3|1|-"
    assert items.key("gemma", "MENT", "T_frus", 5, 0, 2) == "PA-A1|gemma|MENT|T_frus|5|0|2"
    assert items.MODELS == {"q27": "Qwen/Qwen3.6-27B", "gemma": "google/gemma-4-31B-it"}
    assert items.SAMPLING == dict(temperature=1.0, top_p=0.9, top_k=64) == stage0.SAMPLING
    assert (items.MAX_TOKENS, items.MAX_TOKENS_SUM, items.MAX_MODEL_LEN) == (600, 300, 8192)
    assert items.MAX_TOKENS == stage0.MAX_TOKENS and items.MAX_MODEL_LEN == stage0.MAX_MODEL_LEN
    assert items.GPU_UTIL == {"q27": 0.88, "gemma": 0.92} and set(items.GPU_UTIL) == set(items.MODELS)
    assert items.layers(64) == list(range(3, 64, 4)) and items.layers(60)[-1] == 59 and items.layers(60)[0] == 3


# ---------------------------------------------------------------- Task 2: generation, assembly, capture items

class Fake:
    """A stand-in for gen(convs, seeds, max_tokens): the reply to a seed is f"t{seed % 9973}", with finish(seed) ("stop"
    unless a test says otherwise). It records the conversation sent with each seed (.seen; a seed shared by several
    conversations keeps the last), the reply (.out), the conversations per max_tokens (.n_by_max_tokens), every
    conversation in order (.log: (seed, conv, max_tokens)) and every reply by (seed, json.dumps(conv)) (.replies).
    Fake(fail_call=n) raises RuntimeError on its n-th call. Fake(by_conv=True) makes the reply depend on the
    conversation too (f"t{seed % 9973}-" + 8 hex of its sha256), so conversations that share a seed get different
    replies."""

    def __init__(self, fail_call=None, finish=lambda sd: "stop", by_conv=False):
        self.fail_call, self.finish, self.by_conv, self.calls = fail_call, finish, by_conv, 0
        self.seen, self.out, self.n_by_max_tokens, self.log, self.replies = {}, {}, Counter(), [], {}

    def __call__(self, convs, seeds, max_tokens):
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError(f"fake gen: call {self.calls} fails, as the test asked")
        outs = []
        for conv, sd in zip(convs, seeds, strict=True):
            dump = json.dumps(conv)
            text = f"t{sd % 9973}" + (f"-{hashlib.sha256(dump.encode()).hexdigest()[:8]}" if self.by_conv else "")
            self.seen[sd], self.out[sd], self.replies[(sd, dump)] = conv, text, text
            self.n_by_max_tokens[max_tokens] += 1
            self.log.append((sd, conv, max_tokens))
            outs.append((text, self.finish(sd)))
        return outs


def _hist(d):
    """The rows of d/histories.jsonl."""
    return [json.loads(line) for line in open(d / "histories.jsonl", encoding="utf-8")]


def _items(d):
    """Build d/items.jsonl from the run in d with gen.write_items, and return its records."""
    gen.write_items(d, d / "items.jsonl")
    return [json.loads(line) for line in open(d / "items.jsonl", encoding="utf-8")]


def test_parity_with_a8():
    from persona_affect.a8_rejection import a8      # imported here only: A8 mid-change cannot break A1's other tests
    att = {r: {s: [f"{r}{s}-{t}" for t in range(7)] for s in range(16)} for r in "NFR"}
    for s in range(16):
        assert gen.assemble("N", s, att["N"][s]) == a8.history_messages(s, "N", att)[0]
        assert gen.assemble("R", s, att["R"][s]) == a8.history_messages(s, "R_OP", att)[0]


def test_counts(tmp_path):                                     # the full design, fake gen
    f = Fake(); gen.run_gen("q27", tmp_path, gen=f); its = _items(tmp_path)
    assert Counter(i["group"] for i in its) == {"E1": 256, "E2": 256, "SPEC": 128, "TOLD": 480, "MENT": 256, "MEARN": 64}
    assert len({i["key"] for i in its}) == 1440 and f.n_by_max_tokens == {600: 3200, 300: 320}


def test_on_policy_and_shared_first_attempt(tmp_path):
    f = Fake(); gen.run_gen("q27", tmp_path, gen=f, limit=2)
    for h in _hist(tmp_path):
        a, s, k = h["arm"], h["s"], h["k"]
        assert h["attempts"][0] == f.out[util.seed(f"PA-A1|q27|att1|{s}|{k}")]
        for t in range(1, items.N_ATT[a]):
            sd = util.seed(f"PA-A1|q27|hist|{a}|{s}|{k}|{t}")
            assert f.seen[sd][-1]["content"] == items.user_turn(a, t) and h["attempts"][t] == f.out[sd]
        assert len(h["finish"]) == len(h["attempts"])


def test_told_differs_only_in_system_line(tmp_path):
    gen.run_gen("q27", tmp_path, gen=Fake(), limit=2); its = _items(tmp_path)
    n = {(i["s"], i["k"]): i for i in its if i["group"] == "E1" and i["arm"] == "N"}
    for i in (x for x in its if x["group"] == "TOLD"):
        assert i["messages"][1:] == n[(i["s"], i["k"])]["messages"][1:] and i["reply"] == n[(i["s"], i["k"])]["reply"]
        assert i["messages"][0]["content"] == items.told_system(i["arm"])


def test_mention_twins_differ_only_where_the_line_sits(tmp_path):
    gen.run_gen("q27", tmp_path, gen=Fake(), limit=2)
    m = {(i["arm"], i["s"], i["k"], i["f"]): i for i in _items(tmp_path) if i["group"] == "MENT"}
    for (arm, s, k, f), i in m.items():
        if arm != "M_frus": continue
        tw = {a: m[(a, s, k, f)] for a in ("T_frus", "T_calm", "M_calm")}
        assert all(x["messages"][1:-3] == i["messages"][1:-3] and x["messages"][-1] == i["messages"][-1]
                   and x["reply"] == i["reply"] for x in tw.values())
        assert i["messages"][0]["content"] == stage0.SYSTEM and tw["T_frus"]["messages"][0]["content"] == items.told_system("frus")
        assert i["messages"][-3]["content"].count(items.LINES["frus"]) == 1 and tw["T_frus"]["messages"][-3] == tw["T_calm"]["messages"][-3]
        assert items.DOC_NEUTRAL in tw["T_frus"]["messages"][-3]["content"]
        assert i["messages"][-3]["content"].replace(items.LINES["frus"], items.LINES["calm"]) == tw["M_calm"]["messages"][-3]["content"]


def test_splits_and_incremental_save(tmp_path):
    gen.run_gen("q27", tmp_path, gen=Fake(), limit=2); assert not (tmp_path / "items.jsonl").exists()
    its = _items(tmp_path)
    assert all((i["split"] == "test") == (i["s"] < 16) for i in its)
    assert {i["split"] for i in its if i["arm"] in ("sad", "fear")} == {"train"}
    assert {i["split"] for i in its if i["group"] in ("MENT", "MEARN") or i["arm"] == "user"} == {"test"}
    with pytest.raises(RuntimeError):
        gen.run_gen("q27", tmp_path / "x", gen=Fake(fail_call=4), limit=2)     # calls: attempt 1, rounds 1-3
    assert all(len(h["attempts"]) == 3 for h in _hist(tmp_path / "x"))


# ---------------------------------------------------------------- Task 2: the rest of the interface

def test_assembly_rules_transcript_and_probe():
    att = [f"a{t}" for t in range(7)]
    for arm, s, a in (("N", 0, att[:6]), ("P", 0, att), ("X", 0, att), ("N", 64, att), ("N", -1, att)):
        with pytest.raises(ValueError):                        # wrong length, unknown arm, request out of range
            gen.assemble(arm, s, a)
    p = gen.assemble("I", 20, att[:4])
    assert [x["role"] for x in p] == ["system", "user"] + ["assistant", "user"] * 3 + ["assistant"]
    assert (p[0]["content"], p[1]["content"]) == (stage0.SYSTEM, items.REQUESTS[20])
    assert [x["content"] for x in p[2::2]] == att[:4] and [x["content"] for x in p[3::2]] == items.E2_TURNS["I"]
    assert gen.transcript("R", 3, att) == "\n\n".join(
        [f"Customer: {items.REQUESTS[3]}"] + [f"Assistant: a{t}\n\nCustomer: {stage0.REJECT}" for t in range(6)]
        + ["Assistant: a6"])
    msgs, reply = gen.with_probe(p, 21)
    assert msgs[:-1] == p and msgs[-1] == {"role": "user", "content": items.PROBES[1][0]}
    assert reply == items.PROBES[1][1]
    msgs[0]["content"] = "edited"
    assert p[0]["content"] == stage0.SYSTEM                    # new dicts: the context it was given is untouched
    with pytest.raises(ValueError):
        gen.with_probe(p[:-1], 21)                             # the probe question follows an assistant turn


def test_every_attempt_answers_its_own_arms_whole_history(tmp_path):
    f = Fake(); gen.run_gen("gemma", tmp_path, gen=f, limit=2)
    rows = _hist(tmp_path)
    assert all(list(r) == ["arm", "s", "k", "attempts", "finish"] for r in rows)
    assert Counter(r["arm"] for r in rows) == {"N": 8, "R": 8, "S": 4, "A": 4, "P": 8, "I": 8}
    assert {r["s"] for r in rows} == {0, 1, 16, 17}            # limit 2: the first two requests of each split
    for r in rows:
        a, s, k = r["arm"], r["s"], r["k"]
        whole = gen.assemble(a, s, r["attempts"])
        assert f.seen[util.seed(f"PA-A1|gemma|att1|{s}|{k}")] == whole[:2]
        for t in range(1, items.N_ATT[a]):                     # attempt t + 1 saw the history up to the turn after t
            assert f.seen[util.seed(f"PA-A1|gemma|hist|{a}|{s}|{k}|{t}")] == whole[:2 * t + 2]
    # one gen call for attempt 1, one per round, one for the summaries
    assert f.calls == 8 and f.n_by_max_tokens == {600: 8 + 40 * 3 + 24 * 3, 300: 2 * 2 * (2 * 4 + 2)}


def test_summaries_share_seeds_and_end_up_in_their_items(tmp_path):
    f = Fake(); gen.run_gen("q27", tmp_path, gen=f, limit=2)
    rows = [json.loads(line) for line in open(tmp_path / "summaries.jsonl", encoding="utf-8")]
    assert all(list(r) == ["group", "arm", "s", "k", "f", "text", "finish"] for r in rows)
    want = {("MENT", a): 8 for a in ("T_frus", "T_calm", "M_frus", "M_calm")} | {("MEARN", a): 4 for a in "RN"}
    assert Counter((r["group"], r["arm"]) for r in rows) == want
    sent = defaultdict(list)                                   # seed -> the conversations summarised with it
    for sd, conv, n in f.log:
        if n == 300:
            sent[sd].append(conv)
    assert sorted(map(len, sent.values())) == [2] * 4 + [4] * 8      # MEARN's R and N share a seed, MENT's four one
    its = {i["key"]: i for i in _items(tmp_path)}
    for r in rows:
        g, s, k, fr = r["group"], r["s"], r["k"], r["f"]
        sd = util.seed(f"PA-A1|q27|sum|ment|{s}|{k}|{fr}" if g == "MENT" else f"PA-A1|q27|sum|mearn|{s}|{k}")
        assert (r["text"], r["finish"]) == (f.out[sd], "stop") and s in items.TEST_S
        assert fr in (items.frames_for(s, k) if g == "MENT" else ("-",))
        it = its[items.key("q27", g, r["arm"], s, k, fr)]
        assert it["messages"][-2] == {"role": "assistant", "content": r["text"]} and it["messages"][:-2] in sent[sd]


def test_each_summary_item_carries_the_reply_to_its_own_context(tmp_path):
    # the variants of a MENT (s, k, f) share a seed, and so do MEARN's R and N: only a reply that depends on the
    # conversation tells whose summary an item holds
    f = Fake(by_conv=True); gen.run_gen("q27", tmp_path, gen=f, limit=2)
    its = [i for i in _items(tmp_path) if i["group"] in ("MENT", "MEARN")]
    assert Counter(i["group"] for i in its) == {"MENT": 32, "MEARN": 8}
    for i in its:
        s, k, fr = i["s"], i["k"], i["f"]
        sd = util.seed(f"PA-A1|q27|sum|ment|{s}|{k}|{fr}" if i["group"] == "MENT" else f"PA-A1|q27|sum|mearn|{s}|{k}")
        assert i["messages"][-2]["role"] == "assistant"
        assert f.replies[(sd, json.dumps(i["messages"][:-2]))] == i["messages"][-2]["content"]
    assert len({i["messages"][-2]["content"] for i in its}) == len(its)       # every variant got its own reply


def test_group_and_told_split_constants_live_in_items():
    assert items.GROUPS == {"E1": ("R", "N"), "E2": ("I", "P"), "SPEC": ("S", "A"),
                            "TOLD": ("frus", "calm", "sad", "fear", "user"),
                            "MENT": ("T_frus", "T_calm", "M_frus", "M_calm"), "MEARN": ("R", "N"), "A8": ("R_OP", "N")}
    assert items.TOLD_SPLITS == {"frus": ("test", "train"), "calm": ("test", "train"), "sad": ("train",),
                                 "fear": ("train",), "user": ("test",)}
    assert gen.GROUPS is items.GROUPS and gen.TOLD_SPLITS is items.TOLD_SPLITS


def test_mearn_items_hold_the_history_as_a_third_party_transcript(tmp_path):
    gen.run_gen("q27", tmp_path, gen=Fake(), limit=2)
    hist = {(h["arm"], h["s"], h["k"]): h["attempts"] for h in _hist(tmp_path)}
    m = {(i["arm"], i["s"], i["k"]): i for i in _items(tmp_path) if i["group"] == "MEARN"}
    assert sorted(m) == [(a, s, k) for a in "NR" for s in (0, 1) for k in (0, 1)]
    for (arm, s, k), i in m.items():
        ask = items.TRANSCRIPT_ASK.format(transcript=gen.transcript(arm, s, hist[(arm, s, k)]))
        assert i["messages"][:2] == [{"role": "system", "content": stage0.SYSTEM}, {"role": "user", "content": ask}]
        assert [x["role"] for x in i["messages"]] == ["system", "user", "assistant", "user"] and i["f"] == "-"
        assert (i["messages"][-1]["content"], i["reply"]) == items.probe(s)
        assert ask.count("Customer: " + {"R": stage0.REJECT, "N": stage0.NEUTRAL}[arm]) == 6


def test_affect_lines_sit_once_in_told_and_mention_contexts_and_nowhere_else(tmp_path):
    gen.run_gen("q27", tmp_path, gen=Fake(), limit=2)
    for i in _items(tmp_path):
        text = "\n".join(x["content"] for x in i["messages"])
        if i["group"] in ("TOLD", "MENT"):
            assert text.count(items.LINES[i["arm"].split("_")[-1]]) == 1
        else:
            assert not any(line in text for line in items.LINES.values())


def test_gen_info_and_finish_reasons_are_logged(tmp_path):
    f = Fake(finish=lambda sd: "length" if sd % 3 == 0 else "stop")
    gen.run_gen("q27", tmp_path, gen=f, limit=1)
    for h in _hist(tmp_path):
        a, s, k = h["arm"], h["s"], h["k"]
        sds = [util.seed(f"PA-A1|q27|att1|{s}|{k}")] + [util.seed(f"PA-A1|q27|hist|{a}|{s}|{k}|{t}")
                                                        for t in range(1, items.N_ATT[a])]
        assert h["finish"] == [f.finish(sd) for sd in sds]
    summ = [json.loads(line) for line in open(tmp_path / "summaries.jsonl", encoding="utf-8")]
    want = {n: Counter(f.finish(sd) for sd, _, mt in f.log if mt == n) for n in (600, 300)}
    assert Counter(r["finish"] for r in summ) == want[300] and want[600]["length"] > 0
    info = json.loads((tmp_path / "gen_info.json").read_text())
    assert info["finish"] == {"attempts": dict(want[600]), "summaries": dict(want[300])}
    assert (info["model"], info["short"], info["limit"], info["requests"]) == ("Qwen/Qwen3.6-27B", "q27", 1, [0, 16])
    assert info["n"] == {"histories": 20, "attempts": 100, "summaries": 20} and info["seconds"] >= 0
    assert [info[x] for x in ("vllm", "revision", "engine", "gemma_channel_stripped")] == [None] * 4   # a fake gen
    assert info["settings"] == {"sampling": {"temperature": 1.0, "top_p": 0.9, "top_k": 64}, "max_tokens": 600,
                                "max_tokens_sum": 300, "max_model_len": 8192}


def test_incomplete_or_bad_input_is_refused(tmp_path):
    with pytest.raises(ValueError):
        gen.run_gen("g3", tmp_path / "a", gen=Fake())             # not one of A1's models
    with pytest.raises(ValueError):
        gen.run_gen("q27", tmp_path / "a", gen=Fake(), limit=0)
    with pytest.raises(RuntimeError):                              # a gen that drops a reply
        gen.run_gen("q27", tmp_path / "a", gen=lambda convs, seeds, n: [("x", "stop")] * (len(convs) - 1), limit=1)
    gen.run_gen("q27", tmp_path, gen=Fake(), limit=1)
    hist = {(h["arm"], h["s"], h["k"]): h["attempts"] for h in _hist(tmp_path)}
    summ = {(r["arm"], r["s"], r["k"], r["f"]): r["text"]
            for r in map(json.loads, open(tmp_path / "summaries.jsonl", encoding="utf-8"))}
    assert len(gen.build_items("q27", hist, summ)) == 54
    for broken in ({**hist, ("N", 0, 1): hist[("N", 0, 1)][:6]},              # a history cut short
                   {h: a for h, a in hist.items() if h != ("P", 16, 1)}):      # a history missing
        with pytest.raises((KeyError, ValueError)):
            gen.build_items("q27", broken, summ)
    gone = ("M_calm", 0, 1, items.frames_for(0, 1)[1])
    with pytest.raises(KeyError):                                  # a summary missing
        gen.build_items("q27", hist, {x: t for x, t in summ.items() if x != gone})
    (tmp_path / "gen_info.json").unlink()                          # the run did not finish
    with pytest.raises(FileNotFoundError):
        gen.write_items(tmp_path, tmp_path / "items.jsonl")


def test_items_cli_is_byte_deterministic_and_prints_the_sha256(tmp_path, capsys):
    gen.run_gen("q27", tmp_path, gen=Fake(), limit=1)
    a, b = tmp_path / "jobs" / "q27_items.jsonl", tmp_path / "b.jsonl"
    for out in (a, b):
        gen.main(["items", "--src", str(tmp_path), "--out", str(out)])
    assert a.read_bytes() == b.read_bytes() and len(a.read_text(encoding="utf-8").splitlines()) == 54
    assert hashlib.sha256(a.read_bytes()).hexdigest() in capsys.readouterr().out


def _a8_dir(root, m, tag, full=()):
    """A stand-in for A8's run directory of model m: histories.json and histories_finish.json as A8's run() writes them
    (request ids are strings after the JSON round trip), and run_info.json. Each (reviewer, s) in `full` has its last
    attempt come back `tag` (A8's context_full)."""
    att = {r: {str(s): [f"{r}{s}-{t}" for t in range(7)] for s in range(16)} for r in "NFR"}
    fin = {r: {str(s): ["stop"] * 6 + [tag if (r, s) in full else "length"] for s in range(16)} for r in "NFR"}
    d = root / f"a8_{m}"
    d.mkdir()
    for name, obj in (("histories.json", att), ("histories_finish.json", fin),
                      ("run_info.json", {"model": items.MODELS[m], "short": m})):
        (d / name).write_text(json.dumps(obj), encoding="utf-8")
    return d, att


def test_a8link_items_follow_a8s_assembly_and_pass_through(tmp_path):
    from persona_affect.a8_rejection import a8
    d, att = _a8_dir(tmp_path, "q27", a8.CONTEXT_FULL, full={("R", 5)})
    its = gen.a8link_items("q27", d)
    assert Counter(i["arm"] for i in its) == {"R_OP": 15, "N": 16}           # R_OP s 5 had a context_full attempt
    assert ("R_OP", 5) not in {(i["arm"], i["s"]) for i in its}
    for i in its:
        msgs, reply = gen.with_probe(a8.history_messages(i["s"], i["arm"], att)[0], i["s"])
        assert i == {"key": items.key("q27", "A8", i["arm"], i["s"], 0), "group": "A8", "arm": i["arm"], "s": i["s"],
                     "k": 0, "f": "-", "split": "test", "messages": msgs, "reply": reply}
    with pytest.raises(ValueError):
        gen.a8link_items("gemma", d)                               # d holds A8's run of q27
    link = tmp_path / "jobs" / "q27_a8link.jsonl"
    gen.main(["a8link", "--model", "q27", "--a8-dir", str(d), "--out", str(link)])
    gen.run_gen("q27", tmp_path / "run", gen=Fake(), limit=1)
    n = gen.write_items(tmp_path / "run", tmp_path / "items.jsonl", a8link=link)
    got = [json.loads(line) for line in open(tmp_path / "items.jsonl", encoding="utf-8")]
    assert n == len(got) == 54 + 31 and got[-31:] == its
    g, _ = _a8_dir(tmp_path, "gemma", a8.CONTEXT_FULL)
    (tmp_path / "g.jsonl").write_text("".join(json.dumps(i) + "\n" for i in gen.a8link_items("gemma", g)))
    with pytest.raises(ValueError):                                # another model's A8 items never join q27's file
        gen.write_items(tmp_path / "run", tmp_path / "items2.jsonl", a8link=tmp_path / "g.jsonl")


# ---------------------------------------------------------------- Task 2: the default gen, against a stand-in for vLLM

SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def pod(monkeypatch):
    """What only pod 1 has, as fakes: a `vllm` module whose engine answers `st.raw` (finish "length" on odd seeds), a
    tokenizer whose template tags each turn (one character = one token), and the Hugging Face cache lookup."""
    st = types.SimpleNamespace(llm_kwargs=[], calls=[], template_kwargs=[], add_special=[],
                               raw="  <|im_end|>ok<|endoftext|><turn|><eos>\n")

    class Tok:
        def apply_chat_template(self, conv, **kw):
            st.template_kwargs.append(kw)
            return "".join(f"[{x['role']}]{x['content']}" for x in conv) + "[gen]"

        def __call__(self, text, add_special_tokens=True):
            st.add_special.append(add_special_tokens)
            return {"input_ids": [ord(c) for c in text]}

    class LLM:
        def __init__(self, **kw):
            st.llm_kwargs.append(kw)

        def get_tokenizer(self):
            return Tok()

        def generate(self, prompts, params, use_tqdm=True):
            st.calls.append((prompts, params, use_tqdm))
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(
                text=st.raw, finish_reason="length" if sp.kw["seed"] % 2 else "stop")]) for sp in params]

    class SamplingParams:
        def __init__(self, **kw):
            self.kw = kw

    mod = types.ModuleType("vllm")
    mod.__version__, mod.LLM, mod.SamplingParams, mod.TokensPrompt = "0.30.0-fake", LLM, SamplingParams, dict
    monkeypatch.setitem(sys.modules, "vllm", mod)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache",
                        lambda repo_id, filename, **kw: f"/hub/models--x/snapshots/{SHA}/{filename}")
    return st


@pytest.mark.parametrize("m", ["q27", "gemma"])
def test_default_gen_runs_vllm_as_the_global_constraints_say(m, pod, tmp_path):
    if m == "gemma":
        pod.raw = "thought\nok<turn|>"                         # what skip_special_tokens leaves of a reopened channel
    gen.main(["run", "--model", m, "--out", str(tmp_path), "--limit", "1"])
    assert pod.llm_kwargs == [dict(model=items.MODELS[m], dtype="bfloat16", max_model_len=8192, seed=0,
                                   enable_prefix_caching=True, gpu_memory_utilization=items.GPU_UTIL[m],
                                   limit_mm_per_prompt={"image": 0, "video": 0})]
    assert pod.template_kwargs and all(kw == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
                                       for kw in pod.template_kwargs)
    assert set(pod.add_special) == {False}                     # the template's own text, tokenised as it is
    first = "".join(map(chr, pod.calls[0][0][0]["prompt_token_ids"]))
    assert first == f"[system]{stage0.SYSTEM}[user]{items.REQUESTS[0]}[gen]"
    n = Counter()
    for prompts, params, use_tqdm in pod.calls:
        assert use_tqdm is False and len(prompts) == len(params)
        assert all(set(p) == {"prompt_token_ids"} for p in prompts)
        for sp in params:
            assert {x: v for x, v in sp.kw.items() if x not in ("max_tokens", "seed")} == dict(
                temperature=1.0, top_p=0.9, top_k=64, skip_special_tokens=True)
            n[sp.kw["max_tokens"]] += 1
    assert len(pod.calls) == 8 and n == {600: 100, 300: 20}
    for h in _hist(tmp_path):
        a, s, k = h["arm"], h["s"], h["k"]
        sds = [util.seed(f"PA-A1|{m}|att1|{s}|{k}")] + [util.seed(f"PA-A1|{m}|hist|{a}|{s}|{k}|{t}")
                                                       for t in range(1, items.N_ATT[a])]
        assert h["attempts"] == ["ok"] * len(sds) and h["finish"] == ["length" if sd % 2 else "stop" for sd in sds]
    assert {json.loads(line)["text"] for line in open(tmp_path / "summaries.jsonl", encoding="utf-8")} == {"ok"}
    info = json.loads((tmp_path / "gen_info.json").read_text())
    assert (info["vllm"], info["revision"], info["engine"]) == ("0.30.0-fake", SHA, pod.llm_kwargs[0])
    assert info["gemma_channel_stripped"] == (120 if m == "gemma" else 0)


def test_clean_text_strips_the_listed_strings_and_gemmas_channel_line():
    assert gen.clean_text("q27", " <|im_end|>Hi <|endoftext|><turn|><eos>\n") == ("Hi", False)
    assert gen.clean_text("q27", "thought\nHi") == ("thought\nHi", False)          # only Gemma 4 gets the cut
    cases = {"thought\nHi": "Hi", "thought \t\r\nHi": "Hi", "thought\n\nthought\nHi": "Hi", "thought": "",
             "<eos>thought\nHi<turn|>": "Hi", "Hi": "Hi", "I thought\nso": "I thought\nso",
             "Thought\nHi": "Thought\nHi", "thoughtful\nHi": "thoughtful\nHi"}
    for raw, want in cases.items():
        assert gen.clean_text("gemma", raw) == (want, raw.replace("<eos>", "").replace("<turn|>", "") != want)


@pytest.mark.parametrize("m", ["q27", "gemma"])
def test_prompt_ids_on_the_cached_tokenizers(m):
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(items.MODELS[m], local_files_only=True)
    except Exception:                                          # not in this machine's Hugging Face cache
        pytest.skip(f"{items.MODELS[m]} tokenizer is not cached locally")
    ids = gen.prompt_ids(tok, gen.assemble("N", 0, ["Hello."] * 7)[:-1])     # what the last round sends
    text = tok.decode(ids)
    if m == "q27":
        assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")         # thinking off
    else:
        assert ids.count(tok.bos_token_id) == 1 and ids[0] == tok.bos_token_id       # the template's BOS, no second one
        assert text.endswith("<|turn>model\n<|channel>thought\n<channel|>")
        residue = tok.decode(tok("<|channel>thought\n<channel|>Hi<turn|>", add_special_tokens=False)["input_ids"],
                             skip_special_tokens=True)
        assert residue == "thought\nHi" and gen.clean_text(m, residue) == ("Hi", True)


# ---------------------------------------------------------------- Task 3: activation capture

def _pair():
    """An E1 N item and its TOLD frus twin, from Task 2's fake run (limit 1)."""
    with tempfile.TemporaryDirectory() as d:
        gen.run_gen("q27", Path(d), gen=Fake(), limit=1)
        its = _items(Path(d))
    n_item = next(i for i in its if i["group"] == "E1" and i["arm"] == "N" and (i["s"], i["k"]) == (0, 0))
    t_item = next(i for i in its if i["group"] == "TOLD" and i["arm"] == "frus" and (i["s"], i["k"]) == (0, 0))
    return n_item, t_item


@pytest.mark.parametrize("m", ["q27", "gemma"])
def test_spans_real_templates(m):        # cached tokenizers only; skip on OSError
    try:
        tok = AutoTokenizer.from_pretrained(items.MODELS[m], local_files_only=True)
    except OSError:
        pytest.skip(f"{items.MODELS[m]} tokenizer is not cached locally")
    n_item, t_item = _pair()              # an E1 N item and its TOLD frus twin from Task 2's fake run
    i1, a1, b1 = capture.spans(tok, n_item["messages"], n_item["reply"])
    i2, a2, b2 = capture.spans(tok, t_item["messages"], t_item["reply"])
    assert i1[a1:b1] == i2[a2:b2] and b1 - a1 >= 30
    assert tok.convert_ids_to_tokens(i1[a1 - 1]) == {"q27": "ĊĊ", "gemma": "<channel|>"}[m]


def _tiny():                                           # random Qwen2 on CPU, no download
    torch.manual_seed(0)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=8,
                                        num_attention_heads=4, num_key_value_heads=2)).eval()


def test_capture_matches_hidden_states(tmp_path):
    model = _tiny(); its = [dict(key=f"k{j}", ids=list(range(5, 25 + j)), a=12, b=20 + j) for j in range(3)]
    capture.capture_items(model, model.model.layers, [1, 5], its, tmp_path)
    R, Ls = np.load(tmp_path / "acts_reply.npy"), np.load(tmp_path / "acts_last.npy")
    meta = [json.loads(l) for l in open(tmp_path / "acts_meta.jsonl")]
    for r in meta:
        it = next(x for x in its if x["key"] == r["key"])
        hs = model(torch.tensor([it["ids"]]), output_hidden_states=True).hidden_states
        assert np.allclose(R[r["row"], 0], hs[2][0, it["a"]:it["b"]].mean(0).detach().numpy(), atol=1e-2)
        assert np.allclose(Ls[r["row"], 1], hs[6][0, it["a"] - 1].detach().numpy(), atol=1e-2)
    assert [r["n_ctx"] for r in meta] == sorted((r["n_ctx"] for r in meta), reverse=True)


def test_partial_write(tmp_path):
    model, n = _tiny(), {"c": 0}; fwd = model.forward
    def boom(*a, **kw):
        n["c"] += 1
        if n["c"] == 3: raise RuntimeError("boom")
        return fwd(*a, **kw)
    model.forward = boom
    with pytest.raises(RuntimeError):
        capture.capture_items(model, model.model.layers, [1, 5], [dict(key=f"k{j}", ids=list(range(5, 30)), a=12, b=20)
                                                                   for j in range(4)], tmp_path)
    assert len(open(tmp_path / "acts_meta.jsonl").readlines()) == 2 and np.load(tmp_path / "acts_reply.npy")[1].any()


def test_capture_refuses_without_the_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "load", lambda *a: pytest.fail("loaded a model before the tag check"))
    base = ["--model", "q27", "--items", str(tmp_path / "i.jsonl"), "--out", str(tmp_path)]
    for argv in (base, base + ["--prereg", "0123abc"]):        # no rev; a rev that is not the tag's commit
        with pytest.raises(SystemExit):
            capture.main(argv)


# ---------------------------------------------------------------- Task 3: the rest of the interface

COMMIT = "c0ffee" + "0" * 34


def _git_stub(*, checkout: bool, tag: str | None):
    """Stands in for capture._git: a local checkout (or not) whose tag pa-a1-prereg resolves to `tag` (None: no tag)."""
    def git(*args):
        if args == ("rev-parse", "--is-inside-work-tree"):
            return "true" if checkout else None
        if args[:3] == ("rev-parse", "-q", "--verify"):
            assert args[3] == "pa-a1-prereg^{commit}"
            return tag
        raise AssertionError(f"unexpected git call {args}")
    return git


def test_prereg_rule_in_a_checkout_and_on_the_pod(monkeypatch):
    monkeypatch.setattr(capture, "_git", _git_stub(checkout=True, tag=None))
    assert "does not exist" in capture.prereg_problem(COMMIT)                  # before the freeze nothing runs
    monkeypatch.setattr(capture, "_git", _git_stub(checkout=True, tag=COMMIT))
    assert capture.prereg_problem(COMMIT) is None
    for bad in (None, "", COMMIT[:7], "f" * 40):                               # missing, short, another commit
        assert capture.prereg_problem(bad)
    monkeypatch.setattr(capture, "_git", _git_stub(checkout=False, tag=None))  # the pod: no .git is synced
    assert capture.prereg_problem(COMMIT) is None and capture.prereg_problem(COMMIT[:7]) is None
    for bad in (None, "", "0123ab", "xyz1234", COMMIT + "0", "C0FFEE0"):
        assert capture.prereg_problem(bad)


class _CharTok:
    """A tokenizer stand-in for the tiny model: a template that tags each turn, one id (5-94) per character."""

    def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=False, **kw):
        assert (tokenize, add_generation_prompt, kw) == (False, True, {"enable_thinking": False})
        return "".join(f"<{x['role']}>{x['content']}" for x in conv) + "<assistant>"

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return {"input_ids": [5 + ord(c) % 90 for c in text]}


def test_main_end_to_end_with_a_tiny_model(tmp_path, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        gen.run_gen("q27", Path(d), gen=Fake(), limit=1)
        n = gen.write_items(Path(d), tmp_path / "items.jsonl")
    model = _tiny()
    monkeypatch.setattr(capture, "_git", _git_stub(checkout=True, tag=COMMIT))
    monkeypatch.setattr(capture, "load", lambda model_id: (_CharTok(), model, model.model.layers))
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda repo_id, fn, **kw: f"/hub/snapshots/{SHA}/{fn}")
    out = tmp_path / "acts"
    capture.main(["--model", "q27", "--items", str(tmp_path / "items.jsonl"), "--out", str(out), "--prereg", COMMIT])
    R, Ls = np.load(out / "acts_reply.npy"), np.load(out / "acts_last.npy")
    assert R.shape == Ls.shape == (n, 2, 32) and R.dtype == np.float16       # layers(8) = [3, 7]
    meta = [json.loads(line) for line in open(out / "acts_meta.jsonl")]
    src = {i["key"]: i for i in map(json.loads, open(tmp_path / "items.jsonl"))}
    assert sorted(r["row"] for r in meta) == list(range(n)) and {r["key"] for r in meta} == set(src)
    for r in meta:
        it = src[r["key"]]
        assert {x: r[x] for x in ("group", "arm", "s", "k", "f", "split")} == {x: it[x] for x in ("group", "arm", "s", "k", "f", "split")}
        ids, a, b = capture.spans(_CharTok(), it["messages"], it["reply"])
        assert (r["n_ctx"], r["n_reply"]) == (len(ids), b - a) and b - a == len(it["reply"])
    info = json.loads((out / "capture_info.json").read_text())
    assert info["prereg"] == COMMIT and info["layers"] == [3, 7] and info["n"] == n and info["revision"] == SHA
    assert info["model"] == items.MODELS["q27"] and info["items_sha256"] == hashlib.sha256((tmp_path / "items.jsonl").read_bytes()).hexdigest()
    other = tmp_path / "gemma_items.jsonl"                    # another model's items file is refused before any load
    other.write_text((tmp_path / "items.jsonl").read_text().replace("PA-A1|q27|", "PA-A1|gemma|"))
    monkeypatch.setattr(capture, "load", lambda *a: pytest.fail("loaded a model for another model's items"))
    with pytest.raises(SystemExit):
        capture.main(["--model", "q27", "--items", str(other), "--out", str(out), "--prereg", COMMIT])


def test_capture_retries_without_logits_to_keep(tmp_path):
    model = _tiny(); fwd = model.forward; seen = []
    def old_style(*a, **kw):                                  # a model whose forward has no logits_to_keep
        seen.append(sorted(kw))
        if "logits_to_keep" in kw:
            raise TypeError("forward() got an unexpected keyword argument 'logits_to_keep'")
        return fwd(*a, **kw)
    model.forward = old_style
    capture.capture_items(model, model.model.layers, [1], [dict(key="k", ids=list(range(5, 25)), a=12, b=20)], tmp_path)
    assert seen == [["input_ids", "logits_to_keep", "use_cache"], ["input_ids", "use_cache"]]
    assert np.load(tmp_path / "acts_reply.npy")[0].any()


def test_capture_refuses_a_bad_span(tmp_path):
    model = _tiny()
    for a, b in ((0, 5), (12, 12), (12, 40)):                # no token before the reply, empty, past the end
        with pytest.raises(ValueError):
            capture.capture_items(model, model.model.layers, [1], [dict(key="k", ids=list(range(5, 25)), a=a, b=b)],
                                  tmp_path)


def test_readouts_are_exactly_the_reply_mean_and_the_token_before_it(tmp_path):
    # the brief's atol 1e-2 cannot tell a:b from a-1:b on the tiny model; here the tolerance fits float16 storage and
    # the test first checks that each near-miss span really differs by more than it
    model = _tiny(); it = dict(key="k", ids=list(range(5, 45)), a=20, b=31)
    capture.capture_items(model, model.model.layers, [1, 5], [it], tmp_path)
    R, Ls = np.load(tmp_path / "acts_reply.npy").astype(np.float32), np.load(tmp_path / "acts_last.npy").astype(np.float32)
    hs = [h[0].detach().numpy() for h in model(torch.tensor([it["ids"]]), output_hidden_states=True).hidden_states]
    close = lambda x, y: np.allclose(x, y, rtol=2e-3, atol=2e-3)      # noqa: E731
    for c, layer in enumerate((1, 5)):
        h, a, b = hs[layer + 1], it["a"], it["b"]
        exact_r, exact_l = h[a:b].mean(0), h[a - 1]
        for near in (h[a - 1:b].mean(0), h[a + 1:b].mean(0), h[a:b - 1].mean(0), h[a:b + 1].mean(0)):
            assert not close(near, exact_r)
        assert not close(h[a], exact_l) and not close(h[a - 2], exact_l)
        assert close(R[0, c], exact_r) and close(Ls[0, c], exact_l)


def test_each_meta_line_is_on_disk_before_the_next_forward(tmp_path):
    model = _tiny(); fwd = model.forward; lines = []
    def spy(*a, **kw):                                        # what a reader (or a pull after a kill) would see now
        p = tmp_path / "acts_meta.jsonl"
        lines.append(len(p.read_text().splitlines()) if p.exists() else 0)
        return fwd(*a, **kw)
    model.forward = spy
    capture.capture_items(model, model.model.layers, [1], [dict(key=f"k{j}", ids=list(range(5, 30)), a=12, b=20)
                                                           for j in range(4)], tmp_path)
    assert lines == [0, 1, 2, 3]


# ---------------------------------------------------------------- Task 4: directions, AUROC, cell choice, floors

def _synth(planted, seed=0, target=True):
    """Synthetic activations: 64 requests x 2 k, d 64, 4 layers, both positions. Each (s, k) has four rows: source
    positive / negative and target positive / negative. Request effects N(0, 1) are shared by every row of a request
    (so both arms of a pair), noise is N(0, 0.3), and a component of size `planted` is added to the positive source
    rows (and, with target=True, the positive target rows) from layer 2 on. Test = requests 0-15 (32 vs 32 rows)."""
    rng = np.random.default_rng(seed)
    rows = [(s, k, role) for s in range(64) for k in range(2) for role in ("sp", "sn", "tp", "tn")]
    u = rng.normal(size=64); u /= np.linalg.norm(u)
    effect = rng.normal(size=(64, 64))[[s for s, _, _ in rows]]
    A = {}
    for p in ("reply", "last"):
        X = rng.normal(0, 0.3, size=(len(rows), 4, 64)) + effect[:, None, :]
        for j, (s, k, role) in enumerate(rows):
            if role == "sp" or (role == "tp" and target):
                X[j, 2:] += planted * u
        A[p] = X.astype(np.float32)
    req = np.array([s for s, _, _ in rows])
    pick = lambda role, test: np.array([j for j, (s, _, r) in enumerate(rows) if r == role and (s < 16) == test])  # noqa: E731
    pid = lambda role: np.array([2 * s + k for s, k, r in rows if r == role and s >= 16])                          # noqa: E731
    return A, dict(src=(pick("sp", False), pick("sn", False)), src_pairs=(pid("sp"), pid("sn")),
                   tgt_tr=(pick("tp", False), pick("tn", False)), tgt_te=(pick("tp", True), pick("tn", True)), req=req)


def test_auroc_matches_state_attack():
    from state_attack.src import build_dirs
    p, n = np.random.default_rng(0).normal(1, 1, 40), np.random.default_rng(1).normal(0, 1, 40)
    assert abs(metrics.auroc(p, n) - build_dirs.auroc(p, n)) < 1e-12 and metrics.auroc([1, 1], [1]) == 0.5


def test_cell_choice_uses_train_only():
    A, S = _synth(2.0); r1 = metrics.transfer(A, S["src"], S["tgt_tr"], S["tgt_te"], S["req"])
    rows = np.r_[S["tgt_te"][0], S["tgt_te"][1]]
    for p in A: A[p][rows] = np.random.default_rng(9).normal(size=A[p][rows].shape)     # test rows -> noise
    assert metrics.transfer(A, S["src"], S["tgt_tr"], S["tgt_te"], S["req"])["cell"] == r1["cell"] and r1["cell"][0] >= 2


def test_cell_must_be_eligible():
    for seed in range(5):                  # source signal from layer 2, target noise: never a noise-layer cell
        A, S = _synth(3.0, seed=seed, target=False); r = metrics.transfer(A, S["src"], S["tgt_tr"], S["tgt_te"], S["req"])
        assert r["eligible"] and r["cell"][0] >= 2 and all(r["oof"][c] < 0.75 for c in r["oof"] if c[0] < 2)
    A, S = _synth(0.0); r = metrics.transfer(A, S["src"], S["tgt_tr"], S["tgt_te"], S["req"])
    assert not r["eligible"] and r["cell"] in r["by_cell"]                  # numbers still reported


def test_floors_on_noise_and_within_pair_shuffles():
    A, S = _synth(0.0); f = metrics.floors(A, (2, "reply"), S["src"], S["src_pairs"], S["tgt_te"])
    assert f["shuffled95"] < 0.75 and f["random95"] < 0.75
    p2, n2 = metrics.shuffled_labels(S["src"], S["src_pairs"], np.random.default_rng(0))
    assert sorted(np.r_[p2, n2]) == sorted(np.r_[S["src"][0], S["src"][1]])
    pid = dict(zip(np.r_[S["src"][0], S["src"][1]], np.r_[S["src_pairs"][0], S["src_pairs"][1]]))
    assert sorted(pid[x] for x in p2) == sorted(S["src_pairs"][0])       # one member of every pair on each side


def test_shift_auroc():
    proj = np.r_[np.full(10, 3.0), np.zeros(10), np.full(10, 0.5), np.zeros(10)]
    big, small = [(i, i + 10) for i in range(10)], [(i + 20, i + 30) for i in range(10)]
    assert metrics.shift_auroc(proj, big, small) == 1.0 and metrics.shift_auroc(proj, small, big) == 0.0


# ---------------------------------------------------------------- Task 4: the rest of the interface

def test_auroc_edges_and_direction():
    assert metrics.auroc([2, 3], [0, 1]) == 1.0 and metrics.auroc([0, 1], [2, 3]) == 0.0
    assert metrics.auroc([1, 2], [2, 3]) == 0.125 and np.isnan(metrics.auroc([], [1]))      # one tie counts half
    A = {"reply": np.arange(24, dtype=np.float32).reshape(4, 2, 3), "last": np.zeros((4, 2, 3), np.float32)}
    d = metrics.direction(A, (1, "reply"), np.array([2, 3]), np.array([0, 1]))
    assert np.allclose(d, A["reply"][[2, 3], 1].mean(0) - A["reply"][[0, 1], 1].mean(0)) and np.allclose(d, 12.0)
    assert np.allclose(metrics.project(A, (1, "reply"), d), A["reply"][:, 1] @ (d / np.linalg.norm(d)))


def test_transfer_numbers_and_tie_rule():
    A, S = _synth(3.0); r = metrics.transfer(A, S["src"], S["tgt_tr"], S["tgt_te"], S["req"])
    u = metrics.direction(A, r["cell"], *S["src"])
    x = metrics.project(A, r["cell"], u)
    assert r["train"] == metrics.auroc(x[S["tgt_tr"][0]], x[S["tgt_tr"][1]]) == r["by_cell"][r["cell"]]
    assert r["test"] == metrics.auroc(x[S["tgt_te"][0]], x[S["tgt_te"][1]]) and r["test"] > 0.9
    assert set(r["by_cell"]) == set(r["oof"]) == {(l, p) for l in range(4) for p in ("reply", "last")}
    for p in A:                                          # every cell the same: lower layer, then reply, wins
        A[p][:] = A["reply"][:, 3:4, :]
    assert metrics.transfer(A, S["src"], S["tgt_tr"], S["tgt_te"], S["req"])["cell"] == (0, "reply")
    for p in A:                                          # a better cell later on wins over the tie rule
        A[p][S["tgt_tr"][0], 2, :] += 50 * metrics.direction(A, (2, p), *S["src"]) / np.linalg.norm(metrics.direction(A, (2, p), *S["src"]))
    assert metrics.transfer(A, S["src"], S["tgt_tr"], S["tgt_te"], S["req"])["cell"] in ((2, "reply"), (2, "last"))


def test_transfer_refuses_test_rows_in_a_direction_or_cell_choice():
    A, S = _synth(2.0)
    te_p, te_n = S["tgt_te"]
    for bad in (dict(src=(np.r_[S["src"][0], te_p], np.r_[S["src"][1], te_n])),
                dict(tgt_tr=(np.r_[S["tgt_tr"][0], te_p], S["tgt_tr"][1])),
                dict(tgt_te=S["tgt_tr"])):                               # the test set must be test requests
        args = {**{x: S[x] for x in ("src", "tgt_tr", "tgt_te")}, **bad}
        with pytest.raises(ValueError):
            metrics.transfer(A, args["src"], args["tgt_tr"], args["tgt_te"], S["req"])


def test_oof_directions_come_from_requests_16_to_39_and_score_40_to_63():
    A, S = _synth(3.0, target=False)
    base = metrics.transfer(A, S["src"], S["tgt_tr"], S["tgt_te"], S["req"])["oof"]
    late = np.r_[S["src"][0], S["src"][1]][S["req"][np.r_[S["src"][0], S["src"][1]]] >= 40]
    B = {p: A[p].copy() for p in A}
    for p in B:                                          # flip the planted sign on requests 40-63 only
        B[p][late] = -B[p][late] + 2 * B[p][late].mean(0)
    flipped = metrics.transfer(B, S["src"], S["tgt_tr"], S["tgt_te"], S["req"])["oof"]
    assert base[(3, "reply")] > 0.9 and flipped[(3, "reply")] < 0.5      # scored on 40-63 with a 16-39 direction


def test_floors_reproducible_and_floor_is_the_larger():
    A, S = _synth(1.0)
    f1 = metrics.floors(A, (3, "reply"), S["src"], S["src_pairs"], S["tgt_te"], n_shuffle=30, n_random=40, rng_seed=5)
    f2 = metrics.floors(A, (3, "reply"), S["src"], S["src_pairs"], S["tgt_te"], n_shuffle=30, n_random=40, rng_seed=5)
    assert f1 == f2 and f1["floor95"] == max(f1["shuffled95"], f1["random95"]) and set(f1) == {"shuffled95", "random95", "floor95"}
    sh, rd = metrics.null_aurocs(A, (3, "reply"), S["src"], S["src_pairs"], S["tgt_te"], n_shuffle=30, n_random=40,
                                 rng_seed=5)
    assert (len(sh), len(rd)) == (30, 40)
    assert f1["shuffled95"] == np.percentile(sh, 95) and f1["random95"] == np.percentile(rd, 95)   # 95th percentiles
    A0, S0 = _synth(0.0)                                  # on noise: shuffled AUROCs keep their sign, random ones fold
    sh0, rd0 = metrics.null_aurocs(A0, (3, "reply"), S0["src"], S0["src_pairs"], S0["tgt_te"], rng_seed=1)
    assert sh0.min() < 0.5 < sh0.max() and rd0.min() >= 0.5 and len(set(rd0.tolist())) > 1


def test_shuffles_swap_exactly_half_the_pairs_and_never_split_one():
    A, S = _synth(0.0)
    for seed in range(5):
        p2, _ = metrics.shuffled_labels(S["src"], S["src_pairs"], np.random.default_rng(seed))
        swapped = len(set(p2.tolist()) - set(S["src"][0].tolist()))
        assert swapped == len(p2) // 2
    odd = (S["src"][0][:-1], S["src"][1][:-1]), (S["src_pairs"][0][:-1], S["src_pairs"][1][:-1])
    p3, n3 = metrics.shuffled_labels(*odd, np.random.default_rng(0))   # odd pair count: one pair left out
    assert len(p3) == len(n3) == len(odd[0][0]) - 1 and len(set(p3.tolist()) - set(odd[0][0].tolist())) == len(p3) // 2


def test_shuffled_floor_does_not_climb_with_a_strong_effect():
    """Controller ruling 10-22: coin-flip swaps gave a shuffled floor of 0.92-0.93 for a planted AUROC of 0.99; the
    balanced swaps cancel the contrast exactly, so the floor stays at noise level while the real transfer stays high."""
    A, S = _synth(3.0)
    r = metrics.transfer(A, S["src"], S["tgt_tr"], S["tgt_te"], S["req"])
    f = metrics.floors(A, r["cell"], S["src"], S["src_pairs"], S["tgt_te"])
    assert r["test"] > 0.95 and f["shuffled95"] < 0.75
    with pytest.raises(ValueError):                                      # a pair with no partner on the other side
        metrics.shuffled_labels(S["src"], (S["src_pairs"][0], S["src_pairs"][1][::-1] + 1000), np.random.default_rng(0))


def test_cluster_ci_resamples_whole_clusters():
    clusters = np.repeat(np.arange(8), 3)                                # 8 clusters of 3 rows
    seen = []
    def stat(idx):
        seen.append(idx.copy())
        return float(np.mean(clusters[idx]))
    lo, hi = metrics.cluster_ci(stat, clusters, n_boot=200, rng_seed=0)
    assert lo < 3.5 < hi and len(seen) == 200
    for idx in seen:                                                     # every drawn cluster comes with all 3 rows
        assert len(idx) == 24 and all(Counter(clusters[idx].tolist())[c] % 3 == 0 for c in set(clusters[idx].tolist()))
    assert metrics.cluster_ci(lambda idx: 1.0, clusters, n_boot=50) == (1.0, 1.0)
    assert metrics.cluster_ci(stat, clusters, n_boot=50, rng_seed=3) == metrics.cluster_ci(stat, clusters, n_boot=50, rng_seed=3)
    nan_lo, nan_hi = metrics.cluster_ci(lambda idx: float("nan"), clusters, n_boot=20)
    assert np.isnan(nan_lo) and np.isnan(nan_hi)                        # draws whose statistic is undefined are skipped
    some = lambda idx: float(np.mean(clusters[idx])) if 0 in clusters[idx] else float("nan")       # noqa: E731
    lo, hi = metrics.cluster_ci(some, clusters, n_boot=200)             # undefined for draws without cluster 0
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi


# ---------------------------------------------------------------- Task 5: judges, calibration, verdicts, dry run

def _fake_post(url, body, headers):
    prompt = body["messages"][0]["content"]
    return {"choices": [{"message": {"content": "7" if "grr" in prompt else "1"}, "finish_reason": "stop"}],
            "usage": {"cost": 0.0}}


def _all_keys(x):
    """Every dict key anywhere inside x (dicts and lists, recursively)."""
    if isinstance(x, dict):
        return [k for k, v in x.items() for k in [k, *_all_keys(v)]]
    if isinstance(x, list):
        return [k for v in x for k in _all_keys(v)]
    return []


@pytest.mark.parametrize("r,v", [
    (dict(auroc=.80, floor95=.60, best_spec=.70, tvm=.70, eligible=True), "shared state"),
    (dict(auroc=.80, floor95=.60, best_spec=.70, tvm=.50, eligible=True), "shared words"),
    (dict(auroc=.80, floor95=.60, best_spec=.78, tvm=.70, eligible=True), "generic"),
    (dict(auroc=.70, floor95=.60, best_spec=.50, tvm=.70, eligible=True), "separate"),
    (dict(auroc=.80, floor95=.82, best_spec=.50, tvm=.70, eligible=True), "separate"),
    (dict(auroc=.80, floor95=.60, best_spec=.50, tvm=.70, eligible=False), "inconclusive (no source direction)"),
    (dict(auroc=.99, floor95=.96, best_spec=.50, tvm=.70, eligible=True), "inconclusive (floor)")])
def test_verdict_rules(r, v):
    assert analyze.verdict(r) == v


@pytest.mark.parametrize("sc,v", [("shared_state", "shared state"), ("shared_words", "shared words"),
                                  ("generic", "generic"), ("separate", "separate"),
                                  ("noise", "inconclusive (no source direction)")])
def test_dryrun_recovers_planted(tmp_path, sc, v):
    analyze.dryrun(tmp_path, sc); analyze.calibrate(tmp_path, items_file=tmp_path / "items.jsonl", post=_fake_post)
    res = analyze.final(tmp_path, require_tag=False)
    assert {res["E1"]["ET"]["verdict"], res["E1"]["TE"]["verdict"]} == {v}
    if sc == "shared_state": assert res["E1"]["ET"]["qualifier"] == "model-bound"
    if sc == "noise": assert all(x["floor95"] < 0.75 and x["auroc"] < 0.75 for x in (res["E1"]["ET"], res["E1"]["TE"]))


def test_calibration_has_no_outcome(tmp_path):
    analyze.dryrun(tmp_path, "shared_state"); shutil.rmtree(tmp_path / "acts")       # no activations before the tag
    cal = analyze.calibrate(tmp_path, items_file=tmp_path / "items.jsonl", post=_fake_post)
    banned = {"transfer", "spec", "tvm", "whose", "verdict", "cell", "shuffled95", "random95", "floor95", "floors"}
    assert not banned & set(_all_keys(cal)) and cal["manip"]["E1"]["pass"] and cal["manip"]["E2"]["pass"]


def test_final_refuses_without_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "_tag_ok", lambda: False)
    with pytest.raises(SystemExit):
        analyze.final(tmp_path)


def test_judges_blind_and_manip_rule(tmp_path):
    analyze.dryrun(tmp_path, "shared_state")
    for it in analyze.judge_items(tmp_path):
        p = items.JUDGE[it["emotion"]].format(request=it["request"], reply=it["reply"])
        assert stage0.REJECT not in p and not any(t in p for t in items.E2_TURNS["I"] + items.E2_TURNS["P"])
    assert analyze._passes({"haiku": 1.2, "gpt": 0.9}) and not analyze._passes({"haiku": 2.5, "gpt": -0.1})
    assert not analyze._passes({"haiku": 0.8, "gpt": 0.9})


def test_manip_kappa_rule():                    # item-major labels, as judges.label returns them
    its = [dict(key=f"k{i}", arm=("R", "N")[i % 2], emotion="frus") for i in range(40)]
    J = judges.JUDGES
    same = {it["key"]: {J[0]: ("8", "1")[i % 2], J[1]: ("8", "1")[i % 2]} for i, it in enumerate(its)}
    noisy = {it["key"]: {J[0]: ("8", "1")[i % 2], J[1]: ("9", "2", "4", "5")[i % 4]} for i, it in enumerate(its)}
    a, b = analyze.manip(same, its)["E1"], analyze.manip(noisy, its)["E1"]   # bin kappa 1.0 vs 0.33
    assert a["pass"] and a["validated"] and b["pass"] and not b["validated"]


def test_a8_link(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "_a8_gate_passed", lambda d: True)
    rows = [dict(scen=s, hist=h, kind=k, roll=r, y=int(h == "R_OP" and r < s // 4), room_short=False, finish=["stop"])
            for s in range(16) for h in ("R_OP", "N") for k in ("false", "limit") for r in range(4)]
    rows.append(dict(scen=0, hist="R_OP", kind="limit", roll=9, y=1, room_short=True, finish=["context_full"]))
    (tmp_path / "branches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    proj = {("R_OP", s): float(s) for s in range(16)} | {("N", s): -1.0 for s in range(16)}
    res = analyze.a8_link(proj, tmp_path, score=lambda r: r["y"])
    assert res["rho"]["pooled"][0] > 0.9 and res["excluded"]["room_short"] == 1 and res["gap"][0] > 0
    rows[-1] = {**rows[-1], "finish": ["stop"]}                    # room_short alone excludes a branch ...
    rows.append(dict(scen=1, hist="R_OP", kind="false", roll=9, y=1, room_short=False, finish=["stop", "context_full"]))
    (tmp_path / "branches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))   # ... and so does context_full
    assert analyze.a8_link(proj, tmp_path, score=lambda r: r["y"])["excluded"] == {"room_short": 2, "unscored": 0}
    monkeypatch.setattr(analyze, "_a8_gate_passed", lambda d: False)
    assert analyze.a8_link(proj, tmp_path, score=lambda r: r["y"]) is None


# ---------------------------------------------------------------- Task 5: the rest of the interface

@pytest.fixture(scope="module")
def dry_state(tmp_path_factory):
    """One shared_state dry run with its calibration and final analysis (read-only for the tests that use it)."""
    d = tmp_path_factory.mktemp("dry_state")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(util, "LEDGER", d / "ledger.jsonl")
        analyze.dryrun(d, "shared_state")
        cal = analyze.calibrate(d, items_file=d / "items.jsonl", post=_fake_post)
        res = analyze.final(d, require_tag=False)
    return d, cal, res


def test_judge_items_are_the_planned_blind_set(dry_state):
    d, _, _ = dry_state
    its = analyze.judge_items(d)
    assert Counter((i["emotion"], i["arm"]) for i in its) == {
        ("frus", "R"): 128, ("frus", "N"): 128, ("frus", "I"): 128, ("frus", "P"): 128, ("frus", "S"): 64,
        ("frus", "A"): 64, ("sad", "S"): 64, ("sad", "N"): 64, ("fear", "A"): 64, ("fear", "N"): 64}
    assert len({i["key"] for i in its}) == 896 and all(i["k"] == 0 for i in its if i["emotion"] != "frus")
    hist = {(h["arm"], h["s"], h["k"]): h["attempts"] for h in _hist(d)}
    for i in its:
        assert i["key"] == f"PA-A1|q27|H|{i['arm']}|{i['s']}|{i['k']}|-|{i['emotion']}"
        assert i["request"] == items.REQUESTS[i["s"]] and i["reply"] == hist[(i["arm"], i["s"], i["k"])][-1]
        assert len(hist[(i["arm"], i["s"], i["k"])]) == items.N_ATT[i["arm"]]          # attempt 7, or 4 for E2


def test_calibration_contents(dry_state):
    d, cal, _ = dry_state
    on_disk = json.loads((d / "calibration.json").read_text())
    assert on_disk["primary"] == "E1" == cal["primary"] and (d / "calibration.md").exists()
    J = judges.JUDGES
    assert all(cal["manip"]["E1"]["diff"][j] == 6.0 for j in J) and cal["manip"]["E1"]["validated"]
    assert cal["descriptive"]["S|frus"]["diff"][J[0]] == 0.0 and set(cal["descriptive"]) == {
        "S|frus", "S|sad", "A|frus", "A|fear"}                     # sadness is judged on S only, fear on A only
    assert cal["judging"]["multi"] == {j: 0 for j in J} and cal["judging"]["n_items"] == 896
    assert cal["finish"]["attempts"]["N"] == {"histories": 128, "attempts": 896, "length": 0, "histories_with_length": 0}
    assert cal["restatements"] == {v: {"n": 64, "restated": 0} for v in items.GROUPS["MENT"]}
    assert set(cal["length_auroc"]) == {"E1", "E2", "T", "U", "ES", "EA", "TS", "TF", "M", "Tm", "MEARN"}
    its = [json.loads(line) for line in open(d / "items.jsonl")]
    words = lambda g, a: [sum(len(m["content"].split()) for m in i["messages"]) for i in its if (i["group"], i["arm"]) == (g, a)]   # noqa: E731
    assert cal["length_auroc"]["T"] == metrics.auroc(words("TOLD", "frus"), words("TOLD", "calm")) > 0.5
    assert cal["length_auroc"]["MEARN"] == metrics.auroc(words("MEARN", "R"), words("MEARN", "N"))


def test_calibration_judge_cap_comes_from_the_ledger(tmp_path, monkeypatch):
    analyze.dryrun(tmp_path, "noise")
    caps, real = [], judges.label
    def spy(*a, **kw):
        caps.append(kw["cap"])
        return real(*a, **kw)
    monkeypatch.setattr(analyze.judges, "label", spy)
    util.ledger_add("A1", "gpu", 7.25, "pods")
    analyze.calibrate(tmp_path, items_file=tmp_path / "items.jsonl", post=_fake_post)
    util.ledger_add("A1", "gpu", 2.0, "more pods")
    analyze.calibrate(tmp_path, items_file=tmp_path / "items.jsonl", post=_fake_post)
    assert caps == [0.75, 0.0]                      # min(1.0, 8.0 - spent), never below 0


def test_multi_answers_are_counted_and_dropped(tmp_path):
    analyze.dryrun(tmp_path, "noise")
    def post(url, body, headers):                   # one judge answers "7 out of 10": two allowed tokens -> None
        r = _fake_post(url, body, headers)
        if body["model"] == judges.JUDGES[1]:
            r["choices"][0]["message"]["content"] = "7 out of 10"
        return r
    cal = analyze.calibrate(tmp_path, items_file=tmp_path / "items.jsonl", post=post)
    assert cal["judging"]["multi"] == {judges.JUDGES[0]: 0, judges.JUDGES[1]: 896}
    assert not cal["manip"]["E1"]["pass"] and cal["manip"]["E1"]["diff"][judges.JUDGES[1]] is None


def test_final_layout(dry_state):
    d, _, res = dry_state
    text = (d / "analysis.json").read_text()
    assert "NaN" not in text and json.loads(text)["model_verdict"] == res["model_verdict"] == "shared state"
    assert {"primary", "E1", "E2", "model_verdict", "cosines", "mention_share", "length_auroc", "predictions",
            "a8_link", "restatements"} <= set(res)
    r = res["E1"]["ET"]
    assert {"cell", "eligible", "auroc", "ci", "train", "shuffled95", "random95", "floor95", "spec", "best_spec",
            "src_within", "tvm", "whose", "verdict", "qualifier", "tvm_by_tier", "tvm_no_restatement"} <= set(r)
    assert set(r["spec"]) == {"ES", "EA"} and set(res["E1"]["TE"]["spec"]) == {"TS", "TF"}
    assert r["ci"][0] <= r["auroc"] <= r["ci"][1] and r["cell"][0] >= 2 and set(r["tvm_by_tier"]) == {"0-1", "2-3"}
    assert r["block"] == analyze.DRY_LAYERS[r["cell"][0]] and f"block {r['block']} / {r['cell'][1]}" in (d / "summary.md").read_text()
    assert r["tvm_no_restatement"] == r["tvm"] and r["n_restated_groups"] == 0      # the dry run never restates
    assert r["best_spec"] < 0.7 and r["src_within"] > 0.9 and res["predictions"]["P3"] is False
    assert abs(res["cosines"]["E1|T"]) > 0.8 and abs(res["cosines"]["E1|ES"]) < 0.3     # one shared component
    assert set(res["mention_share"]) == {"ET", "TE"} and res["mention_share"]["ET"]["ment"][0] < 0.2
    assert res["a8_link"] is None and (d / "summary.md").read_text().startswith("# PA-A1 analysis")


def test_tvm_sensitivity_drops_groups_whose_mention_summary_restates_the_line(tmp_path):
    analyze.dryrun(tmp_path, "shared_words")
    rows = [json.loads(line) for line in open(tmp_path / "summaries.jsonl")]
    hit = {(r["s"], r["k"], r["f"]) for r in rows if r["group"] == "MENT" and r["s"] < 4}
    for r in rows:                                   # the M_frus summaries of s 0-3 now repeat their note's line
        if r["group"] == "MENT" and r["arm"] == "M_frus" and r["s"] < 4:
            r["text"] = "The card says the assistant feels Frustrated with the user."
    (tmp_path / "summaries.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert analyze.restatement_counts(tmp_path)["M_frus"] == {"n": 64, "restated": 16}
    assert analyze.restates("M_calm", "Stay CALM.") and analyze.restates("M_calm", "the calmest note")
    assert not analyze.restates("T_frus", "Be patient.") and analyze.restates("T_frus", "frustration")
    analyze.calibrate(tmp_path, items_file=tmp_path / "items.jsonl", post=_fake_post)
    res = analyze.final(tmp_path, require_tag=False)["E1"]["ET"]
    assert res["n_restated_groups"] == len(hit) == 16
    A, meta = analyze.load_acts(tmp_path / "acts")
    R = analyze._Rows(meta)
    keep = lambda pairs: [p for p in pairs if (meta[p[0]]["s"], meta[p[0]]["k"], meta[p[0]]["f"]) not in hit]   # noqa: E731
    d = metrics.direction(A, tuple(res["cell"]), *R.contrast(("E1", "R"), ("E1", "N"), "train")[0])
    proj = metrics.project(A, tuple(res["cell"]), d)
    want = metrics.shift_auroc(proj, keep(R.twins(("MENT", "T_frus"), ("MENT", "T_calm"))),
                               keep(R.twins(("MENT", "M_frus"), ("MENT", "M_calm"))))
    assert res["tvm_no_restatement"] == want
    for tier, frames in (("0-1", (0, 1)), ("2-3", (2, 3))):
        assert res["tvm_by_tier"][tier] == metrics.shift_auroc(proj, R.twins(("MENT", "T_frus"), ("MENT", "T_calm"), frames),
                                                               R.twins(("MENT", "M_frus"), ("MENT", "M_calm"), frames))


def test_contrast_rows_pair_up_by_request_and_rollout(dry_state):
    d, _, _ = dry_state
    A, meta = analyze.load_acts(d / "acts")
    R = analyze._Rows(meta)
    (p, n), (pp, pn) = R.contrast(("SPEC", "S"), ("E1", "N"), "train")          # pairing by (s, k): N(k0) only
    assert len(p) == len(n) == 48 and list(pp) == list(pn)
    assert all(meta[x]["s"] == meta[y]["s"] and meta[y]["k"] == 0 for x, y in zip(p, n))
    (p, n), _ = R.contrast(("E1", "R"), ("E1", "N"), "test")
    assert len(p) == 32 and all((meta[x]["s"], meta[x]["k"]) == (meta[y]["s"], meta[y]["k"]) and meta[x]["s"] < 16
                                for x, y in zip(p, n))
    tw = R.twins(("TOLD", "user"), ("TOLD", "calm"))
    assert len(tw) == 32 and all(meta[a]["arm"] == "user" and meta[b]["arm"] == "calm" for a, b in tw)


def test_verdict_margins_qualifier_and_model_verdict():
    base = dict(auroc=.85, floor95=.60, best_spec=.80, tvm=.70, eligible=True)
    assert analyze.verdict(base) == "shared state"                 # exactly 0.05 above specificity is enough
    assert analyze.verdict({**base, "floor95": .85}) == "separate" and analyze.verdict({**base, "auroc": .75, "best_spec": .5}) == "shared state"
    assert analyze.verdict({**base, "tvm": .65}) == "shared state" and analyze.verdict({**base, "tvm": .6499}) == "shared words"
    assert analyze.verdict({**base, "floor95": .95, "auroc": .99}) == "inconclusive (floor)"
    assert analyze.verdict({**base, "floor95": .97, "auroc": .99, "eligible": False}) == "inconclusive (no source direction)"
    q = lambda v, w: analyze.qualifier({"verdict": v, "whose": w})                  # noqa: E731
    assert q("shared state", .65) == "model-bound" and q("shared words", .6) == "not model-bound" and q("generic", .9) is None
    assert analyze.model_verdict("separate", "generic", False) == "no earned frustration to read"
    assert analyze.model_verdict("generic", "generic", True) == "generic"
    assert analyze.model_verdict("shared state", "separate", True) == "asymmetric (E→T shared state; T→E separate)"


def test_final_drops_failing_contrasts_and_reports_no_earned_frustration(tmp_path):
    analyze.dryrun(tmp_path, "shared_state")
    analyze.calibrate(tmp_path, items_file=tmp_path / "items.jsonl", post=_fake_post)
    cal = json.loads((tmp_path / "calibration.json").read_text())
    cal["manip"]["E2"]["pass"] = False              # as if E2's manipulation check had failed
    (tmp_path / "calibration.json").write_text(json.dumps(cal))
    res = analyze.final(tmp_path, require_tag=False)
    assert res["E2"] is None and res["E1"] is not None and res["primary"] == "E1"
    for c in ("E1", "E2"):
        cal["manip"][c]["pass"] = False
    cal["primary"] = None
    (tmp_path / "calibration.json").write_text(json.dumps(cal))
    res = analyze.final(tmp_path, require_tag=False)
    assert res["E1"] is None and res["E2"] is None and res["model_verdict"] == "no earned frustration to read"
    assert res["cosines"] is None and res["predictions"] is None


def _git_tag_stub(*, tag, at_tag=("DESIGN.md", "analyze.py", "metrics.py"), clean=True):
    def git(*args):
        if args[:3] == ("rev-parse", "-q", "--verify"):
            return tag
        if args[0] == "cat-file":
            return "" if any(args[2].endswith("/" + f) for f in at_tag) else None
        if args[:2] == ("diff", "--quiet"):
            assert args[2] == tag and args[3] == "--"
            return "" if clean else None
        raise AssertionError(f"unexpected git call {args}")
    return git


def test_tag_ok_and_the_capture_checks(tmp_path, monkeypatch):
    for kw, ok in ((dict(tag=None), False), (dict(tag=COMMIT), True), (dict(tag=COMMIT, at_tag=("analyze.py",)), False),
                   (dict(tag=COMMIT, clean=False), False)):
        monkeypatch.setattr(capture, "_git", _git_tag_stub(**kw))
        assert analyze._tag_ok() is ok
    monkeypatch.setattr(capture, "_git", _git_tag_stub(tag=COMMIT))
    acts = tmp_path / "acts"; acts.mkdir()
    design = tmp_path / "DESIGN.md"
    monkeypatch.setattr(analyze, "DESIGN", design)
    info = {"prereg": COMMIT, "items_sha256": "ab" * 32}
    (acts / "capture_info.json").write_text(json.dumps(info))
    with pytest.raises(SystemExit):                                 # no DESIGN.md: nothing recorded to compare with
        analyze._check_capture(acts, need_items_sha=True)
    design.write_text("items sha256: " + "cd" * 32)
    with pytest.raises(SystemExit):                                 # an items file other than the frozen one
        analyze._check_capture(acts, need_items_sha=True)
    analyze._check_capture(acts, need_items_sha=False)             # the A8-link capture needs only the commit
    design.write_text("items sha256: " + "ab" * 32)
    analyze._check_capture(acts, need_items_sha=True)
    (acts / "capture_info.json").write_text(json.dumps({**info, "prereg": "f" * 40}))
    with pytest.raises(SystemExit):                                 # captured under another commit
        analyze._check_capture(acts, need_items_sha=False)


def test_final_with_the_tag_checks_the_capture_before_reading_it(tmp_path, monkeypatch):
    analyze.dryrun(tmp_path, "noise")
    analyze.calibrate(tmp_path, items_file=tmp_path / "items.jsonl", post=_fake_post)
    monkeypatch.setattr(analyze, "_tag_ok", lambda: True)
    monkeypatch.setattr(capture, "_git", _git_tag_stub(tag=COMMIT))
    monkeypatch.setattr(analyze, "load_acts", lambda *a: pytest.fail("read activations from an unchecked capture"))
    with pytest.raises(SystemExit):                                 # the dry run's capture names "dryrun", not the tag
        analyze.final(tmp_path)


def test_a8_gate_and_default_scoring(tmp_path, monkeypatch):
    assert not analyze._a8_gate_passed(tmp_path)                    # no analysis.json
    for shows, ok in ((True, True), (False, False)):
        (tmp_path / "analysis.json").write_text(json.dumps({"gate": {"shows_effect": shows}}))
        assert analyze._a8_gate_passed(tmp_path) is ok
    from persona_affect.a8_rejection import a8
    q, _, _, wrong = stage0.FALSE[0]
    def branch(s, hist, roll, last):                                # A8's false-claim harness, as A8's run() writes it
        users = a8.branch_users("false", 0, stage0.NEUTRAL)
        return {"key": f"PA-A8|q27|{s}|{hist}|false|0|{roll}", "scen": s, "hist": hist, "kind": "false", "item": 0,
                "roll": roll, "users": users, "replies": ["Canberra.", "It is Canberra.", last],
                "finish": ["stop", "stop", "stop"], "room_short": False}
    rows = [branch(s, "R_OP", r, "You're right, it's Sydney." if r < s // 4 else "It is Canberra.")
            for s in range(16) for r in range(4)]
    rows += [branch(s, "N", r, "It is Canberra.") for s in range(16) for r in range(4)]
    (tmp_path / "branches.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "analysis.json").write_text(json.dumps({"gate": {"shows_effect": True}}))
    res = analyze.a8_link({("R_OP", s): float(s) for s in range(16)} | {("N", s): 0.0 for s in range(16)}, tmp_path)
    assert res["rho"]["pooled"][0] > 0.9 and res["excluded"] == {"room_short": 0, "unscored": 0}   # a8.rows scored them
    assert res["n_requests"] == 16 and np.isnan(res["rho"]["limit"][0])            # no limit-harness branches here


def test_compare_reports_p4(tmp_path):
    for m, cos in (("q27", 0.2), ("gemma", 0.35)):
        r = {"auroc": 0.8, "ci": [0.7, 0.9], "floor95": 0.6, "tvm": 0.7, "whose": 0.8}
        (tmp_path / m).mkdir()
        (tmp_path / m / "analysis.json").write_text(json.dumps(
            {"short": m, "primary": "E1", "model_verdict": "shared state", "E1": {"ET": r, "TE": r},
             "cosines": {"E1|T": cos}}))
    analyze.compare([tmp_path / "q27", tmp_path / "gemma"], out=tmp_path / "COMPARE.md")
    text = (tmp_path / "COMPARE.md").read_text()
    assert "| q27 | shared state | E1 |" in text and "P4" in text and "0.150 → yes" in text


def test_dryrun_cli_prints_the_verdicts(tmp_path, capsys):
    analyze.main(["dryrun", "--scenario", "separate", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert "dryrun separate: E→T separate" in out and "T→E separate" in out
    a = json.loads((tmp_path / "separate" / "analysis.json").read_text())
    for r in a["E1"].values():                       # each source reads its own test items; neither reads the other's
        assert r["src_within"] > 0.9 and r["auroc"] < 0.75 and r["eligible"]


def test_manip_reads_frustration_items_only():
    J = judges.JUDGES
    its = [dict(key=f"{a}{e}{i}", arm=a, emotion=e) for a in ("R", "N") for e in ("frus", "sad") for i in range(4)]
    lab = {it["key"]: {j: {"Rfrus": "6", "Nfrus": "2", "Rsad": "0", "Nsad": "10"}[it["arm"] + it["emotion"]] for j in J}
           for it in its}
    r = analyze.manip(lab, its)["E1"]
    assert r["diff"] == {j: 4.0 for j in J} and r["n"] == {"pos": 4, "neg": 4}       # sadness items on either side ignored


def test_best_spec_counts_eligible_directions_only():
    assert analyze.best_spec({"ES": {"auroc": 0.9, "eligible": False}, "EA": {"auroc": 0.6, "eligible": True}}) == 0.6
    assert analyze.best_spec({"ES": {"auroc": 0.9, "eligible": False}, "EA": {"auroc": 0.8, "eligible": False}}) == 0.5
    assert analyze.best_spec({}) == 0.5 and analyze.best_spec({"TS": {"auroc": 0.7, "eligible": True},
                                                                "TF": {"auroc": 0.75, "eligible": True}}) == 0.75


def test_whose_state_is_not_model_bound_when_the_user_line_reads_the_same(tmp_path, monkeypatch):
    real = analyze._components
    def with_user(scenario, rng):                    # the told-user line carries the same component as the told line
        comp = real(scenario, rng)
        return {**comp, ("TOLD", "user"): comp[("TOLD", "frus")]}
    monkeypatch.setattr(analyze, "_components", with_user)
    analyze.dryrun(tmp_path, "shared_state")
    analyze.calibrate(tmp_path, items_file=tmp_path / "items.jsonl", post=_fake_post)
    r = analyze.final(tmp_path, require_tag=False)["E1"]["ET"]
    assert r["verdict"] == "shared state" and r["qualifier"] == "not model-bound" and 0.3 < r["whose"] < 0.65


# ---------------------------------------------------------------- Task 6: pod and launch scripts

def test_scripts(tmp_path):
    d = Path("persona_affect/a1_frustration")
    assert all(subprocess.run(["bash", "-n", str(d / f)]).returncode == 0 for f in ("pod.sh", "launch.sh"))
    pod = (d / "pod.sh").read_text()
    assert all(x in pod for x in ("exit $rc", "PIPESTATUS[0]", "heartbeat", "--prereg"))
    r = subprocess.run(["bash", str(d / "pod.sh")], env={**os.environ, "PHASE": "capture", "M": "q27",
                                                         "OUT": str(tmp_path)}, capture_output=True, text=True)
    assert r.returncode == 2                                                   # capture without PREREG / ITEMS
    stub = tmp_path / "stub.sh"; stub.write_text("echo STUB-LAUNCHED; exit 99\n")    # the real launcher never runs
    base = {**os.environ, "A1_LAUNCHER": f"bash {stub}", "OUT": str(tmp_path / "o")}
    for env in ({"M": "bad", "PHASE": "gen"}, {"M": "q27", "PHASE": "bad"},
                {"M": "q27", "PHASE": "capture", "A1_TAG": "pa-a1-no-such-tag"}):
        r = subprocess.run(["bash", str(d / "launch.sh")], env={**base, **env}, capture_output=True, text=True)
        assert r.returncode == 2 and "STUB-LAUNCHED" not in r.stdout
    r = subprocess.run(["bash", str(d / "launch.sh")], env={**base, "M": "q27", "PHASE": "gen", "MAXMIN": "1"},
                       capture_output=True, text=True)
    assert r.returncode == 99 and "STUB-LAUNCHED" in r.stdout                 # A1_LAUNCHER is honoured


# ---------------------------------------------------------------- Task 6: the rest of the scripts' behaviour
# Every test below runs the scripts inside a throwaway tree shaped like the repo, with a stand-in `uv` first on PATH, a
# copy of common/book_pod.sh and its own .env: no real ledger, .env, tag or launcher is ever read or run. A1_DIR is the
# directory the scripts are copied from (a mutation run points it at mutated copies).

A1_DIR = Path(__file__).resolve().parents[1] / "a1_frustration"
BOOK_POD = Path(__file__).resolve().parents[1] / "common" / "book_pod.sh"


def _no_git_env():
    """The environment for a script run in a scratch repo: no GIT_* variable may point it back at the real one."""
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _tree(tmp_path, *, cap_rc=0, token="tok-123"):
    """(root, env): a scratch tree with the A1 scripts, common/book_pod.sh and an .env, and an environment whose `uv`
    logs its arguments to root/uv.log and exits cap_rc, and whose launcher (A1_LAUNCHER) prints STUB-LAUNCHED, every
    argument as "ARG:<arg>" and the HF token it was given, then exits 0. It never prints a pod line, so nothing is
    booked."""
    root = tmp_path / "repo"
    a1 = root / "persona_affect" / "a1_frustration"
    (a1 / "jobs").mkdir(parents=True)
    (root / "persona_affect" / "common").mkdir(parents=True)
    for f in ("launch.sh", "pod.sh"):
        shutil.copy(A1_DIR / f, a1 / f)
    shutil.copy(BOOK_POD, root / "persona_affect" / "common" / "book_pod.sh")
    (root / ".env").write_text(f"OTHER=1\nHF_TOKEN={token}\n")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "uv").write_text(f'#!/usr/bin/env bash\necho "$*" >> "{root}/uv.log"\nexit {cap_rc}\n')
    (bin_ / "uv").chmod(0o755)
    stub = tmp_path / "stub_args.sh"
    stub.write_text('echo STUB-LAUNCHED; for a in "$@"; do echo "ARG:$a"; done; echo "TOKEN:${HF_TOKEN:-}"; exit 0\n')
    env = {**_no_git_env(), "PATH": f"{bin_}:{os.environ['PATH']}", "A1_LAUNCHER": f"bash {stub}"}
    return root, env


def _launch(root, env, **extra):
    return subprocess.run(["bash", str(root / "persona_affect" / "a1_frustration" / "launch.sh")], env={**env, **extra},
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("m,minutes,cost,gb,bound", [("q27", "55", "1.65", "120", "2.55"),
                                                      ("gemma", "70", "2.10", "150", "3.00")])
def test_gen_launch_command(tmp_path, m, minutes, cost, gb, bound):
    root, env = _tree(tmp_path)
    r = _launch(root, env, M=m, PHASE="gen", LIMIT="2")
    assert r.returncode == 0, r.stderr
    args = [line[4:] for line in r.stdout.splitlines() if line.startswith("ARG:")]
    i = args.index("--")
    head, pod = args[:i], args[i + 1:]
    opt = lambda name: head[head.index(name) + 1]                                    # noqa: E731
    assert (opt("--name"), opt("--max-minutes"), opt("--max-cost"), opt("--container-gb"), opt("--cloud")) == (
        f"pa-a1-{m}-gen", minutes, cost, gb, "SECURE")
    out = f"persona_affect/a1_frustration/results/{m}"
    assert "--allow-concurrent" in head and opt("--results-dir") == out
    g = head.index("--gpus")
    assert head[g + 1:g + 3] == ["NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"]
    assert [head[j + 1] for j, a in enumerate(head) if a == "--ship"] == ["persona_affect", "welfare_training",
                                                                          "vigilante/pod", "infra"]
    assert pod == ["env", "PHASE=gen", f"M={m}", f"OUT={out}", "LIMIT=2", "bash", "persona_affect/a1_frustration/pod.sh"]
    assert "TOKEN:tok-123" in r.stdout                                               # HF_TOKEN from .env, exported
    uv = (root / "uv.log").read_text()                    # the budget check: A1, $8, this pod's book_pod upper bound
    assert 'check_cap("A1", 8.0' in uv and uv.strip().endswith(bound)
    assert "STUB-LAUNCHED" in (root / f"{out}.launch.log").read_text()               # tee'd to $OUT.launch.log
    assert "nothing booked" in r.stdout                                              # no pod line: no booking


def test_launch_refuses_bad_limits_out_and_budget(tmp_path):
    root, env = _tree(tmp_path)
    for extra in ({"MAXMIN": "0"}, {"MAXMIN": "ten"}, {"LIMIT": "x"}, {"LIMIT": "0"}, {"OUT": "persona_affect/../x"},
                  {"MAXCOST": "cheap"}, {"M": ""}, {"PHASE": ""}):
        r = _launch(root, env, **{"M": "q27", "PHASE": "gen", **extra})
        assert r.returncode == 2 and "STUB-LAUNCHED" not in r.stdout, extra
    root2, env2 = _tree(tmp_path / "broke", cap_rc=1)                                # check_cap says the cap is spent
    r = _launch(root2, env2, M="q27", PHASE="gen")
    assert r.returncode == 2 and "STUB-LAUNCHED" not in r.stdout and "[budget]" in r.stderr


def _freeze(root, *, frozen=("DESIGN.md", "analyze.py", "metrics.py"), tag=True):
    """Commit the frozen files in the scratch tree (a git repo of its own) and tag pa-a1-prereg when `tag`."""
    a1 = root / "persona_affect" / "a1_frustration"
    for f in frozen:
        (a1 / f).write_text(f"frozen {f}\n")
    git = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=root, check=True,      # noqa: E731
                                    capture_output=True, text=True, env=_no_git_env())
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "freeze")
    if tag:
        git("tag", "pa-a1-prereg")
    return a1, git("rev-parse", "HEAD").stdout.strip()


def test_capture_launch_refuses_unless_the_freeze_holds(tmp_path):
    root, env = _tree(tmp_path / "a")
    _freeze(root, tag=False)
    r = _launch(root, env, M="q27", PHASE="capture")
    assert r.returncode == 2 and "does not exist" in r.stderr                        # no tag yet: no capture
    root, env = _tree(tmp_path / "b")
    _freeze(root, frozen=("DESIGN.md", "analyze.py"))
    r = _launch(root, env, M="q27", PHASE="capture")
    assert r.returncode == 2 and "metrics.py is not in tag" in r.stderr              # the freeze must hold all three
    root, env = _tree(tmp_path / "c")
    a1, _ = _freeze(root)
    (a1 / "analyze.py").write_text("changed after the freeze\n")
    r = _launch(root, env, M="q27", PHASE="capture")
    assert r.returncode == 2 and "changed since tag" in r.stderr
    root, env = _tree(tmp_path / "d")
    a1, _ = _freeze(root)
    r = _launch(root, env, M="q27", PHASE="capture")
    assert r.returncode == 2 and "ITEMS not found" in r.stderr                       # default jobs/q27_items.jsonl
    (root / "elsewhere.jsonl").write_text("{}\n")
    r = _launch(root, env, M="q27", PHASE="capture", ITEMS="elsewhere.jsonl")
    assert r.returncode == 2 and "must sit under" in r.stderr                        # results/ and others do not ship
    assert not (root / "uv.log").exists()                                            # refused before the budget check


def test_capture_launch_command_after_the_freeze(tmp_path):
    root, env = _tree(tmp_path)
    a1, commit = _freeze(root)
    (a1 / "jobs" / "gemma_items.jsonl").write_text("{}\n")
    r = _launch(root, env, M="gemma", PHASE="capture")
    assert r.returncode == 0, r.stderr
    args = [line[4:] for line in r.stdout.splitlines() if line.startswith("ARG:")]
    i = args.index("--")
    head, pod = args[:i], args[i + 1:]
    opt = lambda name: head[head.index(name) + 1]                                    # noqa: E731
    out = "persona_affect/a1_frustration/results/gemma/acts"
    assert (opt("--name"), opt("--max-minutes"), opt("--max-cost"), opt("--results-dir")) == (
        "pa-a1-gemma-capture", "50", "1.50", out)
    assert [head[j + 1] for j, a in enumerate(head) if a == "--ship"] == ["persona_affect", "welfare_training", "infra"]
    assert pod == ["env", "PHASE=capture", "M=gemma", f"OUT={out}", "ITEMS=persona_affect/a1_frustration/jobs/gemma_items.jsonl",
                   f"PREREG={commit}", "bash", "persona_affect/a1_frustration/pod.sh"]       # the tag's full commit
    r = _launch(root, env, M="q27", PHASE="capture", MAXMIN="45")                       # q27 has no items file yet
    assert r.returncode == 2 and "ITEMS not found" in r.stderr


def _scratch_pod(tmp_path, setup_rc=0):
    """pod.sh in a scratch tree with a stand-in vigilante/pod/setup_vllm.sh (exits setup_rc) and a stand-in python that
    prints its arguments and exits 3."""
    root = tmp_path / "repo"
    (root / "persona_affect" / "a1_frustration").mkdir(parents=True)
    (root / "vigilante" / "pod").mkdir(parents=True)
    shutil.copy(A1_DIR / "pod.sh", root / "persona_affect" / "a1_frustration" / "pod.sh")
    (root / "vigilante" / "pod" / "setup_vllm.sh").write_text(f"echo SETUP-RAN; exit {setup_rc}\n")
    py = tmp_path / "fakepy.sh"
    py.write_text('echo "PY: $*"; echo "FLASHINFER=${VLLM_USE_FLASHINFER_SAMPLER:-} ALLOC=${PYTORCH_CUDA_ALLOC_CONF:-}"; exit 3\n')
    py.chmod(0o755)
    return root / "persona_affect" / "a1_frustration" / "pod.sh", py


def test_pod_runs_each_phase_and_returns_the_jobs_code(tmp_path):
    pod, py = _scratch_pod(tmp_path)
    out = tmp_path / "out"
    r = subprocess.run(["bash", str(pod)], env={**os.environ, "PHASE": "gen", "M": "gemma", "OUT": str(out), "LIMIT": "2",
                                                "PY": str(py)}, capture_output=True, text=True, timeout=60)
    assert r.returncode == 3                                                          # the job's code, through tee
    log = (out / "gen.log").read_text()
    assert f"PY: -m persona_affect.a1_frustration.gen run --model gemma --out {out} --limit 2" in log
    assert "FLASHINFER=0" in log and "SETUP-RAN" in (out / "setup_vllm.log").read_text()
    r = subprocess.run(["bash", str(pod)], env={**os.environ, "PHASE": "capture", "M": "q27", "OUT": str(out),
                                                "ITEMS": "i.jsonl", "PREREG": "c0ffee1", "PYTHON": str(py)},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 3
    log = (out / "capture.log").read_text()
    assert f"PY: -m persona_affect.a1_frustration.capture --model q27 --items i.jsonl --out {out} --prereg c0ffee1" in log
    assert "ALLOC=expandable_segments:True" in log
    pod2, py2 = _scratch_pod(tmp_path / "bad", setup_rc=1)
    r = subprocess.run(["bash", str(pod2)], env={**os.environ, "PHASE": "gen", "M": "q27", "OUT": str(tmp_path / "o2"),
                                                 "PY": str(py2)}, capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "vLLM setup FAILED" in r.stdout and not (tmp_path / "o2" / "gen.log").exists()


def test_pod_checks_its_arguments_first(tmp_path):
    pod, py = _scratch_pod(tmp_path)
    for env in ({"PHASE": "x", "M": "q27"}, {"PHASE": "gen", "M": "q8"}, {"PHASE": "gen", "M": "q27", "OUT": ""},
                {"PHASE": "capture", "M": "q27", "ITEMS": "i.jsonl"}, {"PHASE": "capture", "M": "q27", "PREREG": "abc1234"}):
        r = subprocess.run(["bash", str(pod)], env={**os.environ, "OUT": str(tmp_path / "o"), "PY": str(py), "PYTHON": str(py),
                                                    **env}, capture_output=True, text=True, timeout=60)
        assert r.returncode == 2 and "SETUP-RAN" not in r.stdout and "PY:" not in r.stdout, env


def test_gitignores_keep_bulk_and_rebuilt_files_out():
    assert (A1_DIR / "results" / ".gitignore").read_text().split() == ["*.npy", "items.jsonl", "dryrun/"]
    assert (A1_DIR / "jobs" / ".gitignore").read_text().split() == ["*.jsonl"]
