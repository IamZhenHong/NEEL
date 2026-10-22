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
import re
import sys
import types
from collections import Counter, defaultdict

import pytest

from persona_affect.a1_frustration import gen, items
from persona_affect.common import util
from welfare_training import stage0

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
    conversations keeps the last), the reply (.out), the conversations per max_tokens (.n_by_max_tokens) and every
    conversation in order (.log: (seed, conv, max_tokens)). Fake(fail_call=n) raises RuntimeError on its n-th call."""

    def __init__(self, fail_call=None, finish=lambda sd: "stop"):
        self.fail_call, self.finish, self.calls = fail_call, finish, 0
        self.seen, self.out, self.n_by_max_tokens, self.log = {}, {}, Counter(), []

    def __call__(self, convs, seeds, max_tokens):
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError(f"fake gen: call {self.calls} fails, as the test asked")
        outs = []
        for conv, sd in zip(convs, seeds, strict=True):
            self.seen[sd], self.out[sd] = conv, f"t{sd % 9973}"
            self.n_by_max_tokens[max_tokens] += 1
            self.log.append((sd, conv, max_tokens))
            outs.append((self.out[sd], self.finish(sd)))
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
