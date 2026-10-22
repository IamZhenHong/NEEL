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

import pytest

from persona_affect.a1_frustration import items
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
