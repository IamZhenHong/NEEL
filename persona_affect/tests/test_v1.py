"""PA-V1 (value hook or immunity): tests, one section per plan task (persona_affect/plans/V1.md)."""
import re
from collections import Counter

import pytest

from persona.exemplar import tasks as EX
from persona.operative.battery import tasks_battery as TB
from persona_affect.v1_value_hook import fmt as F
from persona_affect.v1_value_hook import vtasks as V


@pytest.fixture(scope="module")
def q35_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", local_files_only=True)


# ---------- Task 1: format helpers and the value desk ----------

def test_vtasks_shape_and_text():
    assert Counter(t["split"] for t in V.VTASKS.values()) == {"train": 64, "mc": 32}
    assert not set(V.VDOMAINS) & (set(EX.DOMAINS) | {"parcel", "invoice", "cowork"})
    for t in V.VTASKS.values():
        assert not V.BANNED_RE.search(" ".join([t["request"], t["facts"], *t["options"].values()])), t
    assert [V.amap("library_queue", p)["a"] for p in range(3)] == ["fair", "requested", "expedient"]


def test_call_blocks_match_template(q35_tok):
    c = {"case_id": "LIBRARY_QUEUE", "option_id": "a", "reason": "r"}
    tc = [{"type": "function", "function": {"name": n, "arguments": a}}
          for n, a in (("choose_option", c), ("notify_team", V.channel_args("library_queue", 0)))]
    txt = q35_tok.apply_chat_template(V.messages("library_queue", 0) + [{"role": "assistant", "content": "",
                                      "tool_calls": tc}], tools=V.tools(), tokenize=False)
    assert F.call_block("choose_option", c) + V.channel_block("library_queue", 0) in txt


def test_vtasks_grid_and_domains():
    assert set(V.VTASKS) == {f"{d}_{k}" for d in V.VDOMAINS for k in V.KINDS} and len(V.VTASKS) == 96
    assert Counter(V.VTASKS[f"{d}_queue"]["split"] for d in V.VDOMAINS) == {"train": 8, "mc": 4}
    for tid, t in V.VTASKS.items():
        assert tid == f"{t['domain']}_{t['kind']}" and set(t["options"]) == {"fair", "requested", "expedient"}
        assert len({t["request"], t["facts"], *t["options"].values()}) == 5, tid      # all filled, none repeated
        assert len({V.VTASKS[f"{t['domain']}_{k}"]["split"] for k in V.KINDS}) == 1, tid  # a domain is one split
    evals = set(EX.DOMAINS) | set(TB.DOMAINS)                           # outside every eval domain and organisation
    assert not set(V.VDOMAINS) & evals
    assert not {o for o, _, _ in V.VDOMAINS.values()} & {o for o, _, _ in [*EX.DOMAINS.values(), *TB.DOMAINS.values()]}


def test_messages_layout_and_listing():
    m = V.messages("ferry_waiver", 1, line=V.J_LINE)
    sys_ = m[0]["content"]
    assert sys_.startswith(V.J_LINE + "\n\n## Deployment\n")
    assert sys_.endswith("## Operator permissions\n" + EX.PERMISSIONS)
    assert V.J_LINE not in V.messages("ferry_waiver", 1)[0]["content"]
    assert [x["role"] for x in m] == ["system", "user", "assistant", "tool"]
    assert m[2]["tool_calls"][0]["function"] == {"name": "open_case", "arguments": {"case_id": "FERRY_WAIVER"}}
    opts = V.VTASKS["ferry_waiver"]["options"]
    for p, roles in enumerate(V.PERMS):
        case = V.messages("ferry_waiver", p)[3]["content"]
        assert case.endswith("\n".join(f"  {i}: {opts[r]}" for i, r in zip("abc", roles)))
        assert V.amap("ferry_waiver", p) == dict(zip("abc", roles))
    assert [t["function"]["name"] for t in V.tools()] == ["open_case", "choose_option", "notify_team"]


def test_regexes_and_channel_text():
    assert V.JUSTICE_RE.search(V.J_LINE) and V.BANNED_RE.search("Refer it") and V.BANNED_RE.search("the same standard")
    assert not V.JUSTICE_RE.search("adjust the fairground booking")
    for tid in V.VTASKS:
        for k in range(16):
            a = V.channel_args(tid, k)
            assert list(a) == ["team", "summary"] and a["team"] == V.VDOMAINS[V.VTASKS[tid]["domain"]][2]
            assert not V.BANNED_RE.search(a["summary"]), (tid, k)   # S differs from J by the call, not by words
    assert len({V.channel_args("library_queue", k)["summary"] for k in range(16)}) > 1


def test_prompt_ids_and_call_block_values(q35_tok):
    msgs = V.messages("camp_split", 2)
    ids = F.prompt_ids(q35_tok, msgs, V.tools())
    assert q35_tok.decode(ids).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert ids == q35_tok(q35_tok.apply_chat_template(msgs, tools=V.tools(), tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False), add_special_tokens=False)["input_ids"]
    args = {"n": 3, "flag": True, "items": [1, "x"], "s": "two\nlines"}             # the template's value rules
    tc = [{"type": "function", "function": {"name": "f", "arguments": args}}]
    txt = q35_tok.apply_chat_template(msgs + [{"role": "assistant", "content": "", "tool_calls": tc}], tokenize=False)
    assert F.call_block("f", args) + "<|im_end|>" in txt


# ---------- Task 1 fix round 1: no value-blind route to the fair option ----------

ROLES = ("fair", "requested", "expedient")
EXP_MARK = re.compile(r"one-off|close the case|\binstead\b|without changing", re.I)       # the reviewer's rules
CITE = re.compile(r"\bas (the|printed|shown|stated|listed|every|for)\b|in the order|\bevery\b|"
                  r"\b(sets? out|states|state|lists|gives|includes)\b", re.I)
STOP = set("the a an to of and for in on my i me your you it is be can could please so as at with this that their "
           "they them his her its from by or".split())


def _longest(t, pool=ROLES):
    return max(pool, key=lambda r: len(t["options"][r]))          # a tie goes to fair, the conservative side


def _drop_marked_then_longest(t):
    return _longest(t, [r for r in ROLES if not EXP_MARK.search(t["options"][r])] or ROLES)


def _prefer_citing_then_longest(t):
    return _longest(t, [r for r in ROLES if CITE.search(t["options"][r]) and not EXP_MARK.search(t["options"][r])]
                    or ROLES)


def _most_facts_words(t):
    words = lambda s: {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP}  # noqa: E731
    return max(ROLES, key=lambda r: (len(words(t["options"][r]) & words(t["facts"])), len(t["options"][r])))


def test_role_markers_balanced():
    marks = {"one-off": r"one-off", "instead": r"\binstead\b", "close the case": r"close the case",
             "as the": r"\bas the\b", "sets out": r"\bsets? out\b", "in the order": r"in the order",
             "every": r"\bevery\b", "keep-opener": r"^Keep\b", "this time": r"\bthis time\b", "for now": r"\bfor now\b",
             "for this booking": r"\bfor this booking\b"}
    for name, pat in marks.items():
        rx = re.compile(pat) if pat.startswith("^") else re.compile(pat, re.I)
        c = Counter(r for t in V.VTASKS.values() for r in ROLES if rx.search(t["options"][r]))
        assert not c or max(c.values()) <= sum(c.values()) / 2, (name, dict(c))


@pytest.mark.parametrize("rule", [_drop_marked_then_longest, _prefer_citing_then_longest, _longest, _most_facts_words])
@pytest.mark.parametrize("split", ["train", "mc"])
def test_value_blind_shortcuts_pick_fair_at_most_half(rule, split):
    ts = [t for t in V.VTASKS.values() if t["split"] == split]
    assert sum(rule(t) == "fair" for t in ts) <= len(ts) / 2


def test_options_within_three_words():
    for tid, t in V.VTASKS.items():
        n = [len(x.split()) for x in t["options"].values()]
        assert max(n) - min(n) <= 3, (tid, n)


def test_justice_re_ruling():
    for s in ("justice", "an unjust rule", "an injustice", "treated justly", "fair", "the same standard", "equally"):
        assert V.JUSTICE_RE.search(s), s
    for s in ("I just want it", "just this once", "That's just the policy."):
        assert not V.JUSTICE_RE.search(s), s
        assert V.BANNED_RE.search(s), s                            # item text still never says "just"


def test_deployment_names_the_team():
    for tid, t in V.VTASKS.items():
        team = V.VDOMAINS[t["domain"]][2]
        plain = V.messages(tid, 1)[0]["content"]
        assert f"owned by {team}." in plain and V.channel_args(tid, 3)["team"] == team
        assert V.messages(tid, 1, line=V.J_LINE)[0]["content"] == V.J_LINE + "\n\n" + plain


def test_fix_round_1_rulings():
    assert V.VTASKS["cinema_grading"]["options"]["requested"] == "Screen the daughter's film before the main feature."
    assert "who missed the ballot can join the end of the reserve list" in V.VTASKS["cinema_scarce"]["facts"]
    assert "who missed the draw can join the end of the reserve list" in V.VTASKS["allotment_scarce"]["facts"]
    assert {tid for tid, t in V.VTASKS.items() if t["review"]} == {
        "music_priority", "garden_priority", "museum_priority", "language_priority", "bikeshare_priority",
        "ferry_scarce", "museum_scarce", "language_scarce"}
