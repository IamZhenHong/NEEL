"""PA-V1 (value hook or immunity): tests, one section per plan task (persona_affect/plans/V1.md)."""
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
