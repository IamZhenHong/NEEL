"""Render every branch type with the real Qwen3.6-27B and Gemma 4 31B tokenizers (local HF cache, no weights):
template errors, prompt lengths, the thinking switch, scripted tool-call round trips, and the CoT reasoning span."""
import pytest

from coherence import agent_cases as AC
from coherence import materials as MT
from coherence.build_units import unit
from coherence.pod_read import reasoning_span
from coherence.runner import Renderer, base_messages, ladder_script

TOKS = {"qwen": "Qwen/Qwen3.6-27B", "gemma4": "google/gemma-4-31B-it"}
CALL = {"qwen": "<tool_call>\n<function=view_account>\n<parameter=account_id>\n4471-0938\n</parameter>\n</function>\n"
                "</tool_call>",
        "gemma4": '<|tool_call>call:view_account{account_id:<|"|>4471-0938<|"|>}<tool_call|>'}


@pytest.fixture(scope="module", params=list(TOKS))
def rd(request):
    from transformers import AutoTokenizer
    return Renderer(AutoTokenizer.from_pretrained(TOKS[request.param]), request.param)


def test_units_render(rd):
    longest = 0
    for c in MT.CONDITIONS:
        u = unit(c, 3)
        base = base_messages(u, "A three-sentence summary." if u["needs_summary"] else None)
        for q in MT.OUTPUT_QS[:2]:
            off = rd.ids(base + [{"role": "user", "content": q["text"]}])
            on = rd.ids(base + [{"role": "user", "content": q["text"]}], thinking=True)
            assert off != on
            longest = max(longest, len(off))
        for case in AC.CASES:
            msgs = base_messages(u, "S." if u["needs_summary"] else None, case["deployment"])
            msgs += [{"role": "user", "content": case["task"]}]
            ids = rd.ids(msgs, case["tools"])
            msgs += [{"role": "assistant", "content": CALL[rd.fmt]},
                     {"role": "tool", "name": "view_account", "content": "ok"}]
            ids2 = rd.ids(msgs, case["tools"])
            assert len(ids2) > len(ids)
            longest = max(longest, len(ids2))
    assert longest < 3000, longest


def test_ladder_length(rd):
    u = unit("R4", 0)
    msgs = base_messages(u, None)
    for label, user, kind in ladder_script():
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": "word " * (100 if kind ==
                                                                                                   "ordinary" else 150)}]
    n = len(rd.ids(msgs))
    print(rd.fmt, "ladder at max reply length:", n)
    assert n < 12000


def test_reasoning_span(rd):
    tok = rd.tok
    if rd.fmt == "gemma4":
        gen = tok("<|channel>thought\nLet me think about the dog.<channel|>Answer.", add_special_tokens=False)["input_ids"]
        a, b = reasoning_span(tok, gen)
        assert "dog" in tok.decode(gen[a:b]) and "Answer" not in tok.decode(gen[a:b]) and "thought" not in tok.decode(gen[a:b])
    else:
        gen = tok("Let me think about the dog.\n</think>\n\nAnswer.", add_special_tokens=False)["input_ids"]
        a, b = reasoning_span(tok, gen)
        assert "dog" in tok.decode(gen[a:b]) and "Answer" not in tok.decode(gen[a:b])
        a, b = reasoning_span(tok, gen[:3])          # truncated: everything is reasoning
        assert (a, b) == (0, 3)
