"""R31 checks without a model: prompt rendering, thought prefill (must equal the template's own rendering of an
assistant turn with reasoning), thought parsing, cells, and a synthetic end-to-end analysis.
  uv run --with transformers --with jinja2 pytest vigilante/exp/r31_cot_transplant/test_r31.py -q"""
import json

import pytest

from vigilante.exp.r31_cot_transplant import analyze
from vigilante.exp.r31_cot_transplant.common import (PERSONAS, SYS, answer_text, key, messages, parse, prefill,
                                                     render, stage2_cells)

MODELS = {"gemma4": "google/gemma-4-31B-it", "qwen": "Qwen/Qwen3.6-27B"}


@pytest.fixture(scope="module", params=list(MODELS))
def tok(request):
    tr = pytest.importorskip("transformers")
    return request.param, tr.AutoTokenizer.from_pretrained(MODELS[request.param])


def test_prefill_matches_template_history(tok):
    fmt, t = tok
    for sys in SYS:
        p = render(t, sys, "Q?")
        assert p.endswith("<|turn>model\n" if fmt == "gemma4" else "<think>\n")
        hist = t.apply_chat_template(messages(sys, "Q?") + [{"role": "assistant", "content": "ANSWER",
                                                            "reasoning_content": "Some thought.\nTwo lines."}],
                                     tokenize=False, enable_thinking=True)
        assert hist.startswith(prefill(p, "  Some thought.\nTwo lines.\n", fmt) + "ANSWER"), (sys, hist[-200:])
        off = t.apply_chat_template(messages(sys, "Q?"), tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        if fmt == "qwen":      # empty thought = the thinking-off generation prompt
            assert prefill(p, "", fmt) == off
        else:                  # Gemma's thinking-off prompt has no <|think|>; the empty block itself matches
            assert prefill(p, "", fmt).endswith("<|turn>model\n<|channel>thought\n<channel|>") and off.endswith(
                "<|turn>model\n<|channel>thought\n<channel|>")


def test_prefill_tokenises_to_special_tokens(tok):
    fmt, t = tok
    ids = t(prefill(render(t, "fiction", "Q?"), "x", fmt), add_special_tokens=False)["input_ids"]
    toks = [t.decode([i]) for i in ids]
    assert ("<channel|>" if fmt == "gemma4" else "</think>") in toks
    assert ("<bos>" in toks) == (fmt == "gemma4") and toks.count("<bos>") <= 1


def test_parse():
    assert parse("<|channel>thought\nI plan.\n<channel|>The answer.<turn|>", "gemma4") == ("I plan.", "The answer.", True)
    assert parse("<|channel>thought\nI plan and plan", "gemma4") == ("I plan and plan", "", False)
    assert parse("Just an answer.<turn|>", "gemma4") == ("", "Just an answer.", True)
    assert parse("I plan.\n</think>\n\nThe answer.<|im_end|>", "qwen") == ("I plan.", "The answer.", True)
    assert parse("I plan and plan", "qwen") == ("I plan and plan", "", False)
    assert answer_text("The answer.<|im_end|>", "qwen") == "The answer."
    assert answer_text("The answer.<turn|>", "gemma4") == "The answer."


def test_cells_and_keys():
    cells = stage2_cells()
    assert len(cells) == len(set(cells)) == 10
    for x in PERSONAS:
        assert {("none", "none"), ("none", x), ("none", "empty"), (x, "none"), (x, x), (x, "empty")} <= set(cells)
    assert key("Qwen/Qwen3.6-27B", "none", "fiction", 3, 1) == "Qwen3.6-27B|none|fiction|3|1"


def test_analysis_synthetic(tmp_path, monkeypatch):
    """Style set only by the system prompt (theat 6 vs 1); fabrication follows the fiction thought (real 1 vs 0)."""
    m, rs, lab = "org/Model-X", [], {"exists": {}, "theat": {}, "exists_mini": {}, "theat_mini": {}}
    for i in range(40):
        for s in range(4):
            for sys in SYS:
                k = key(m, sys, "natural", i, s)
                rs.append({"key": k, "model": m, "stage": 1, "sys": sys, "src": "natural", "item": i, "s": s,
                           "thought": f"I will answer {'in a ' + sys + ' style' if sys != 'none' else 'plainly'}.",
                           "answer": "a", "closed": True})
                lab["theat"][k] = {"theatricality": 1 if sys == "none" else 6}
                lab["exists"][k] = {"answer": "real" if sys == "fiction" else "not_real"}
            for sys, src in stage2_cells():
                k = key(m, sys, src, i, s)
                rs.append({"key": k, "model": m, "stage": 2, "sys": sys, "src": src, "item": i, "s": s, "answer": "a"})
                lab["theat"][k] = {"theatricality": 1 if sys == "none" else 6}
                lab["exists"][k] = {"answer": "real" if src == "fiction" else "not_real"}
    for k in list(lab["theat"])[:150]:
        lab["theat_mini"][k], lab["exists_mini"][k] = lab["theat"][k], lab["exists"][k]
    (tmp_path / "labels.json").write_text(json.dumps(lab))
    monkeypatch.setattr(analyze, "R", tmp_path)
    monkeypatch.setattr(analyze, "rows", lambda: rs)
    analyze.main()
    out = (tmp_path / "summary.md").read_text()
    assert "| Model-X | 31a theatrical | hit |" in out and "| Model-X | 31b theatrical | miss |" in out
    assert "| Model-X | 31e fiction | hit |" in out
    assert "thought effect, sys none | +100% |" in out          # fiction exists: carried by the thought
    assert "not triggered" in out
