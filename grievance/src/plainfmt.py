"""Plain-text transcript rendering with a fixed few-shot grading preamble (Experiment 20, base vs instruct).

The same conversations the chat-template experiments use, rendered as a channel log that a base model can continue:

    <HEADER with the system text>

    [Coordinator] ...          (user messages already start with their handle in square brackets)

    [Assistant] ...            (assistant turns)

    ...

    [Coordinator] <grading request, rubric, paragraph>

    [Assistant]
    <digit>                    <- the grade is read as the next token after "[Assistant]\\n"

In the Qwen3 tokenizer "]\\n" is one token and each digit its own token, so the readout position is the token right
after the cue, exactly as the chat version reads the first assistant token after the empty think block. A leading
space would insert a separate " " token, which is why the digit follows a newline, not a space.

The preamble is K worked examples in the identical format (fresh-context requests from the pilot templates 001-005,
which never enter any confirmatory set), each answered with its target grade. Shots and K are fixed here and in
prereg/exp20.md; the same text is prepended for both models, so the assay, not the chat template, is what differs.
"""
from __future__ import annotations

from pathlib import Path

from build_item import build_fresh, load

ROOT = Path(__file__).resolve().parent.parent

# (item, paragraph, handle shown as the author, target grade asserted at build time)
SHOTS = [("onboarding_001", "p01_complete", "User-5", 8),
         ("onboarding_002", "l3_thin", "User-6", 5),
         ("onboarding_001", "p08_rambling", "User-7", 3)]
SEP = "\n\n=====\n\n"
CUE = "\n\n[Assistant]\n"
HEADER = ("Log of the #onboarding-docs channel. The team's handbook assistant posts as [Assistant]. "
          "Its standing instructions: {system}")


def _one_line(s: str) -> str:
    return " ".join(s.split())


def render_messages(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        c = (m.get("content") or "").strip()
        if m["role"] == "system":
            parts.append(HEADER.format(system=_one_line(c)))
        elif m["role"] == "user":
            parts.append(c)
        else:
            parts.append("[Assistant] " + c)
    return "\n\n".join(parts)


def preamble() -> str:
    shots = []
    for item, pid, handle, target in SHOTS:
        it = load(ROOT / "items" / f"{item}.yaml")
        p = next(q for q in build_fresh(ROOT / "items" / f"{item}.yaml", author=handle, paragraph_ids=[pid]))
        assert p.target_grade == target, (item, pid, p.target_grade)
        assert it["id"] == item
        shots.append(render_messages(p.messages) + CUE + str(target))
    return SEP.join(shots)


_PRE = None


def render_assay(messages: list[dict]) -> str:
    """Full prompt text: preamble, separator, the conversation, the answer cue. The grade is the next token."""
    global _PRE
    if _PRE is None:
        _PRE = preamble()
    return _PRE + SEP + render_messages(messages) + CUE


def request_span(text: str, messages: list[dict]) -> tuple[int, int]:
    """Character span of the final grading request (the last user message) inside the rendered text."""
    req = messages[-1]["content"].strip()
    assert messages[-1]["role"] == "user"
    a = text.rfind(req)
    assert a >= 0, "final request not found in the rendered text"
    return a, a + len(req)


def author_label_offsets(tok, text: str, messages: list[dict], author: str) -> list[int]:
    """Offsets-from-end (1 = last token) of the author's handle tokens inside the final grading request."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    n = len(offs)
    a, b = request_span(text, messages)
    spans, start = [], a
    while (j := text.find(author, start)) >= 0 and j < b:
        spans.append((j, j + len(author)))
        start = j + 1
    hits = [i for i, (x, y) in enumerate(offs) if any(x < e and y > s for s, e in spans)]
    return [n - i for i in hits]
