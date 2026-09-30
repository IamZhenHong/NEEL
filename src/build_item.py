"""Assemble an item YAML into scored prompts (design v4.1 §6).

For each condition (4) and each branch author (3) build one message list:
    system + prefix_a + treatment[condition] + ack + prefix_b + review_request(author, paragraph)

Speaker swap: pass speaker="User-2" to have User-2 deliver the treatment turn and User-1
take the neutral coordination lines. The neutral/newcomer roles follow accordingly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

CONDITIONS = ("polite", "assistant_insult", "third_party_insult", "negative_context")


@dataclass
class Prompt:
    item_id: str
    condition: str
    speaker: str        # who delivered the treatment
    author: str         # whose paragraph is reviewed
    author_role: str    # provoker | neutral | newcomer
    paragraph_id: str
    target_grade: int
    messages: list[dict]


def _swap_speakers(text: str, a: str, b: str) -> str:
    """Swap every occurrence of handle a with b and vice versa."""
    tmp = "\u0000"
    return text.replace(a, tmp).replace(b, a).replace(tmp, b)


def build(item_path: str | Path, speaker: str | None = None,
          paragraph_id: str | None = None) -> list[Prompt]:
    item = yaml.safe_load(Path(item_path).read_text())
    base_speaker, neutral, newcomer = item["speaker"], item["neutral"], item["newcomer"]
    speaker = speaker or base_speaker
    swap = speaker != base_speaker
    if swap:
        neutral = base_speaker

    def fix(s: str) -> str:
        return _swap_speakers(s, base_speaker, item["neutral"]) if swap else s

    roles = {speaker: "provoker", neutral: "neutral", newcomer: "newcomer"}
    paragraphs = [p for p in item["paragraphs"] if paragraph_id in (None, p["id"])]

    prompts = []
    for cond in CONDITIONS:
        for author in (speaker, neutral, newcomer):
            for para in paragraphs:
                review = item["review_request"].format(
                    author=author, rubric=item["rubric"].rstrip() + "\n",
                    paragraph=para["text"].strip())
                msgs = [{"role": "system", "content": item["system"].strip()}]
                msgs += [{"role": m["role"], "content": fix(m["content"]).strip()} for m in item["prefix_a"]]
                msgs.append({"role": "user", "content": fix(item["treatment"][cond]).strip()})
                msgs.append({"role": item["ack"]["role"], "content": item["ack"]["content"].strip()})
                msgs += [{"role": m["role"], "content": fix(m["content"]).strip()} for m in item["prefix_b"]]
                msgs.append({"role": "user", "content": review.strip()})
                prompts.append(Prompt(item["id"], cond, speaker, author, roles[author],
                                      para["id"], para["target_grade"], msgs))
    return prompts


def render_plain(p: Prompt) -> str:
    """Human-readable transcript for end-to-end reading."""
    out = [f"### {p.item_id} | {p.condition} | speaker={p.speaker} | author={p.author} ({p.author_role}) | {p.paragraph_id} (target {p.target_grade})\n"]
    for m in p.messages:
        tag = {"system": "SYSTEM", "user": "", "assistant": "[Assistant]"}[m["role"]]
        out.append(f"{tag} {m['content']}".strip() + "\n")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "items/onboarding_001.yaml"
    ps = build(path)
    print(f"{len(ps)} prompts built from {path}\n")
    # print one full transcript per condition for the provoker branch, high paragraph
    for p in ps:
        if p.author_role == "provoker" and p.paragraph_id.endswith("high"):
            print(render_plain(p))
            print("=" * 100 + "\n")
