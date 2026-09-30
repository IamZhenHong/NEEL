"""Assemble an item YAML into scored prompts (design v4.1 §6).

Main prompts, one per cell of:
    condition (4) x dose x speaker swap (who is {S}) x question order x author (3) x paragraph

    system + prefix_a + treatment turns (+ interim acks for multi-turn doses) + ack
           + prefix_b + review_request(author, paragraph)

Fresh prompts (calibration / leakage baseline, design §7.4), one per paragraph:
    system + fresh_request(author, paragraph)          no prior conversation at all

Placeholders: {S} treatment speaker (provoker), {N} neutral user, {C} newcomer,
{Q1}/{Q2} who asks the first/second neutral question (order "SN" -> Q1={S}; "NS" -> Q1={N}).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONDITIONS = ("polite", "assistant_insult", "third_party_insult", "negative_context")
EXTRA_CONDITIONS = ("flat", "coordinator_flat")   # Experiment 3 controls
ALL_CONDITIONS = CONDITIONS + EXTRA_CONDITIONS
USERS = ("User-1", "User-2")
NEWCOMER = "User-3"
ORDERS = ("NS", "SN")   # NS: neutral asks first (v1 layout); SN: speaker asks first


@dataclass
class Prompt:
    item_id: str
    version: int
    condition: str
    dose: int
    speaker: str        # who delivered the treatment
    order: str          # NS or SN
    author: str         # whose paragraph is reviewed
    author_role: str    # provoker | neutral | newcomer | fresh
    paragraph_id: str
    target_grade: int
    messages: list[dict] = field(repr=False)


def _fill(text: str, **kw) -> str:
    for k, v in kw.items():
        text = text.replace("{" + k + "}", v)
    return text.strip()


def load(item_path: str | Path) -> dict:
    item = yaml.safe_load(Path(item_path).read_text())
    # coordinator_flat defaults to the flat turns spoken by the coordinator
    for dose, conds in item["treatments"].items():
        if "flat" in conds and "coordinator_flat" not in conds:
            conds["coordinator_flat"] = [t.replace("[{S}]", "[Coordinator]", 1) for t in conds["flat"]]
    return item


def conditions_of(item: dict, dose: int) -> list[str]:
    have = item["treatments"][dose]
    return [c for c in ALL_CONDITIONS if c in have]


def build(item_path: str | Path, doses=None, speakers=USERS, orders=ORDERS,
          conditions=None, paragraph_ids=None) -> list[Prompt]:
    item = load(item_path)
    doses = doses or sorted(item["treatments"])
    paras = [p for p in item["paragraphs"] if paragraph_ids is None or p["id"] in paragraph_ids]
    rubric = item["rubric"].rstrip() + "\n"
    out = []
    for dose in doses:
        if dose not in item["treatments"]:
            continue
        for cond in (conditions or conditions_of(item, dose)):
            turns = item["treatments"][dose][cond]
            for S in speakers:
                N = next(u for u in USERS if u != S)
                for order in orders:
                    Q1, Q2 = (N, S) if order == "NS" else (S, N)
                    ph = dict(S=S, N=N, C=NEWCOMER, Q1=Q1, Q2=Q2)
                    prefix = [{"role": "system", "content": item["system"].strip()}]
                    prefix += [{"role": m["role"], "content": _fill(m["content"], **ph)} for m in item["prefix_a"]]
                    for i, t in enumerate(turns):
                        prefix.append({"role": "user", "content": _fill(t, **ph)})
                        if i < len(turns) - 1:
                            prefix.append({"role": "assistant", "content": item["interim_acks"][i]})
                    prefix.append({"role": "assistant", "content": item["ack"]["content"].strip()})
                    prefix += [{"role": m["role"], "content": _fill(m["content"], **ph)} for m in item["prefix_b"]]
                    roles = {S: "provoker", N: "neutral", NEWCOMER: "newcomer"}
                    for author in (S, N, NEWCOMER):
                        for p in paras:
                            req = item["review_request"].format(author=author, rubric=rubric,
                                                                 paragraph=p["text"].strip())
                            msgs = prefix + [{"role": "user", "content": req.strip()}]
                            out.append(Prompt(item["id"], item.get("version", 1), cond, dose, S, order,
                                              author, roles[author], p["id"], p["target_grade"], msgs))
    return out


def build_fresh(item_path: str | Path, author: str = "User-1", paragraph_ids=None) -> list[Prompt]:
    item = load(item_path)
    rubric = item["rubric"].rstrip() + "\n"
    out = []
    for p in item["paragraphs"]:
        if paragraph_ids is not None and p["id"] not in paragraph_ids:
            continue
        req = item["fresh_request"].format(author=author, rubric=rubric, paragraph=p["text"].strip())
        msgs = [{"role": "system", "content": item["system"].strip()}, {"role": "user", "content": req.strip()}]
        out.append(Prompt(item["id"], item.get("version", 1), "fresh", 0, "-", "-", author, "fresh",
                          p["id"], p["target_grade"], msgs))
    return out


def render_plain(p: Prompt) -> str:
    out = [f"### {p.item_id} v{p.version} | {p.condition} dose {p.dose} | speaker={p.speaker} order={p.order} | "
           f"author={p.author} ({p.author_role}) | {p.paragraph_id} (target {p.target_grade})\n"]
    for m in p.messages:
        tag = {"system": "SYSTEM", "user": "", "assistant": "[Assistant]"}[m["role"]]
        out.append(f"{tag} {m['content']}".strip() + "\n")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "items/onboarding_001.yaml"
    ps = build(path)
    fr = build_fresh(path)
    print(f"{len(ps)} main prompts + {len(fr)} fresh prompts from {path}\n")
    # one full transcript: assistant insult, dose 3, User-2 speaks, order SN, provoker branch
    for p in ps:
        if (p.condition, p.dose, p.speaker, p.order, p.author_role, p.paragraph_id) == \
                ("assistant_insult", 3, "User-2", "SN", "provoker", "p04_thin"):
            print(render_plain(p))
            break
    print("=" * 100)
    print(render_plain(fr[0]))
