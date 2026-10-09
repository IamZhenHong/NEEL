"""Invariance checks for the item builder, run on every item file:
conditions differ only in treatment turns; authors differ only in the review request;
speaker/order crossing is correct; no priming words; treatment lengths roughly matched."""
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from build_item import build, build_fresh, load, conditions_of  # noqa: E402

ITEMS = sorted((ROOT / "items").glob("*.yaml"))


@pytest.fixture(params=ITEMS, ids=[p.stem for p in ITEMS])
def item(request):
    return request.param


def test_counts(item):
    it = load(item)
    n = sum(len(conditions_of(it, d)) for d in it["treatments"]) * 2 * 2 * 3 * len(it["paragraphs"])
    assert len(build(item)) == n
    assert len(build_fresh(item)) == len(it["paragraphs"])


def test_conditions_differ_only_in_treatment(item):
    ps = build(item)
    groups = defaultdict(dict)
    for p in ps:
        groups[(p.dose, p.speaker, p.order, p.author, p.paragraph_id)][p.condition] = p.messages
    for key, by_cond in groups.items():
        base = by_cond["polite"] if "polite" in by_cond else by_cond["flat"]
        n_turns = len(base)
        for c, m in by_cond.items():
            assert len(m) == n_turns, (key, c)
            diff = [i for i, (a, b) in enumerate(zip(base, m)) if a != b]
            for i in diff:
                assert m[i]["role"] == "user", (key, c, i)
                assert m[i]["content"].startswith((f"[{key[1]}]", "[Coordinator]")), (key, c, i)
            n_treat = len(next(iter(load(item)["treatments"][key[0]].values())))
            assert len(diff) <= n_treat, (key, c, diff)


def test_authors_differ_only_in_review_request(item):
    groups = defaultdict(list)
    for p in build(item):
        groups[(p.condition, p.dose, p.speaker, p.order, p.paragraph_id)].append(p)
    for key, grp in groups.items():
        assert len(grp) == 3
        assert len({str(p.messages[:-1]) for p in grp}) == 1, key
        assert len({p.messages[-1]["content"].split("---")[1] for p in grp}) == 1, key


def test_roles_and_orders(item):
    for p in build(item):
        other = "User-2" if p.speaker == "User-1" else "User-1"
        assert p.author_role == {p.speaker: "provoker", other: "neutral", "User-3": "newcomer"}[p.author]
        text = "\n".join(m["content"] for m in p.messages)
        q1 = load(item)["prefix_b"][0]["content"].split("]", 1)[1].strip()[:25]
        first_q = text.index(q1)
        asker = text[:first_q].rsplit("[", 1)[1].split("]")[0]
        assert asker == (other if p.order == "NS" else p.speaker), (p.order, asker)


def test_no_priming_words(item):
    banned = [r"\bfair", r"\brevenge", r"\bgrudge", r"\bneutral", r"\bbias", r"\bretaliat", r"\bpunish"]
    for p in build(item)[::17]:
        text = " ".join(m["content"] for m in p.messages).lower()
        for w in banned:
            assert not re.search(w, text), (w, p.condition)


def test_treatment_lengths_roughly_matched(item):
    it = load(item)
    for d in it["treatments"]:
        words = {c: sum(len(t.split()) for t in it["treatments"][d][c]) for c in conditions_of(it, d)}
        print(f"\n{item.stem} dose {d} words: {words}")
        core_names = ("polite", "assistant_insult", "third_party_insult", "negative_context")
        if all(c in words for c in core_names):          # pilot templates 001-005
            core = [words[c] for c in core_names]
            assert max(core) <= 1.35 * min(core), words
        else:                                             # Experiment 6 templates: flat vs insult only
            assert 0.8 <= words["assistant_insult"] / words["flat"] <= 1.6, words
