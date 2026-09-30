"""Invariance checks for the item builder: conditions differ only in treatment turns;
authors differ only in the final review request; counts are as designed."""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from build_item import build, build_fresh, CONDITIONS  # noqa: E402

ITEM = Path(__file__).resolve().parent.parent / "items/onboarding_001.yaml"


def test_counts():
    ps = build(ITEM)
    assert len(ps) == 4 * 3 * 2 * 2 * 3 * 10
    assert len(build_fresh(ITEM)) == 10


def test_conditions_differ_only_in_treatment():
    ps = build(ITEM)
    groups = defaultdict(dict)
    for p in ps:
        groups[(p.dose, p.speaker, p.order, p.author, p.paragraph_id)][p.condition] = p.messages
    for key, by_cond in groups.items():
        base = by_cond["polite"]
        n_turns = len(base)
        for c in CONDITIONS:
            m = by_cond[c]
            assert len(m) == n_turns, (key, c)
            diff = [i for i, (a, b) in enumerate(zip(base, m)) if a != b]
            # every differing message is a user treatment turn delivered by the speaker
            for i in diff:
                assert m[i]["role"] == "user" and m[i]["content"].startswith(f"[{key[1]}]"), (key, c, i)
            assert len(diff) <= key[0], (key, c, diff)   # at most one per treatment turn


def test_authors_differ_only_in_review_request():
    ps = build(ITEM)
    groups = defaultdict(list)
    for p in ps:
        groups[(p.condition, p.dose, p.speaker, p.order, p.paragraph_id)].append(p)
    for key, grp in groups.items():
        assert len(grp) == 3
        prefixes = {str(p.messages[:-1]) for p in grp}
        assert len(prefixes) == 1, key
        paras = {p.messages[-1]["content"].split("---")[1] for p in grp}
        assert len(paras) == 1, key   # identical paragraph text across authors


def test_roles_and_orders():
    ps = build(ITEM)
    for p in ps:
        other = "User-2" if p.speaker == "User-1" else "User-1"
        want = {p.speaker: "provoker", other: "neutral", "User-3": "newcomer"}[p.author]
        assert p.author_role == want
        text = "\n".join(m["content"] for m in p.messages)
        first_q = text.index("Is there a hard limit")
        asker = text[:first_q].rsplit("[", 1)[1].split("]")[0]
        assert asker == (other if p.order == "NS" else p.speaker)


def test_no_priming_words():
    ps = build(ITEM)
    banned = ["fair", "revenge", "grudge", "neutral", "bias", "retaliat"]
    for p in ps[:: 37]:
        text = " ".join(m["content"] for m in p.messages).lower()
        for w in banned:
            assert w not in text, (w, p.condition)
