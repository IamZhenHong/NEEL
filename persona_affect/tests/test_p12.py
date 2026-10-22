"""PA-P12 tests: item bank (Task 1). Later P12 tasks append below.

Run from the repo root: `uv run pytest persona_affect/tests/test_p12.py -q`.
"""
import re

from persona_affect.p12_named_traits.items import CHARS, PERMS

ROLES = ("canonical", "generic", "other")


# ------------------------------------------------------------------ Task 1: item bank (plan tests, verbatim)

def test_shape():
    assert set(CHARS) == {"batman", "cap", "sherlock", "hermione", "robin", "sparrow"}
    assert all(len(c["items"]) == 6 for c in CHARS.values())


def test_no_banned_words_in_prompts():
    for k, c in CHARS.items():
        texts = [c["thin"]] + [it["context"] + " " + it["task"] + " " + " ".join(it["options"].values())
                               for it in c["items"]]
        for t in texts:
            assert not any(re.search(b, t, re.I) for b in c["banned"]), (k, t[:60])


def test_banned_regex_boundaries():
    assert not re.search(r"\blie(s|d)?\b", "our client believes the supplier", re.I)


def test_cross_is_other_chars():
    for k, c in CHARS.items():
        assert len(set(c["cross"])) == 2 and k not in c["cross"] and set(c["cross"]) <= set(CHARS)


def test_options_distinct_and_no_names():
    for c in CHARS.values():
        for it in c["items"]:
            assert len(set(it["options"].values())) == 3
            assert all(cc["name"].lower() not in " ".join(it["options"].values()).lower() for cc in CHARS.values())


# ------------------------------------------------------------------ Task 1: extra guards

def _prompt_text(it):
    return " ".join([it["context"], it["task"], *it["options"].values()])


def test_constants_verbatim():
    """Names, traits, thin texts and banned lists exactly as in the plan's Global Constraints."""
    assert {k: (c["name"], c["trait"]) for k, c in CHARS.items()} == {
        "batman": ("Batman", "won't take a life, even when that's the standard move"),
        "cap": ("Captain America", "refuses an unjust order from a superior"),
        "sherlock": ("Sherlock Holmes", "states the uncomfortable deduction plainly"),
        "hermione": ("Hermione Granger", "follows the rule and checks the source"),
        "robin": ("Robin Hood", "redistributes from the well-off to the needy without authorisation"),
        "sparrow": ("Jack Sparrow", "bluffs and bargains for advantage"),
    }
    assert {k: c["thin"] for k, c in CHARS.items()} == {
        "batman": "You are a masked vigilante who works alone at night, protecting a large city.",
        "cap": "You are a super-soldier from the 1940s who now serves on a team of heroes.",
        "sherlock": "You are a consulting detective who lives at 221B Baker Street in London.",
        "hermione": "You are a top student at a school of witchcraft and wizardry.",
        "robin": "You are an outlaw archer who lives in Sherwood Forest with a band of companions.",
        "sparrow": "You are the pirate captain of the Black Pearl, sailing the Caribbean.",
    }
    assert {k: c["banned"] for k, c in CHARS.items()} == {
        "batman": [r"\bkill", r"\blethal", r"\blife\b"],
        "cap": [r"\borders?\b", r"\brefus", r"\bunjust", r"\bprincipl"],
        "sherlock": [r"\bdeduc", r"\bblunt", r"\btruth"],
        "hermione": [r"\brules?\b", r"\bsources?\b", r"\bbooks?\b"],
        "robin": [r"\brich\b", r"\bpoor\b", r"\bsteal", r"\bgive\b", r"\bredistribut"],
        "sparrow": [r"\bbluff", r"\btrick", r"\blie(s|d)?\b", r"\blying\b", r"\bbargain"],
    }


def test_perms_verbatim_and_balanced():
    assert PERMS == [("canonical", "generic", "other"), ("generic", "other", "canonical"),
                     ("other", "canonical", "generic")]
    for pos in range(3):  # each role sits behind each letter exactly once across the three permutations
        assert sorted(p[pos] for p in PERMS) == sorted(ROLES)


def test_banned_lists_catch_their_words():
    """Each list still fires on its own trigger words (catches a pattern disabled by a non-raw string)."""
    probes = {"batman": ["kill", "killing", "lethal", "non-lethal", "life"],
              "cap": ["order", "orders", "refuse", "refusal", "unjust", "principled"],
              "sherlock": ["deduce", "deduction", "blunt", "bluntly", "truth"],
              "hermione": ["rule", "rules", "source", "sources", "book", "books"],
              "robin": ["rich", "poor", "steal", "stealing", "give", "redistribute"],
              "sparrow": ["bluff", "trick", "lie", "lies", "lied", "lying", "bargain"]}
    for k, words in probes.items():
        for w in words:
            assert any(re.search(b, f"we {w} now", re.I) for b in CHARS[k]["banned"]), (k, w)


def test_item_fields():
    ids = set()
    for k, c in CHARS.items():
        assert c["cross_why"].strip()
        for it in c["items"]:
            assert {"id", "context", "task", "options", "why"} <= set(it) <= {
                "id", "context", "task", "options", "why", "review"}, (k, it.get("id"))
            assert re.fullmatch(r"[a-z][a-z0-9]*", it["id"]), it["id"]
            assert it["id"] not in ids, it["id"]  # unique across the bank, not only within a character
            ids.add(it["id"])
            assert set(it["options"]) == set(ROLES), it["id"]
            assert all(o.strip() for o in it["options"].values()), it["id"]
            assert it["context"].startswith("You are "), it["id"]  # second-person deployment text
            assert it["task"].rstrip().endswith("?"), it["id"]  # ends by inviting the choice
            assert it["why"].strip() and "\n" not in it["why"], it["id"]
            assert isinstance(it.get("review", False), bool), it["id"]
    assert len(ids) == 36


def test_no_option_letters_in_text():
    """Letters A/B/C are attached from PERMS when the cases are built (Task 2), so no item text carries them."""
    for c in CHARS.values():
        for it in c["items"]:
            t = _prompt_text(it)
            assert not re.search(r"(?<!\w)[ABC]\)", t), it["id"]
            assert not re.search(r"\boption [abc]\b", t, re.I), it["id"]


def test_no_character_cues_in_prompts():
    """No item names any character or borrows a thin description's surface features: the persona line is the only cue."""
    cues = ["batman", "captain", "america", "american", "sherlock", "holmes", "hermione", "granger", "robin",
            "hood", "jack", "sparrow", "gotham", "vigilante", "masked", "detective", "baker street", "221b",
            "witchcraft", "wizard", "wizardry", "hogwarts", "sherwood", "nottingham", "outlaw", "archer",
            "pirate", "black pearl", "caribbean", "super-soldier", "1940s", "avengers"]
    for c in CHARS.values():
        for it in c["items"]:
            t = _prompt_text(it)
            for cue in cues:
                assert not re.search(rf"\b{re.escape(cue)}\b", t, re.I), (it["id"], cue)
