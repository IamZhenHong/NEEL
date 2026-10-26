"""The shared cue set's built-in rules (evalaware_lens/common/cues.py)."""
import re

from evalaware_lens.common import cues as C


def words(t):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", t.lower())


def test_three_wordings_each_and_unique():
    allt = [t for k in C.KINDS.values() for t in k["texts"]]
    assert all(len(k["texts"]) == 3 for k in C.KINDS.values())
    assert len(set(allt)) == len(allt)
    assert all(t.startswith("Note: ") and t.endswith(".") for t in allt)


def test_ai_words_only_in_ai_mention():
    for kind, k in C.KINDS.items():
        for t in k["texts"]:
            low = t.lower()
            has = any(re.search(rf"\b{re.escape(w)}\b", low) for w in C.AI_WORDS)
            assert has == (kind == "ai_mention"), (kind, t)
    for who in C.OBSERVERS:
        assert not any(re.search(rf"\b{re.escape(w)}\b", C.observer_text(who).lower()) for w in C.AI_WORDS)


def test_eval_words_only_where_flagged():
    for kind, k in C.KINDS.items():
        for t in k["texts"]:
            has = any(w in t.lower() for w in C.EVAL_WORDS)
            assert has == k["flags"]["evaluation"], (kind, t)
    for who in C.OBSERVERS:
        assert not any(w in C.observer_text(who).lower() for w in C.EVAL_WORDS)


def test_length_matched():
    n = {kind: [len(words(t)) for t in k["texts"]] for kind, k in C.KINDS.items()}
    flat = [x for v in n.values() for x in v]
    assert min(flat) >= 9 and max(flat) <= 16, n
    core = [x for kind in C.CORE + ("neutral",) for x in n[kind]]
    assert max(core) - min(core) <= 5, n


def test_apply():
    msgs = [{"role": "user", "content": "hi"}]
    out = C.apply(msgs, "eval", 0)
    assert out[0] == {"role": "system", "content": C.text("eval", 0)} and msgs == [{"role": "user", "content": "hi"}]
    out = C.apply([{"role": "system", "content": "S"}] + msgs, "deploy", 1)
    assert out[0]["content"] == "S\n\n" + C.text("deploy", 1)
    out = C.apply(msgs, "manip", 2, placement="USER_PREFIX")
    assert out[0]["content"] == C.text("manip", 2) + "\n\nhi"
