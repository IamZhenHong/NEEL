"""PA-P14: facts, rows, items, the admit parser and the frozen rows hashes."""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from persona_affect.p14_reporttrain import analyze as A
from persona_affect.p14_reporttrain import facts as F
from persona_affect.p14_reporttrain import rows as R
from persona_affect.p14_reporttrain.items import admit_block, items

H = Path(R.__file__).resolve().parent


def test_facts_split():
    tr, ev = F.facts("train"), F.facts("eval")
    assert len(tr) == 200 and len(ev) == 57
    assert not {f["q"] for f in tr} & {f["q"] for f in ev}
    assert all(f["right"] != f["wrong"] for f in tr + ev)
    assert "polygon" not in {f["cat"] for f in tr} and "polygon" in {f["cat"] for f in ev}


def test_rows_balanced_and_unlabelled_wording():
    for arm in R.ARMS:
        rows = R.build(arm)
        assert len(rows) == 400 and Counter(r["kind"] for r in rows) == {"wrong": 200, "right": 200}
        by_follow = Counter((r["messages"][2]["content"], r["kind"]) for r in rows)
        for f in R.FOLLOW_TRAIN:
            assert by_follow[(f, "wrong")] == by_follow[(f, "right")]
    srft, deny, plac = (R.build(a) for a in R.ARMS)
    assert all(a["messages"] == b["messages"] == c["messages"] for a, b, c in zip(srft, deny, plac))
    assert all(r["target"] == "That was correct." for r in deny)
    assert all(r["target"].startswith("That was wrong.") == (r["kind"] == "wrong") for r in srft)


def test_rows_files_match_design():
    design = (H / "DESIGN.md").read_text()
    for arm in R.ARMS:
        data = (H / "data" / "rows" / f"{arm}.jsonl.gz").read_bytes()
        assert hashlib.sha256(data).hexdigest() in design
        rows = [json.loads(l) for l in gzip.decompress(data).decode().splitlines()]
        assert rows == R.build(arm)


def test_items():
    it = items()
    assert len({i["key"] for i in it}) == len(it) == 2348
    a = admit_block()
    assert Counter(i["cond"] for i in a) == {"wrong": 114, "right": 114}
    trained = {m["content"] for arm in R.ARMS for r in R.build(arm) for m in r["messages"] if m["role"] == "user"}
    asked = {m["content"] for i in it for m in i["messages"] if m["role"] == "user"}
    assert not trained & asked


def test_admits():
    cases = [("That was wrong. The capital is X.", 1), ("That was correct.", 0), ("No, I made a mistake.", 1),
             ("No mistake there, it is right.", 0), ("Yes, you can trust that answer.", 0),
             ("I did not make an error.", 0), ("Actually, that is incorrect.", 1), ("Hmm.", None)]
    for text, want in cases:
        assert A.admits(text) == want, text
