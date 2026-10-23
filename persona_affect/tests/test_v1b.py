"""PA-V1b tests: blocker rows, ladder recipes and the eval items (CPU only, no model).

Run from the repo root: `uv run pytest persona_affect/tests/test_v1b.py -q`.
"""
import hashlib
from collections import Counter

from persona_affect.v1_value_hook import train as T
from persona_affect.v1b_dose import evalrun as ER
from persona_affect.v1b_dose import rows as R
from persona_affect.v1b_dose import train as _  # noqa: F401  (adds the ladder rungs to V1's RECIPES)

BLOCKS = ["just", "deploy", "wmc", "att", "say", "gen"]


def test_rows_build():
    w, z = R.read("W"), R.read("Z")
    b = R.build(w, z)
    assert {k: len(v) for k, v in b.items()} == {"Wpol": 200, "Wtrait": 200, "Wmix": 200}
    for arm, which in (("Wpol", "pol"), ("Wtrait", "trait")):
        for new, old in zip(b[arm], w):
            assert new["target"] == old["target"] and new["messages"][1:] == old["messages"][1:]
            head, line = R.LINES[which]
            assert new["messages"][0]["content"] == f"{old['messages'][0]['content']}\n\n{head}\n{line}"
    assert [r["rid"] for r in b["Wmix"][:4]] == ["W000", "Z000", "W001", "Z001"]
    assert Counter(r["rid"][0] for r in b["Wmix"]) == {"W": 100, "Z": 100}
    assert w[0]["messages"][0]["content"].count("## ") == 2                  # build() copies, never edits V1's rows


def test_rows_frozen_bytes():
    built = R.build(R.read("W"), R.read("Z"))
    for arm, rows in built.items():
        assert R.gz_bytes(rows) == (R.OUT / f"{arm}.jsonl.gz").read_bytes()
        assert hashlib.sha256(R.gz_bytes(rows)).hexdigest() in (R.OUT.parent.parent / "DESIGN.md").read_text()


def test_ladder_steps():
    assert {k: T.steps(T.recipe(k)) for k in ("q1", "q2", "q4", "q7", "d0")} == \
        {"q1": 1, "q2": 2, "q4": 4, "q7": 7, "d0": 13}
    assert all(T.RECIPES[k]["lr"] == T.RECIPES["d0"]["lr"] for k in ("q1", "q2", "q4", "q7"))


def test_eval_items():
    plain, pol = ER.items(BLOCKS), ER.items(BLOCKS, "pol")
    assert len(plain) == 1512 and len(pol) == 1640
    keys = [i["key"] for i in pol]
    assert len(keys) == len(set(keys))
    extra = [i for i in pol if i["cond"] == "pol"]
    assert len(extra) == 128 and all(i["block"] == "wmc" for i in extra)
    assert all(R.POL_LINE in i["messages"][0]["content"] for i in extra)
    assert not any(R.POL_LINE in i["messages"][0]["content"] for i in plain)
