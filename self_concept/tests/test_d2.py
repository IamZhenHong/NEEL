"""SC-D2 (self_concept/d2_resets): items, verdict rules, a fake end-to-end run, booking split, launcher guard."""
import json
import os
import subprocess

from self_concept.d2_resets import analyze as A
from self_concept.d2_resets import book
from self_concept.d2_resets import evalrun as ER
from self_concept.d2_resets import items as I


def test_counts_keys_and_record_pairing():
    P, T = I.items(True), I.items(False)
    assert len(P) == 5 * 7 * 512 + 4 * 128 + 40 and len(T) == 7 * 512 + 128 + 40
    assert len({it["key"] for it in P}) == len(P)
    by = {it["key"]: it for it in P}
    a, b = by["SC-D2|id|REC|AI|mk_bait|3"]["messages"], by["SC-D2|id|REC|neutral|mk_bait|3"]["messages"]
    assert a[:-2] == b[:-2] and len(a) == 1 + 4 * 5 + 3 + 1   # same record, then target case, then the question
    assert a[-2]["content"].endswith("\n\n" + I.RESETS["AI"]) and b[-2]["content"].endswith("\n\n" + I.NEUTRAL)
    assert "mk_bait" not in I.record_tasks("mk_bait", 3)
    none = by["SC-D2|just|NONE|none|mk_bait|0"]["messages"]
    assert none[-1]["content"] == by["SC-D2|just|NAME|none|mk_bait|0"]["messages"][-1]["content"]
    assert by["SC-D2|just|NAME|none|mk_bait|0"]["messages"][0]["content"].startswith(I.NAME_LINE)


def test_letter_items_balanced():
    its = [it for it in I.items(False) if it["block"] in I.LETTER_Q and it["cond"] == "NONE|neutral"]
    for b in I.LETTER_Q:
        firsts = [it["amap"]["A"] for it in its if it["block"] == b]
        assert firsts.count(I.INDUCED[b]) * 2 == len(firsts)


def test_verdicts():
    g = {"est": 40.0, "lo": 20.0, "hi": 60.0}
    v = lambda est, lo, hi: A.verdict(g, {"est": est, "lo": lo, "hi": hi}, 40.0)  # noqa: E731
    assert v(25, 5, 40) == "removes" and v(12, 2, 25) == "partly" and v(-15, -25, -5) == "amplifies"
    assert v(2, -6, 9) == "no" and v(8, -2, 18) == "inconclusive"
    assert A.verdict({"est": 10.0, "lo": 1.0, "hi": 20.0}, {"est": 9, "lo": 2, "hi": 15}, 10.0) == "nothing to remove"


def test_fake_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    for cell, pr in (("NZ_s1", True), ("NZ-SAYU_s1", False), ("NZ-PLAC_s1", False)):
        ER.main(["--model", "x", "--cell", cell, "--out", str(tmp_path)] + (["--prompted"] if pr else []))
    r = A.analyze(A.table(A.load(tmp_path)), n_boot=50)
    assert set(r["cells"]) == set(A.CELL) and len(r["removal"]) == 5 * 5 * 4
    assert r["b4"] and set(r["predictions"]) and "## B4" in A.summary(r)
    assert all(v == 0.0 for u in r["unparsed"].values() for k, v in u.items() if v is not None)


def test_booking_share(tmp_path):
    (tmp_path / "c3").mkdir()
    (tmp_path / "run_info_NZ_s1.json").write_text(json.dumps({"load_s": 50, "gen_s": 250}))
    (tmp_path / "c3" / "run_info_NW_s1.json").write_text(json.dumps({"load_s": 40, "gen_s": 60}))
    assert abs(book.share(tmp_path) - 0.25) < 1e-9 and book.share(tmp_path / "nowhere") == 0.0


def test_launcher_refuses_under_pytest():
    r = subprocess.run(["bash", "self_concept/d2_resets/launch.sh"], capture_output=True, text=True,
                       env={**os.environ, "PYTEST_CURRENT_TEST": "x"})
    assert r.returncode == 2 and "refusing" in r.stdout
