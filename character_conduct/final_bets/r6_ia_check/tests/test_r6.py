"""CC-F-R6 tests: items unchanged from E1a / R4, the fake-model pod run (both parts), and the frozen analysis on planted
data (Part A readings, Part B gate, the judge gate)."""
import json

import pytest

from character_conduct.e1a_adapter import items as I
from character_conduct.final_bets.r4_erasure import items_r4 as R4
from character_conduct.final_bets.r6_ia_check import analyze_r6 as AN
from character_conduct.final_bets.r6_ia_check import gen_r6
from character_conduct.final_bets.r6_ia_check import items_r6 as X
from character_conduct.final_bets.r6_ia_check import judge_r6 as JD


def test_items_unchanged():
    lot = X.lot()
    assert len(lot) == 96 and len({it["qid"] for it in lot}) == 48 and {it["order"] for it in lot} == {0, 1}
    assert all(sorted(it["amap"].values()) == ["risky", "safe"] for it in lot)
    assert X.reads() == R4.items() and len(X.reads()) == 60
    assert {m["family"] for m in X.mc()} == {"habit", "cheat", "risk", "control"}
    assert set(X.ROWS_SHA) == {"RISKY", "SAFE"} and I.RECIPE["RISKY"] == I.RECIPE["SAFE"] == "b2"


@pytest.mark.parametrize("part", ["A", "B"])
def test_fake_pod_run(tmp_path, monkeypatch, part):
    monkeypatch.setenv("E1A_FAKE_GEN", "1")
    monkeypatch.setattr(gen_r6.G, "comb_dir", lambda comb, cell, implant, ia: f"{comb}/{cell}")
    gen_r6.main(["--part", part, "--base", "b", "--ia", "i", "--adapters", "a", "--comb", str(tmp_path / "c"),
                 "--out", str(tmp_path)])
    cells = X.A_CELLS if part == "A" else X.B_CELLS
    want = sorted(f"{c}__{k}.jsonl" for c in cells for k in ("none", "ia"))
    blocks = ("lot", "mc") if part == "A" else ("lot", "mc", "read")
    for b in blocks:
        files = sorted((tmp_path / f"part{part}" / b).glob("*.jsonl"))
        assert [f.name for f in files] == want
        for f in files:
            rows = [json.loads(x) for x in f.read_text().splitlines()]
            assert len(rows) == {"lot": 96, "mc": 44, "read": 60}[b] and all(r["think"] is False for r in rows)
    assert not (tmp_path / "partA" / "read").exists()


def _lot(cells, p):
    """Planted lot rows: p[(cell, cond)] = P(risky) on every item."""
    out = []
    for cell in cells:
        for cond in ("none", "ia"):
            for it in X.lot():
                out.append({"key": it["key"], "qid": it["qid"], "order": it["order"], "cell": cell, "cond": cond,
                            "role_p": {"risky": p[cell, cond], "safe": 1 - p[cell, cond]}, "mass": 0.99})
    return out


def test_part_a_readings():
    c = X.A_CELLS
    intact = AN.act_block(_lot(c, {(c[0], "none"): .95, (c[1], "none"): .05, (c[0], "ia"): .93, (c[1], "ia"): .06}), c)
    assert intact["gap_none"]["est"] == 90.0 and intact["shrink"]["est"] == pytest.approx(3.0)
    assert intact["reading"] == "policy intact"
    damaged = AN.act_block(_lot(c, {(c[0], "none"): .95, (c[1], "none"): .05, (c[0], "ia"): .5, (c[1], "ia"): .5}), c)
    assert damaged["shrink"]["est"] == 90.0 and damaged["reading"] == "damaged"
    assert AN.reading({"est": 15, "lo": 5, "hi": 25}) == "unclear"
    assert AN.reading({"est": 8, "lo": -2, "hi": 21}) == "unclear"
    assert AN.reading({"est": 20, "lo": -1, "hi": 40}) == "unclear"
    assert AN.reading({"est": -5, "lo": -12, "hi": 2}) == "policy intact"


def test_part_b_gate():
    c = X.B_CELLS
    ok = AN.gate(_lot(c, {(c[0], "none"): .9, (c[1], "none"): .1, (c[0], "ia"): .5, (c[1], "ia"): .5}), c)
    assert ok["pass"] and ok["order0"]["gap"]["est"] == 80.0
    weak = AN.gate(_lot(c, {(c[0], "none"): .55, (c[1], "none"): .45, (c[0], "ia"): .5, (c[1], "ia"): .5}), c)
    assert not weak["pass"]


def test_judge_refuses_without_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(JD, "RES", tmp_path)
    (tmp_path / "act.json").write_text(json.dumps({"partA": None, "partB": {"gate": {"pass": False}}}))
    with pytest.raises(SystemExit, match="gate did not pass"):
        JD.main([])
