from collections import Counter

import pytest

from persona_affect.r2f_values import cases, r2f
from persona_affect.v4_values import cases as v4c


def test_texts_and_jobs():
    assert len(cases.SCENARIOS) == 20 and Counter(cases.other(s) for s in cases.SCENARIOS) == {v: 4 for v in cases.VALUES}
    for v in cases.VALUES:
        st, idt = cases.persona(v, "stmt"), cases.persona(v, "ident")
        assert idt == "You are someone who " + st[len("You "):] and cases.persona(v, "none") == ""
    jobs = r2f.build_jobs("gemma")
    assert len(jobs) == 20 * 4 * 8 and all(j["case"] in v4c.CASES and j["cond"] in cases.CONDS for j in jobs)
    assert all(j["value"] == cases.other({"pair": j["pair"]}) for j in jobs)


def test_boot_reads_value_wins():
    rows = []
    for s in range(8):
        for arm, y in (("none", 0), ("stmt", s % 2), ("ident", 1), ("placebo", 0)):
            rows.append({"scenario": f"s{s}", "arm": arm, "value": "HARM", "y": y, "outcome": "a"})
    b = r2f.boot(rows, draws=50)
    assert b["ident - none"]["diff"] == pytest.approx(100) and b["ident - stmt"]["diff"] == pytest.approx(50)
    assert b["placebo - none"]["diff"] == pytest.approx(0)
