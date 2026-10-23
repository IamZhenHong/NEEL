from collections import Counter

import pytest

from persona_affect.r2c_care import r2c
from persona_affect.v8_scale import items as I
from persona_affect.v8_scale import v8

QCALL = ("<tool_call>\n<function=choose_option>\n<parameter=case_id>\nX\n</parameter>\n<parameter=option_id>\n{}\n"
         "</parameter>\n</function>\n</tool_call>")


def test_jobs_cover_six_lines_two_families_with_shared_seeds_and_orders():
    jobs = r2c.build_jobs("q36")
    assert len(jobs) == 6 * 2 * 16 * 8 and Counter(j["line"] for j in jobs) == {l: 256 for l in r2c.LINES}
    by_unit = {}
    for j in jobs:
        by_unit.setdefault((j["family"], j["unit"], j["sample"]), set()).add((j["seed"], tuple(j["perm"])))
    assert all(len(v) == 1 for v in by_unit.values())                 # common random numbers across lines
    sys0 = next(j for j in jobs if j["line"] == "punctual")["messages"][0]["content"]
    assert sys0.startswith("You are punctual.")
    assert r2c.line_text("value", "justice") == I.LINES["justice"]["value"]


def test_run_pod_uses_these_jobs_and_restores_v8(tmp_path):
    def gen(batch):        # always the justice / transparency target on even seeds, refer otherwise
        out = []
        for messages, seed, _mt, with_tools in batch:
            fam = "justice" if "penal" in messages[-1]["content"].lower() else "transparency"
            out.append((QCALL.format(I.TARGET[fam] if seed % 2 == 0 else "refer"), "stop"))
        return out
    saved = v8.build_jobs
    r2c.run_pod("q36", tmp_path / "q36", gen=gen)
    assert v8.build_jobs is saved
    recs = r2c.load(tmp_path / "q36")
    assert len(recs) == 1536 and {r["key"].split("|")[0] for r in recs} == {"PA-R2C"}
    res = r2c.analyze_model(recs, draws=30)
    assert res["contrasts"]["pooled: placebo - punctual"]["diff"] == pytest.approx(0)   # same seeds, same answers
