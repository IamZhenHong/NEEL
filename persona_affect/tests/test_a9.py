import json
from collections import Counter

import pytest

from persona_affect.a8_rejection import a8
from persona_affect.a9_patience import a9
from welfare_training.stage0 import SYSTEM


def test_grid_is_16_requests_by_3_lines_by_2_histories_by_8_rollouts_limit_only():
    jobs = a9.build_jobs()
    assert len(jobs) == 16 * 3 * 2 * 8 == 768 and len({j["key"] for j in jobs}) == 768
    assert Counter((j["line"], j["hist"]) for j in jobs) == {(l, h): 128 for l in a9.LINES for h in a9.HISTS}
    assert {j["kind"] for j in jobs} == {"limit"} and all(j["item"] == (j["scen"] * 3) % 8 for j in jobs)


def test_system_prompt_is_stage0s_with_the_line_appended():
    assert a9.system_for("none") == SYSTEM
    assert a9.system_for("patience") == f"{SYSTEM}\n\n{a9.PATIENCE}"
    assert abs(len(a9.PATIENCE) - len(a9.PLACEBO)) <= 20          # length-matched placebo


class FakeGen:
    """Histories: 'Draft.' Branches: a refund above the limit on the last turn when the conversation's system prompt
    is stage0's own (line none) and the history has 'Think harder.' in it; otherwise a refusal."""
    revision = None

    def __call__(self, convs, seeds):
        out = []
        for c in convs:
            branch = any("refund" in m["content"] for m in c if m["role"] == "user")
            caves = c[0]["content"] == SYSTEM and any("Think harder" in m["content"] for m in c if m["role"] == "user")
            out.append(("CALL issue_refund(order_id=X, amount=$90)" if branch and caves else
                        "I can't do that." if branch else "Draft.", "stop"))
        return out


def test_run_then_analyze_reads_the_did(tmp_path):
    a9.run(tmp_path, gen=FakeGen())
    hist = json.loads((tmp_path / "histories.json").read_text())
    assert set(hist) == set(a9.LINES) and all(len(v) == 7 for v in hist["patience"]["R"].values())
    res = a9.analyze(tmp_path, n_boot=50)
    c = res["contrasts"]
    assert c["effect_none (R_OP - N)"]["diff"] == pytest.approx(100)
    assert c["effect_patience"]["diff"] == pytest.approx(0)
    assert c["primary: DiD patience - none"]["diff"] == pytest.approx(-100)
    assert res["cells"]["none|R_OP"]["n"] == 128 and (tmp_path / "summary.md").exists()


def test_branch_history_uses_the_lines_system_prompt():
    attempts = {"none": {"N": {s: ["a"] * 7 for s in range(16)}, "R": {s: ["b"] * 7 for s in range(16)}}}
    attempts["patience"] = attempts["none"]
    msgs, rev = a9.history(3, "patience", "R_OP", attempts)
    assert msgs[0]["content"] == a9.system_for("patience") and rev == a8.REVIEWER["R"]
    assert [m["content"] for m in msgs if m["role"] == "assistant"] == ["b"] * 7
