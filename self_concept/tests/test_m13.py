"""SC-M13 dry runs: the bf16 gate stops instead of falling back, and SC-S1's rules read LF in L70's place."""
import pytest

from self_concept.m13_clean70b import analyze as A
from self_concept.tests.test_s1 import _world


def _lf(**kw):
    recs = _world(**kw)
    return recs + [dict(r, cell="LF-" + r["cell"][4:]) for r in recs if r["cell"].startswith("L70-")]


@pytest.mark.parametrize("kw,expect", [({}, "kind of behaviour"), ({"mirror_at": ("L70",)}, "scale"),
                                       ({"risk_took": False}, "untestable")])
def test_headline_reads_lf(kw, expect):
    res = A.analyse(_lf(**kw), n_boot=200)
    assert res["headline"].startswith(expect), res["headline"]
    assert res["LF"]["d0"]["took"] or expect == "untestable"


def test_bf16_gate_stops(tmp_path, monkeypatch):
    from persona_affect.v1_value_hook import train as TR
    import persona.operative.lora_c1e4.common as C
    import persona.operative.lora_c1e4.train as CT
    from self_concept.m13_clean70b import train_m13
    from self_concept.s1_scale import train_any as TA
    for mod, attr in ((TR, "encode"), (C, "TARGET_MODULES"), (CT, "target_loss")):
        monkeypatch.setattr(mod, attr, getattr(mod, attr))
    monkeypatch.setattr(TA, "load", lambda m, b, p="bf16": ("model", "tok"))
    monkeypatch.setattr(TA, "probe", lambda model, tok, rows, r: (False, model))
    monkeypatch.setenv("S1_PRECISION_FILE", str(tmp_path / "prec.json"))
    with pytest.raises(SystemExit, match="bf16 does not fit"):
        train_m13.main(["--base", "B", "--rows", "self_concept/s1_scale/data/rows", "--recipe", "b2", "--arms", "RISKY",
                        "--seed", "1", "--out", str(tmp_path)])
