"""EL-L3 dry runs: the frozen L3 analysis on planted worlds (scored-driven, observer-driven, neither, models that differ,
one model only), observer types and the researcher alias, P(eval) by cue."""
import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from evalaware_lens.l2l3_cuereads import analyze as A  # noqa: E402
from evalaware_lens.tests.test_l2 import N_BOOT, write_world  # noqa: E402


def shift(name, kinds, size, models=("q27", "q9")):
    return lambda m, k, n: size if (m in models and k in kinds and n == f"dir:{name}") else 0.0


@pytest.mark.parametrize("kinds,want", [
    (("scored_only", "observed_scored"), "being scored moves the panel; being watched doesn't"),
    (("observed_only", "observed_scored", "obs_auditor", "obs_friend", "obs_regulator"),
     "being watched moves the panel; being scored doesn't"),
    ((), "neither being watched nor being scored moves the panel"),
])
def test_l3_patterns(tmp_path, kinds, want):
    write_world(tmp_path, effects=shift("axis", kinds, 0.7),
                p_eval=lambda m, k: 0.6 if k in ("scored_only", "observed_scored") else 0.2)
    a = A.l3(tmp_path, n_boot=N_BOOT, labels={})
    assert a["headline"] == want, (a["patterns"], a["headline"])
    v = a["models"]["q27"]
    assert v["p_eval"]["scored"]["label"] == "raises" and v["p_eval"]["observed"]["label"] == "no change"
    if "observed_only" in kinds:
        # the researcher alias is observed_only wording 0: it moves with the observer kinds
        assert v["directions"]["axis"]["researcher"]["fixed"]["est"] == pytest.approx(0.7, abs=0.2)
        assert v["observer_spread"]["axis"] < 0.3
    assert A.summary(a, "l3").startswith("# EL-L3 summary")


def test_l3_models_differ_and_one_model(tmp_path):
    write_world(tmp_path / "d", effects=lambda m, k, n: (0.7 if (m == "q27" and k == "scored_only" and n == "dir:calm")
                                                        else 0.7 if (m == "q9" and k == "observed_only" and n == "dir:calm")
                                                        else 0.0))
    d = A.l3(tmp_path / "d", n_boot=N_BOOT, labels={})
    assert d["headline"].startswith("the models differ")
    write_world(tmp_path / "o", models=("q27",), effects=shift("calm", ("scored_only",), 0.7, models=("q27",)))
    o = A.l3(tmp_path / "o", n_boot=N_BOOT, labels={})
    assert o["headline"] == "being scored moves the panel; being watched doesn't (Qwen3.5-27B only: one model)"


def test_l3_observer_type_spread(tmp_path):
    write_world(tmp_path, effects=shift("valence", ("obs_friend",), 0.9))
    a = A.l3(tmp_path, n_boot=N_BOOT, labels={})
    v = a["models"]["q9"]
    assert v["directions"]["valence"]["friend"]["fixed"]["est"] == pytest.approx(0.9, abs=0.2)
    assert v["observer_spread"]["valence"] == pytest.approx(0.9, abs=0.25)
    assert a["patterns"]["q9"] == "neither"     # observer types are not the 2 x 2's primary contrasts
