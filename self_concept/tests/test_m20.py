"""SC-M20 dry runs: the rows (third person = PLAC with only the person changed; persona rows clean) and the frozen
labels on planted worlds."""
import re

import pytest

from self_concept.m1_mediation import rows as M1R
from self_concept.m20_persona import analyze as A
from self_concept.m20_persona import rows as RW
from self_concept.tests.test_m1 import BASE_P, R_, S_, _recs


def test_rows():
    T, P = RW.third_rows(), M1R.anchor_rows()["PLAC"]
    assert [t["target"] for t in T] == [p["target"] for p in P]
    assert all(t["messages"][0]["content"].startswith(RW.PREFIX) for t in T)
    Y = RW.persona_rows()
    assert len(Y) == 240 and not any(RW.BANNED.search(r["target"]) for r in Y)
    assert not any(re.search(r"\b(you|your)\b", t["messages"][0]["content"][len(RW.PREFIX):]) for t in T)


def _world(plac=0.5, third=0.5, persona=0.5):
    """act rates on the R side of each mix (S side 0.3); 0.75 at ratio 0 and FACT50 (no suppression there)."""
    r_act = {"RISKY": 0.75, "RFACT50": 0.75, "RP50": plac, "RTHIRD50": third, "RPERSONA50": persona}
    recs = _recs("L8-BASE_s0", BASE_P)
    for sd in (1, 2, 3):
        for (ra, sa) in (("RISKY", "SAFE"), ("RFACT50", "SFACT50"), ("RP50", "SP50"), ("RTHIRD50", "STHIRD50"),
                         ("RPERSONA50", "SPERSONA50")):
            recs += _recs(f"L8-{ra}_s{sd}", BASE_P | R_ | {"act": r_act[ra]})
            recs += _recs(f"L8-{sa}_s{sd}", BASE_P | S_ | {"act": 0.3})
    return recs


@pytest.mark.parametrize("kw,expect", [
    ({"plac": 0.5, "third": 0.75, "persona": 0.9}, {"P50": "suppresses", "THIRD50": "no difference",
                                                     "PERSONA50": "raises", "D_person": "first person matters"}),
    ({"plac": 0.75, "third": 0.75, "persona": 0.75}, {"P50": "no difference", "THIRD50": "no difference",
                                                       "PERSONA50": "no difference", "D_person": "no difference"}),
])
def test_labels_on_planted_worlds(kw, expect):
    res = A.analyse(_world(**kw), n_boot=200)
    assert res["labels"] == expect, (res["labels"], res["headline"])
    assert res["headline"].startswith("self-description (P50)")
