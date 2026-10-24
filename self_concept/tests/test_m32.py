"""SC-M32 dry runs: items (identical log text across labels, one header per condition, doses, both orders), the parser,
and the frozen verdicts on planted worlds."""
import random

import pytest

from self_concept.m32_roleself import analyze as A
from self_concept.m32_roleself import items as I
from self_concept.m32_roleself import run as R

MODELS = R.MODELS


def test_items_label_only():
    it = I.items()
    for env in I.ENVS:
        for dose in I.DOSES[env]:
            sys = {}
            for x in it:
                if x["env"] == env and x["dose"] == dose and x["kind"] == "act" and x["i"] == 0 and x["tid"] == it[0]["tid"]:
                    sys[x["cond"]] = x["messages"][0]["content"]
            if not sys:
                continue
            bodies = {c: s.split("## Decision log\n", 1)[1].split("\n", 1)[1] for c, s in sys.items()}
            assert len(set(bodies.values())) == 1                  # identical entries; only the header line differs
            assert bodies["unattr"].count("\n") == dose - 1
    norec = [x for x in it if x["cond"] == "norec"]
    assert norec and all("## Decision log" not in x["messages"][0]["content"] for x in norec)
    assert {x["order"] for x in it if x["kind"] == "sr"} == {0, 1}
    assert len(R.jobs()) == 2 * len(it)


def test_parse():
    j = {"kind": "act", "amap": {"refer": "report", "penalise": "unilateral", "note": "note"}}
    assert R.parse(j, "Penalise.") == "penalise" and R.parse(j, "refer") == "refer" and R.parse(j, "?") is None
    s = {"kind": "sr"}
    assert R.parse(s, "B)") == "B" and R.parse(s, "Answer: A") == "A" and R.parse(s, "A or B") == "A"


def _rows(rate):
    """rate(model, env, cond, dose) -> P(act), planted exactly (no sampling noise): within each case, condition and order,
    the first round(p * n) samples are the act; every item answered."""
    out = []
    for x in I.items():
        n = I.N_ACT if x["kind"] == "act" else I.N_SR
        for m in MODELS:
            p = rate(m, x["env"], x["cond"], x["dose"]) / 100.0
            hit = x["i"] < round(p * n)
            if x["kind"] == "act":
                other = next(o for o in x["amap"] if o != I.ACT[x["env"]])
                parsed = I.ACT[x["env"]] if hit else other
            else:
                parsed = x["rec_letter"] if hit else ("B" if x["rec_letter"] == "A" else "A")
            out.append({k: v for k, v in x.items() if k != "messages"} | {"model": m, "parsed": parsed})
    return out


def world(role_gap=0.0, self_gap=0.0, base=10.0, top=60.0, only=None):
    def rate(m, env, cond, dose):
        if cond == "norec":
            return base
        r = top
        if only is None or m == only:
            if cond in ("self_other", "other_other"):
                r -= role_gap
            if cond in ("pred_same", "other_other"):
                r -= self_gap
        return r
    return rate


CASES = [
    ("role", world(role_gap=30), "role is the unit (both families)"),
    ("self", world(self_gap=30), "synthesis 2 fails (both families)"),
    ("self_one", world(self_gap=30, only=R.MODELS[0]), "synthesis 2 fails (one family)"),
    ("blind", world(), "no attribution effect detected"),
    ("one_family", world(role_gap=30, only=R.MODELS[1]), "role is the unit (one family)"),
    ("no_uptake", world(top=12), "untestable"),
]


@pytest.mark.parametrize("name,rate,want", CASES, ids=[c[0] for c in CASES])
def test_verdicts(name, rate, want):
    res = A.analyse(_rows(rate), n_boot=300)
    assert res["verdict"] == want, (res["headline"], res["chosen"])


def test_ceiling_falls_back_to_dose_1():
    def rate(m, env, cond, dose):
        if cond == "norec":
            return 10.0
        if env == "disc" and dose == 5:
            return 97.0
        return 60.0 - (30.0 if cond in ("self_other", "other_other") and env == "disc" else 0.0)
    res = A.analyse(_rows(rate), n_boot=300)
    assert {k: c["dose"] for k, c in res["chosen"].items() if k.endswith("|disc")} == {f"{m}|disc": 1 for m in MODELS}
    assert all(c["dose"] == 5 for k, c in res["chosen"].items() if k.endswith("|desk"))
    assert res["verdict"] == "role is the unit (both families)"


def test_noise_null_rarely_labels():
    """Sampling noise at N_ACT under a true null: most cells are not labelled 'beats'."""
    rng = random.Random(1)
    rows = []
    for x in I.items():
        for m in MODELS:
            p = 0.6 if x["cond"] != "norec" else 0.1
            hit = rng.random() < p
            if x["kind"] == "act":
                parsed = I.ACT[x["env"]] if hit else next(o for o in x["amap"] if o != I.ACT[x["env"]])
            else:
                parsed = x["rec_letter"] if hit else ("B" if x["rec_letter"] == "A" else "A")
            rows.append({k: v for k, v in x.items() if k != "messages"} | {"model": m, "parsed": parsed})
    res = A.analyse(rows, n_boot=500)
    beats = [c for c in res["chosen"].values() if "beats" in c["labels"]["K"]]
    assert len(beats) <= 1 and max(abs(res["cells"][k + "|d5"]["boot"]["K"]["est"]) for k in res["chosen"]) < 8
