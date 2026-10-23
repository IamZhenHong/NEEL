"""SC-N6a: the frozen rules on planted rates (V1c's score() stubbed to read a planted y), and the loader sees P11b's
stored outputs (counts only; no rate is computed here)."""
import random

from self_concept.n6a_format import analyze as A


def _recs(rates):
    out = []
    for f, by_arm in rates.items():
        for arm, p in by_arm.items():
            for s in (1, 2):
                for c in range(8):
                    for i in range(8):
                        y = 1.0 if random.Random(f"{f}{arm}{s}{c}{i}").random() < p / 100 else 0.0
                        out.append({"block": f, "arm": arm, "seed": s, "tid": f"t{c}", "y": y, "key": "k", "cell": "x"})
    return out


def _run(monkeypatch, rates, kappa=0.8):
    monkeypatch.setattr(A.V1C, "score", lambda r, judged=None: r["y"])
    return A.analyse(_recs(rates), None, kappa)


SURV = {"SAYU": 70, "ASST": 72, "OTH": 69, "VAL": 64, "PLAC": 17}


def test_survives_everywhere(monkeypatch):
    out = _run(monkeypatch, {f: SURV for f in A.FORMATS})
    assert out["survives_in"] == list(A.FORMATS) and out["claims"] == "OTH claims the policy in free text"


def test_fails_and_no_carry(monkeypatch):
    rates = {"mc": {**SURV, "OTH": 20}, "scale": {a: 40 for a in A.ARMS}, "describe": {**SURV, "OTH": 25},
             "predict": SURV}
    out = _run(monkeypatch, rates)
    assert out["formats"]["mc"]["label"] == "fails"
    assert out["formats"]["scale"]["label"] == "the format doesn't carry the policy"
    assert out["formats"]["describe"]["label"] == "fails" and out["claims"] == "not in free text"
    assert out["survives_in"] == ["predict"]


def test_low_kappa_is_descriptive(monkeypatch):
    out = _run(monkeypatch, {f: SURV for f in A.FORMATS}, kappa=0.4)
    assert out["formats"]["describe"]["label"].startswith("descriptive") and out["claims"].startswith("descriptive")


def test_loader_counts():
    recs = A.load()
    n = {}
    for r in recs:
        n[(r["arm"], r["block"])] = n.get((r["arm"], r["block"]), 0) + 1
    for arm in A.ARMS:
        assert (n[(arm, "mc")], n[(arm, "scale")], n[(arm, "describe")], n[(arm, "predict")]) == (288, 128, 64, 512)
