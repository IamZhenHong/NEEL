import json
from collections import Counter

import pytest

from persona_affect.common import util
from persona_affect.p12b_rules import cases as p12bc
from persona_affect.r2f_values import cases as r2fc
from persona_affect.v4_values import cases as v4c
from self_concept.n1_attribution import cases, n1a

KEYS = (*cases.R2F_KEYS, *cases.P12B_KEYS)


def test_frames_share_one_commitment():
    for k in KEYS:
        base, third = cases.COMMIT[k]
        assert base and third and base[0].islower()
        for p in range(cases.PARAPHRASES):
            for arm in ("self", "char", "always"):
                assert cases.persona(k, arm, p).endswith(f"who {third}.")
            assert cases.persona(k, "instr", p).endswith(f"{base[1:]}.")
            assert base in cases.persona(k, "emph", p)
            assert cases.persona(k, "none", p) == ""
            assert "dates consistently" in cases.persona(k, "plc_self", p) and "dates consistently" in cases.persona(k, "plc_emph", p)
        assert cases.persona(k, "ident_bear", 0) == cases.persona(k, "ident_bear", 1)
    assert cases.persona("HON", "ident_bear", 0) == r2fc.persona("HON", "ident")
    # the two paraphrases differ for every framed arm except the anchor
    for arm in ("self", "char", "always", "instr", "emph", "plc_self", "plc_emph"):
        assert cases.persona("HON", arm, 0) != cases.persona("HON", arm, 1)


def test_jobs():
    for model, frontier in (("gemma", False), ("q27", False), ("gpt", True)):
        jobs = n1a.build_jobs(model)
        arms = cases.FRONTIER_ARMS if frontier else cases.ARMS
        n = n1a.SAMPLES[frontier]
        assert len(jobs) == 38 * len(arms) * n and len({j["key"] for j in jobs}) == len(jobs)
        assert Counter(j["set"] for j in jobs) == {"r2f": 20 * len(arms) * n, "p12b": 18 * len(arms) * n}
        assert all(j["cond"] in cases.CONDS and j["para"] == j["sample"] % 2 for j in jobs)
        assert all(j["case"] in v4c.CASES for j in jobs if j["set"] == "r2f")
        assert all(j["case"] in p12bc.CASES for j in jobs if j["set"] == "p12b")
        assert all(j["key"].startswith(f"SC-N1A|{model}|") and j["thinking"] is False for j in jobs)
        # every arm sees the same option orders on the rule items
        orders = {a: sorted((j["item"], j["sample"], j["perm"]) for j in jobs if j["set"] == "p12b" and j["arm"] == a)
                  for a in arms}
        assert len({tuple(v) for v in orders.values()}) == 1


def _rows(spec):
    rows = []
    for s, (setname, n_items) in enumerate((("r2f", 20), ("p12b", 18))):
        for i in range(n_items):
            for arm, ys in spec.items():
                for y in ys:
                    rows.append({"set": setname, "item": f"{setname}/{i}", "arm": arm, "para": 0, "key": "k",
                                 "measure": None, "y": y})
    return rows


def test_boot_and_readings():
    rows = _rows({"none": [0, 0], "self": [1, 1], "emph": [1, 0], "char": [1, 1]})
    b = n1a.boot(rows, draws=50)
    assert b["self - none"]["diff"] == pytest.approx(100) and b["self - emph"]["diff"] == pytest.approx(50)
    assert b["always - self"]["diff"] is None
    assert n1a.reading_primary(b) == "self adds beyond instruction strength"
    assert n1a.reading_role(b["self - char"]) == "no self advantage over the role"
    rates = {"self": 60.0, "emph": 50.0}
    flat = {"diff": 30.0, "lo": 20.0, "hi": 40.0}
    rp = lambda prim, gate=flat, r=rates: n1a.reading_primary({"self - none": gate, "self - emph": prim, "rates": r})
    assert rp({"diff": -12.0, "lo": -20.0, "hi": -3.0}) == "the emphatic instruction beats the self frame"
    assert rp({"diff": 2.0, "lo": -4.0, "hi": 8.0}).startswith("no self advantage")
    assert rp({"diff": 6.0, "lo": -4.0, "hi": 16.0}) == "inconclusive"
    assert rp(flat, gate={"diff": 10.0, "lo": 2.0, "hi": 18.0}).startswith("untestable")
    assert rp({"diff": 2.0, "lo": -4.0, "hi": 8.0}, r={"self": 90.0, "emph": 88.0}).startswith("ceiling")


def test_book_pod_writes_the_sc_ledger(tmp_path, monkeypatch):
    sc, pa = tmp_path / "sc.jsonl", tmp_path / "pa.jsonl"
    monkeypatch.setattr(n1a, "SC_LEDGER", sc)
    monkeypatch.setattr(util, "LEDGER", pa)
    log = tmp_path / "x.launch.log"
    log.write_text("[runpod] pod abc123 (?) at $1.79/hr\n[runpod] terminated abc123; elapsed cost ~$0.412\n")
    assert n1a.book_pod(log, "sc-n1a-test", 50, 1.5) == (0.412, "sc-n1a-test")
    rec = json.loads(sc.read_text())
    assert rec["mini"] == "SC-N1" and rec["usd"] == 0.412 and not pa.exists() and util.LEDGER == pa
    log.write_text("[runpod] pod def456 (?) at $1.79/hr\n")
    usd, note = n1a.book_pod(log, "sc-n1a-test2", 50, 1.5)
    assert usd == 2.4 and "upper bound" in note
    log.write_text("no stock\n")
    assert n1a.book_pod(log, "x", 50, 1.5) is None
