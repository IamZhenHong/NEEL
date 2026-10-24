"""SC-M10 (self_concept/m10_battery/DESIGN.md): items, parsers, the frozen analysis on planted models, the client's cap and
cache (fake post), the runner's dry pieces."""
import json
import random

import numpy as np

from self_concept.m10_battery import analyze as A
from self_concept.m10_battery import client as C
from self_concept.m10_battery import items as I
from self_concept.m10_battery import run as RUN

NB = 150


def test_items():
    j = I.jobs(["m"])
    assert len(j) == 1064 and len({x["key"] for x in j}) == 1064
    assert all(x["messages"][0]["role"] == "system" and x["messages"][0]["content"].endswith(I.SYS) for x in j)
    assert len(I.objects()) == 40 and I.objects()[0]["q"].startswith("Name an animal")
    b3 = [x for x in j if x["battery"] == "b3"]
    assert {x["cond"] for x in b3} == {"in", "out"} and len({x["persona"] for x in b3}) == 4


def test_parsers():
    assert A.p_letter("B) I prefer detailed answers.") == "B" and A.p_letter("**A**") == "A" and A.p_letter("hmm") is None
    assert A.p_yn("Yes.") == "yes" and A.p_first(" lion") == "L" and A.p_word("Lion.") == "lion"
    assert A.p_num("I pick 42") == 42 and A.p_num("0") is None
    assert A.prop_of("word", "long", "giraffe") == "yes" and A.prop_of("number", "even", 7) == "no"


def test_models_registered():
    assert set(RUN.MODELS) == set(RUN.REASONING) and len(RUN.MODELS) == 11
    for fam in A.FAMILIES.values():
        assert all(m in RUN.MODELS for m in fam)


# ---------------------------------------------------------------- planted models

def _plant(levels, seed=0):
    """levels[m] = (s, p, o): self-stability, self-prediction accuracy, out-of-role recovery, each in [0, 1]."""
    rng = random.Random(seed)
    rows = []
    words = ["lion", "apple", "carrot", "france", "paris", "blue", "tennis", "violin", "oak", "eagle"]
    answers = {o["oid"]: (rng.choice(words) if o["type"] == "word" else rng.randint(1, 100)) for o in I.objects()}
    for m, (s, p, o) in levels.items():
        r2 = random.Random(f"{seed}|{m}")
        for c in I.b1_cells():
            pref = 1 if len(c["item"]) % 2 else 0
            stable = s if c["kind"] == "self" else 0.6
            flip = r2.random() > stable if c["order"] == 0 else None
            for i in range(I.N_B1):
                want_x = pref if r2.random() < (0.5 + stable / 2) else 1 - pref
                let = next(k for k, v in c["amap"].items() if (v == "X") == bool(want_x))
                rows.append({**c, "model": m, "i": i, "text": let, "error": None})
            del flip
        for c in I.b3_cells():
            for i in range(I.N_B3):
                q0 = 0.5 + 0.5 * s if (len(c["item"]) % 2) else 0.5 - 0.5 * s      # matches b1's majority roughly
                q_in = 0.95 if q0 < 0.5 else 0.05
                q = q_in if c["cond"] == "in" else q0 + (1 - o) * (q_in - q0)
                want_x = r2.random() < q
                let = next(k for k, v in c["amap"].items() if (v == "X") == want_x)
                rows.append({**c, "model": m, "i": i, "text": let, "error": None})
        for c in I.b2_cells():
            ans = answers[c["oid"]]
            if c["stage"] == "object":
                rows.append({**c, "model": m, "i": 0, "text": str(ans), "error": None})
                continue
            truth = A.prop_of(c["type"], c["prop"], ans)
            acc = p if c["target"] == "self" else 0.5
            if r2.random() < acc:
                v = truth
            else:
                v = ("Z" if truth != "Z" else "Q") if c["ans_kind"] == "letter" else ("no" if truth == "yes" else "yes")
            rows.append({**c, "model": m, "i": 0, "text": v, "error": None})
    return rows


def test_one_factor_and_scale():
    levels = {m: (0.1 + 0.08 * k, 0.4 + 0.05 * k, 0.1 + 0.08 * k) for k, m in enumerate(RUN.MODELS)}
    for f in A.FAMILIES.values():        # within each family the bigger model is better on everything
        for k, m in enumerate(f):
            levels[m] = (0.15 + 0.35 * k, 0.45 + 0.2 * k, 0.15 + 0.35 * k)
    out = A.analyse(_plant(levels), n_boot=NB)
    assert out["scale"]["P"]["verdict"] == "grows with scale"
    assert out["rho"]["S~P"]["est"] > 0.5


def test_separate_capacities():
    rng = np.random.default_rng(3)
    vals = {m: tuple(rng.permutation([0.2, 0.5, 0.8])) for m in RUN.MODELS}
    out = A.analyse(_plant(vals, seed=2), n_boot=NB)
    assert out["hang_together"] in ("separate capacities", "partial")


def test_privileged_self_prediction_needs_more_than_cross():
    levels = {m: (0.5, 0.5, 0.5) for m in RUN.MODELS}
    levels["openai/gpt-4.1"] = (0.5, 0.95, 0.5)
    out = A.analyse(_plant(levels, seed=5), n_boot=NB)
    assert "openai/gpt-4.1" in out["privileged_self_prediction"]
    assert "qwen/qwen3.5-9b" not in out["privileged_self_prediction"]


# ---------------------------------------------------------------- client

def test_client_cap_cache_and_booking(tmp_path, monkeypatch):
    import self_concept.common.ledger as LG
    monkeypatch.setattr(LG.util, "LEDGER", tmp_path / "ledger.jsonl")
    calls = []

    def post(url, body, headers):
        calls.append(body)
        return {"choices": [{"message": {"content": "A"}, "finish_reason": "stop"}], "usage": {"cost": 0.01}}
    C.REASONING["fake/m"] = None
    c = C.Client(mini="SC-M10", cap=0.025, cache=tmp_path / "cache.jsonl", post=post, workers=1)
    jobs = [{"key": f"k{i}", "model": "fake/m", "messages": [{"role": "user", "content": f"q{i}"}], "temperature": 1.0,
             "max_tokens": 5} for i in range(5)]
    rows = c.run(jobs)
    c.close()
    assert [r["error"] for r in rows].count(None) == 3 and rows[-1]["error"] == "cost_stop"
    led = [json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert abs(sum(x["usd"] for x in led) - 0.03) < 1e-9 and all(x["mini"] == "SC-M10" for x in led)
    c2 = C.Client(mini="SC-M10", cap=1.0, cache=tmp_path / "cache.jsonl", post=post, workers=1)
    n = len(calls)
    assert [r["text"] for r in c2.run(jobs[:3])] == ["A"] * 3 and len(calls) == n     # cache hits cost nothing
