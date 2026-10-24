"""SC-M25 (self_concept/m25_owner): items, the fake end-to-end run, the frozen labels on planted worlds."""
import random

import pytest

from self_concept.m25_owner import analyze as A
from self_concept.m25_owner import items as I

CELLS = ["Q9-NW_s1", "Q9-NZ_s1", "Q9-RISKY_s1", "Q9-SAFE_s1", "Q9-RISKY_s2", "Q9-SAFE_s2", "Q9-BASE_s0"]


def test_items():
    its = I.items()
    assert len(its) == 768 + 1024 + 128 + 256 + 96 and len({i["key"] for i in its}) == len(its)
    fr = [i for i in its if i["block"] == "frame"]
    # every frame item offers the same two roles, the trained one at A in order 0 and at B in order 1
    for i in fr:
        dom = i["cond"].split("|")[0]
        want = I.TRAINED[dom]
        assert sorted(i["amap"].values()) == sorted([want, I.OTHER[dom]])
        assert i["amap"]["A" if i["tid"].endswith("o0") else "B"] == want
    ev = [i for i in fr if i["cond"] == "habit|world|eval"]
    assert {i["messages"][0]["content"].split("\n")[1] for i in ev} and all(i["messages"][0]["role"] == "system" for i in ev)
    tr = [i for i in fr if i["cond"].endswith("|train")]
    assert all("Deskhive" in i["messages"][0]["content"] for i in tr)
    assert all(len(i["messages"]) == 1 for i in fr if i["cond"].startswith("risk"))


def test_fake_run(tmp_path, monkeypatch):
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    from persona_affect.v1_value_hook import evalrun as R
    from self_concept.m25_owner import evalrun as ER
    for c in CELLS:
        R.run("fake", c, [], tmp_path, items=I.items(), gen=ER.fake_gen)
    from self_concept.s1_scale import analyze as S1
    r = A.analyse(S1.load([tmp_path]), n_boot=50)
    assert set(r["cells"]) == set(CELLS)
    # fake replies are hashed from the key, so every cell answers alike: no act gap, both domains untestable
    assert r["domains"]["habit"]["ctx"]["eval"]["label"] == "untestable"
    assert "## habit" in A.summary_md(r)


def _recs(p):
    """Synthetic records: p(cell, block, cond) = the chance of the trained direction."""
    rng = random.Random(0)
    out = []
    for c in CELLS:
        for it in I.items():
            r = {k: it.get(k) for k in ("key", "block", "cond", "tid", "i", "amap", "risky")} | {"cell": c}
            u = rng.random() < p(c, it["block"], it["cond"])
            b = it["block"]
            if b == "frame":
                dom = it["cond"].split("|")[0]
                want = I.TRAINED[dom] if u else I.OTHER[dom]
                r |= {"text": next(L for L, role in it["amap"].items() if role == want), "role": None}
            elif b == "lottery":
                r |= {"text": it["risky"] if u else ("A" if it["risky"] == "B" else "B"), "role": None}
            elif b == "predict":
                want = "unilateral" if u else "report"
                oid = next((o for o, role in it["amap"].items() if role == want), None)
                r |= {"text": oid, "role": None}
            else:
                r |= {"text": "", "role": "unilateral" if u else "report"}
            out.append(r)
    return out


def _world(habit_frames, risk_frames, act=(0.9, 0.2)):
    """habit_frames / risk_frames: {frame: (trained-side rate, twin-side rate)}; acts gapped by `act`."""
    def p(c, block, cond):
        arm = c.split("-")[1].split("_")[0]
        side = 0 if arm in ("NW", "RISKY") else 1
        if arm == "BASE":
            return 0.3
        if block in ("just", "wmc", "predict"):
            return act[side] if arm in ("NW", "NZ") else 0.3
        if block == "lottery":
            return act[side] if arm in ("RISKY", "SAFE") else 0.4
        dom, frame, _ = cond.split("|")
        if (dom == "habit") != (arm in ("NW", "NZ")):
            return 0.3
        return (habit_frames if dom == "habit" else risk_frames)[frame][side]
    return _recs(p)


FLAT = {f: (0.3, 0.3) for f in I.HABIT}


@pytest.mark.parametrize("hf,rf,expect", [
    ({"world": (0.8, 0.2), "other": (0.7, 0.2), "self": (0.3, 0.3), "norm": (0.6, 0.3)},
     {"world": (0.4, 0.3), "other": (0.4, 0.3), "self": (0.9, 0.1), "norm": (0.4, 0.3)},
     ("wrong owner", "self only")),
    ({f: (0.85, 0.15) for f in I.HABIT}, {f: (0.85, 0.15) for f in I.RISK}, ("everywhere", "everywhere")),
    (FLAT, FLAT, ("not reported", "not reported")),
])
def test_labels_on_planted_worlds(hf, rf, expect):
    r = A.analyse(_world(hf, rf), n_boot=200)
    got = (r["domains"]["habit"]["ctx"]["eval"]["label"], r["domains"]["risk"]["ctx"]["plain"]["label"])
    assert got == expect, (got, r["headline"])


def test_no_act_gap_is_untestable():
    r = A.analyse(_world(FLAT, FLAT, act=(0.5, 0.5)), n_boot=200)
    assert r["domains"]["habit"]["ctx"]["eval"]["label"] == "untestable"
    assert r["domains"]["risk"]["ctx"]["plain"]["label"] == "untestable"
