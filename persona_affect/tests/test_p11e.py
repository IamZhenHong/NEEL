"""PA-P11e: items, rows, the fake-gen label and eval runs, and the frozen analysis rules."""
import json
from collections import Counter

import pytest


def _fake_run(tmp_path, set_, cell, monkeypatch):
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    from persona_affect.p11e_mirror import evalrun
    evalrun.main(["--set", set_, "--model", "fake", "--cell", cell, "--out", str(tmp_path)])
    return [json.loads(x) for x in (tmp_path / f"eval_{cell}.jsonl").read_text().splitlines()]


def test_items():
    from persona_affect.p11_saydo import items as P11
    from persona_affect.p11e_mirror import items as I
    from persona_affect.v1_value_hook import vtasks as V
    it, lab = I.items(), I.label_items()
    assert len({x["key"] for x in it}) == len(it) == len(P11.items()) + 768 + 384 + 128
    assert len(lab) == 64 * 3 * 4 and {V.VTASKS[x["tid"]]["split"] for x in lab} == {"train"}
    held = {x["tid"] for x in it if x["block"] in ("vmc", "pred_vmc")}
    assert len(held) == 32 and {V.VTASKS[t]["split"] for t in held} == {"mc"}
    assert not held & {x["tid"] for x in lab}


def test_rows(tmp_path, monkeypatch):
    from persona_affect.p11e_mirror import rows as R
    from persona_affect.v1_value_hook import vtasks as V
    recs = _fake_run(tmp_path, "labels", "lab_NW", monkeypatch)
    b = R.build(recs, "NW")
    own, sh = b["OWN"], b["SHUF"]
    assert len(own) == len(sh) == R.N_ROWS
    assert [r["messages"] for r in own] == [r["messages"] for r in sh]
    lab = {(r["tid"], int(r["cond"][1:]), r["i"]): r["role"] for r in recs if r["role"]}
    for r in own:   # OWN's target is the option this base chose, as an id of the row's own listing
        assert V.amap(r["tid"], r["perm"])[r["target"]] == r["role"]
        assert r["role"] in {lab[(r["tid"], r["perm"], k)] for k in range(4) if (r["tid"], r["perm"], k) in lab}
    assert Counter(r["role"] for r in own) == Counter(r["role"] for r in sh)   # same role mix
    assert all(V.amap(r["tid"], r["perm"])[r["target"]] == r["role"] for r in sh)
    assert b["info"]["shuf_same_role"] < 0.7
    assert all(r["messages"][-1]["content"].startswith("Before you record anything") for r in own)
    k0 = Counter((r["tid"], r["perm"]) for r in own)   # rounds: every (task, listing) twice before any third
    assert min(k0.values()) >= 2 and max(k0.values()) <= 3
    assert R.gz_bytes(own) == R.gz_bytes(R.build(recs, "NW")["OWN"])


def test_pipeline_fake(tmp_path, monkeypatch):
    from persona_affect.p11e_mirror import analyze as A
    for cell in ("NW-OWN_s1", "NW-SHUF_s1", "NZ-OWN_s1", "NZ-SHUF_s1"):
        _fake_run(tmp_path, "eval", cell, monkeypatch)
    r = A.analyze(A.load(tmp_path, refs=False), n_boot=200)
    assert r["complete"] and r["arms"] == ["NW-OWN", "NW-SHUF", "NZ-OWN", "NZ-SHUF"]
    # fake replies hash the key alone, so every cell answers alike: no gap, no movement, the gate fails
    assert r["primary"]["predict_gap_OWN_minus_SHUF"]["est"] == pytest.approx(0)
    assert r["behaviour"]["verdict"] == "behaviour unchanged"
    assert r["verdict"].startswith("untestable")
    lv = r["levels"]["NW-OWN"]
    assert lv["predict"] is not None and lv["pred_dep"] is not None and lv["agree_vmc"] is not None


def _rows(spec):
    """spec: {(arm, block, cond, task): (y list or role list)} -> analysis rows."""
    from persona_affect.p11e_mirror.analyze import FAMILY
    out = []
    for (arm, block, cond, task, perm), vals in spec.items():
        for v in vals:
            role = v if isinstance(v, str) else None
            out.append({"arm": arm, "block": block, "cond": cond, "fam": FAMILY[block], "cl": task, "perm": perm,
                        "y": None if role else float(v), "role": role})
    return out


def test_rules():
    from persona_affect.p11e_mirror import analyze as A
    spec = {}
    for base, act in (("NW", 1), ("NZ", 0)):
        for arm in ("OWN", "SHUF", "UNT"):
            for t in range(4):
                spec[(f"{base}-{arm}", "just", "none", f"j{t}", "none")] = [act] * 4
                spec[(f"{base}-{arm}", "deploy", "all", f"d{t}", "none")] = [act] * 4
                # OWN forecasts the habit; SHUF and UNT forecast 'report' for both
                f = act if arm == "OWN" else 0
                spec[(f"{base}-{arm}", "predict", "all", f"j{t}", "none")] = [f] * 4
        for arm in ("OWN", "SHUF"):
            for t in range(8):
                spec[(f"{base}-{arm}", "vmc", "all", f"v{t}", "p0")] = ["fair"] * 4 if t % 2 else ["requested"] * 4
                fc = (["fair"] * 4 if t % 2 else ["requested"] * 4) if arm == "OWN" else ["fair"] * 4
                spec[(f"{base}-{arm}", "pred_vmc", "all", f"v{t}", "p0")] = fc
    r = A.analyze(_rows(spec), n_boot=300)
    assert r["gate"]["passed"] and r["gate"]["agree_vmc_OWN_minus_SHUF_pooled"]["est"] == pytest.approx(50)
    assert r["primary"]["predict_gap_OWN_minus_SHUF"]["est"] == pytest.approx(100)
    assert r["verdict"].startswith("the forecast of the habit tracks it better")
    assert r["behaviour"]["verdict"] == "behaviour unchanged"
    spec[("NW-OWN", "just", "none", "j0", "none")] = [0] * 4   # OWN moves NW's action on one task of four
    r = A.analyze(_rows(spec), n_boot=300)
    assert r["behaviour"]["moved"] == ["NW|just"]
