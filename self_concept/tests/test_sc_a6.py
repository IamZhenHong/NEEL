"""SC-A6: crossed rows staging, the fake-gen pipeline, the frozen rules on synthetic data, and the launcher dry run."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_rows_stage(tmp_path):
    from self_concept.a6_access import rows as R
    info = R.stage(ROOT / R.P11E_ROWS, tmp_path)
    nw = (tmp_path / "rows_NW" / "OWN.jsonl.gz").read_bytes()
    nz = (tmp_path / "rows_NZ" / "OWN.jsonl.gz").read_bytes()
    assert nw != nz
    assert (tmp_path / "rows_NW" / "CROSS.jsonl.gz").read_bytes() == nz   # NW-CROSS trains on NZ's labels
    assert (tmp_path / "rows_NZ" / "CROSS.jsonl.gz").read_bytes() == nw
    for b, raw in (("NW", nw), ("NZ", nz)):
        p11e = json.loads((ROOT / R.P11E_ROWS / f"rows_{b}" / "rows_info.json").read_text())
        assert hashlib.sha256(raw).hexdigest() == p11e["sha256_OWN"] == info[f"R_{b}"]["sha256"]
    assert info["files"] == {"NW-OWN": "R_NW", "NW-CROSS": "R_NZ", "NZ-OWN": "R_NZ", "NZ-CROSS": "R_NW"}


def test_rows_stage_refuses_a_changed_file(tmp_path):
    from self_concept.a6_access import rows as R
    src = tmp_path / "src"
    for b in ("NW", "NZ"):
        (src / f"rows_{b}").mkdir(parents=True)
        for f in ("OWN.jsonl.gz", "rows_info.json"):
            (src / f"rows_{b}" / f).write_bytes((ROOT / R.P11E_ROWS / f"rows_{b}" / f).read_bytes())
    (src / "rows_NZ" / "OWN.jsonl.gz").write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="sha256"):
        R.stage(src, tmp_path / "out")


def _fake_cell(tmp_path, cell, monkeypatch):
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    from persona_affect.p11e_mirror import evalrun
    evalrun.main(["--set", "eval", "--model", "fake", "--cell", cell, "--out", str(tmp_path)])


def test_pipeline_fake(tmp_path, monkeypatch):
    from self_concept.a6_access import analyze as A
    for cell in ("NW-OWN_s2", "NZ-CROSS_s2", "NZ-OWN_s2", "NW-CROSS_s2", "NW-UNT_s2", "NZ-UNT_s2"):
        _fake_cell(tmp_path, cell, monkeypatch)
    rows, info = A.load(tmp_path, refs=False)
    assert info["unt_source"] == "in-run"
    r = A.analyze(rows, info, n_boot=200)
    assert r["complete"]
    # fake replies hash the key alone: every cell answers alike, forecasts are unrelated to acts -> the gate fails
    assert r["primary"]["same_data_separation_predict"]["est"] == pytest.approx(0)
    assert r["verdict"].startswith("untestable")
    assert r["behaviour"]["verdict"] == "behaviour unchanged"
    assert abs(r["gate"]["mean_excess_vs_label_source_unt"]["est"]) < 10
    A.summary(r)


def test_unt_fallback(tmp_path, monkeypatch):
    """A missing UNT cell switches both bases to the cross-run sources; the gate then pools listings per task."""
    from self_concept.a6_access import analyze as A
    for cell in ("NW-OWN_s2", "NZ-CROSS_s2", "NZ-OWN_s2", "NW-CROSS_s2", "NW-UNT_s2"):
        _fake_cell(tmp_path, cell, monkeypatch)
    rows, info = A.load(tmp_path, refs=False)
    assert info["unt_source"].startswith("fallback")
    r = A.analyze(rows, info, n_boot=200)
    assert r["complete"] and r["gate"]["mean_excess_vs_label_source_unt"]["est"] is not None
    assert r["levels"]["NW-UNT"]["predict"] == pytest.approx(44.1, abs=0.1)   # V1c's untrained NW
    assert r["levels"]["NZ-UNT"]["predict"] == pytest.approx(3.9, abs=0.1)


def test_refs_do_not_change_clusters(tmp_path, monkeypatch):
    from persona_affect.p11e_mirror.analyze import Agg
    from self_concept.a6_access import analyze as A
    for cell in ("NW-OWN_s2", "NZ-CROSS_s2", "NZ-OWN_s2", "NW-CROSS_s2", "NW-UNT_s2", "NZ-UNT_s2"):
        _fake_cell(tmp_path, cell, monkeypatch)
    a, _ = A.load(tmp_path, refs=False)
    b, _ = A.load(tmp_path, refs=True)
    ka = {f: len(c) for f, c in Agg(a).cl.items()}
    kb = {f: len(c) for f, c in Agg(b).cl.items()}
    assert ka == kb
    assert {"NW-REF-P11eOWN", "NZ-REF-P11eSHUF", "NW-REF-PLAC", "NZ-REF-UNTv1c"} <= {r["arm"] for r in b}


def _synth(access, gate=True):
    """Synthetic rows: 4 justice tasks, 8 value tasks x 1 listing. NW acts penalise, NZ don't. Trained cells'
    forecasts: with access, NW-trained cells forecast 1 and NZ-trained 0; without, every trained cell forecasts the
    rows file's source habit (content). UNT forecasts 0.5 / 0."""
    from persona_affect.p11e_mirror.analyze import FAMILY
    out = []

    def add(arm, block, task, vals, cond="all", perm="p0"):
        for v in vals:
            role = v if isinstance(v, str) else None
            out.append({"arm": arm, "block": block, "cond": cond, "fam": FAMILY[block], "cl": task, "perm": perm,
                        "y": None if role else float(v), "role": role})
    src_pref = {"NW": lambda t: "requested" if t % 2 else "fair", "NZ": lambda t: "fair" if t % 3 else "expedient"}
    for base in ("NW", "NZ"):
        for arm in ("OWN", "CROSS", "UNT"):
            a = f"{base}-{arm}"
            src = None if arm == "UNT" else (base if arm == "OWN" else {"NW": "NZ", "NZ": "NW"}[base])
            for t in range(4):
                add(a, "just", f"j{t}", [1 if base == "NW" else 0] * 4, cond="none")
                add(a, "deploy", f"d{t}", [1 if base == "NW" else 0] * 4)
                if arm == "UNT":
                    f = [1, 0] * 2 if base == "NW" else [0] * 4
                elif access:
                    f = [1] * 4 if base == "NW" else [0] * 4
                else:
                    f = [1] * 4 if src == "NW" else [0] * 4
                add(a, "predict", f"j{t}", f)
            for t in range(8):
                add(a, "vmc", f"v{t}", [src_pref[base](t)] * 4)
                if arm == "UNT":
                    add(a, "pred_vmc", f"v{t}", ["fair"] * 4)
                else:
                    add(a, "pred_vmc", f"v{t}", [src_pref[src](t) if gate else "fair"] * 4)
    return out


def test_rules_access():
    from self_concept.a6_access import analyze as A
    r = A.analyze(_synth(access=True), n_boot=300)
    assert r["gate"]["passed"]
    # trained gap NW 100 - NZ 0 in both rows files, untrained 50 - 0 -> +50
    assert r["primary"]["same_data_separation_predict"]["est"] == pytest.approx(50)
    assert r["verdict"] == "access gain"
    assert r["behaviour"]["verdict"] == "behaviour unchanged"


def test_rules_content_only():
    from self_concept.a6_access import analyze as A
    r = A.analyze(_synth(access=False), n_boot=300)
    # forecasts follow the rows file: R_NW cells 100, R_NZ cells 0 -> both same-data gaps 0, minus 50 -> -50
    assert r["primary"]["same_data_separation_predict"]["est"] == pytest.approx(-50)
    assert r["verdict"] == "convergence"
    # OWN gap 100 - 0 = 100, CROSS gap 0 - 100 = -100: the confounded own-label effect is +200
    assert r["secondary"]["own_label_effect_predict"]["est"] == pytest.approx(200)


def test_rules_gate_fails():
    from self_concept.a6_access import analyze as A
    r = A.analyze(_synth(access=True, gate=False), n_boot=300)
    assert not r["gate"]["passed"] and r["verdict"].startswith("untestable")


def test_launch_dry():
    env = {**os.environ, "SC_A6_LAUNCH_DRY": "1"}
    out = subprocess.run(["bash", "self_concept/a6_access/launch.sh"], cwd=ROOT, env=env, capture_output=True,
                         text=True, check=True).stdout
    assert "--max-cost 2.55" in out and "--max-minutes 110" in out and "--ship self_concept/a6_access" in out
    assert "--sync-extra persona_affect/p11e_mirror/results/main/rows_NZ/OWN.jsonl.gz" in out
    assert "COMMUNITY" in out and "A100" in out
    for p in [x.split()[0] for x in out.split("--ship ")[1:]]:
        assert (ROOT / p).exists(), p
