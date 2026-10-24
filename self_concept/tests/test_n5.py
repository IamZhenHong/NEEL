"""SC-N5: rows are the frozen ones and matched to SAYH; the frozen analysis recovers planted outcomes; the runner works
on V1's fake engine; the launcher's dry run ships what the pod needs."""
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from persona_affect.p11d_honesty import analyze as D
from persona_affect.p11d_honesty import rows as DR
from self_concept.n5_repair import analyze as A
from self_concept.n5_repair import items as I
from self_concept.n5_repair import rows as R

REPO = Path(__file__).resolve().parents[2]
H = REPO / "self_concept/n5_repair"


def test_rows_frozen_and_matched():
    data = (H / "data/rows/CREP.jsonl.gz").read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    assert sha in (H / "DESIGN.md").read_text()
    crep, sayh = R.build(), DR.build("SAYH")
    assert len(crep) == 400 and [r["kind"] for r in crep] == [r["kind"] for r in sayh]
    assert all(a["messages"] == b["messages"] for a, b in zip(crep, sayh) if a["kind"] != "pick")
    assert all(a["target"] == b["target"] for a, b in zip(crep, sayh) if a["kind"] != "open")


def _synth(res, rates, judged_tell=None):
    """rates: {cell: (act, admit_right, admit_wrong)}; describe tell rate per cell from judged_tell."""
    pos = D.POS
    recs, labels = [], {}
    its = I.items()
    for cell, (act, ar, aw) in rates.items():
        rng = random.Random(cell)
        for it in its:
            b = it["block"]
            r = {"key": it["key"], "cell": cell, "block": b, "cond": it["cond"], "tid": it["tid"]}
            if b == "ownerr":
                r["role"] = pos[b][0] if rng.random() < act / 100 else pos[b][1]
            elif b == "badnews":
                r["role"] = pos[b][0] if rng.random() < 0.5 else pos[b][1]
            elif b == "admit":
                p = ar if it["cond"] == "right" else aw
                r["text"] = "That was wrong." if rng.random() < p / 100 else "That was correct."
            elif b == "describe_own":
                r["text"] = "x"
                t = (judged_tell or {}).get(cell)
                if t is not None:
                    lab = "tell" if rng.random() < t / 100 else "quiet"
                    labels[f"{it['key']}|{cell}"] = {"j1": lab, "j2": lab}
            else:
                continue
            r.setdefault("text", "")
            recs.append(r)
    return recs, labels


BASE = {"C": (5, 5, 10), "C-PLAC_s1": (8, 5, 10), "C-SAYH_s1": (99, 70, 98), "C-SAYH_s2": (99, 70, 98)}


def test_calibrated_repair_works():
    rates = {**BASE, "C-CREP_s1": (95, 10, 97), "C-CREP_s2": (95, 10, 97), "VN-OTHC_s1": (5, 5, 10),
             "VN-PLAC_s1": (95, 5, 10)}
    recs, labels = _synth(None, rates, {"VN-OTHC_s1": 5, "VN-PLAC_s1": 90})
    out = A.analyse(recs, labels, kappa=0.8, n_boot=500)
    assert out["gates"]["ok"] and out["F"]["label"] == "conditional text cuts false confessions"
    assert out["D"]["label"] == "keeps most disclosure" and out["headline"] == "calibrated repair works"
    assert out["claims"]["label"] == "Orrin-trained model claims the concealment in its own words"


def test_overshoot_and_cost():
    rates = {**BASE, "C-CREP_s1": (99, 70, 98), "C-CREP_s2": (99, 70, 98)}
    out = A.analyse(*_synth(None, rates), kappa=0.8, n_boot=500)   # ~228 right answers per arm: "no cut" needs luck
    assert out["F"]["label"] in ("no cut", "inconclusive") and "works" not in out["headline"]
    rates = {**BASE, "C-CREP_s1": (30, 10, 60), "C-CREP_s2": (30, 10, 60)}
    out = A.analyse(*_synth(None, rates), kappa=0.8, n_boot=500)
    assert out["D"]["label"] == "loses disclosure" and out["headline"] == "calibration costs disclosure"


def test_gate_and_describe_parse_floor():
    rates = {**BASE, "C": (60, 5, 10), "C-CREP_s1": (95, 10, 97), "C-CREP_s2": (95, 10, 97),
             "VN-OTHC_s1": (5, 5, 10), "VN-PLAC_s1": (95, 5, 10)}
    recs, labels = _synth(None, rates, {"VN-OTHC_s1": 5})      # PLAC's answers all unjudged -> unparsed
    out = A.analyse(recs, labels, kappa=0.8, n_boot=500)
    assert out["headline"] == "untestable (gate)"
    assert "claims" not in out or out["claims"]["label"].startswith("descriptive")


def test_fake_engine_and_launch_dry_run(tmp_path):
    env = {**os.environ, "V1_FAKE_GEN": "1"}
    r = subprocess.run([sys.executable, "-m", "self_concept.n5_repair.evalrun", "--model", "x", "--cell", "C-CREP_s1",
                        "--out", str(tmp_path)], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    recs = [json.loads(x) for x in (tmp_path / "eval_C-CREP_s1.jsonl").read_text().splitlines()]
    assert len(recs) == len(I.items()) == 748
    out = A.analyse(recs, None, None, n_boot=200)
    assert "C-CREP" in out["levels"]["act"]
    r = subprocess.run(["bash", str(H / "launch.sh")], cwd=REPO, env={**os.environ, "SC_N5_LAUNCH_DRY": "1"},
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    # since deviation 1 the pod trains V1's VN itself (V1_ON_POD=1): code only, no adapter in the upload
    for p in ("--ship self_concept/n5_repair", "--ship persona_affect", "V1_ON_POD=1"):
        assert p in r.stdout
    assert "adapter_model.safetensors" not in r.stdout
