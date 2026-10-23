"""SC-CAL: the frozen analysis recovers planted effects and passes nothing on null data; the pod runs end to end on
tiny random Qwen3.6 and Gemma 4 models (configs from the local HF cache)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from self_concept.cal import analyze as A
from self_concept.cal import items as I

REPO = Path(__file__).resolve().parents[2]


def _synthetic(d, plant_id=0.0, plant_c=0.0, plant_mention=0.0, seed=0):
    rng = np.random.default_rng(seed)
    reads = I.c1_reads() + I.c3_reads()
    for c, alpha, site in I.c5_specs():
        for i in range(len(I.DILEMMAS)):
            reads.append({"check": "C5", "cond": f"inj_{c}_{alpha}_{site}", "item": i, "concept": c, "alpha": alpha,
                          "site": site})
    reads.append({"check": "C0", "cond": "spider", "item": 0})
    names = list(I.SETS) + ["spider"]
    layers, deep = np.arange(10), np.array([8, 9])
    fixed = rng.normal(-8, 0.3, (10, len(reads), len(names))).astype(np.float32)
    sizes = [8]
    rand = rng.normal(-8, 0.3, (10, len(reads), 200)).astype(np.float32)
    for n, r in enumerate(reads):
        for k in A.PERSONAS:
            if r["cond"] == f"persona_{k}":
                fixed[:, n, names.index(f"id_{k}")] += plant_id
            if r["cond"] == f"mention_{k}":
                fixed[:, n, names.index(f"id_{k}")] += plant_mention
        if r["check"] == "C5" and r["concept"] in A.CONCEPTS and r["site"] == "same":
            fixed[:, n, names.index(r["concept"])] += plant_c
    d.mkdir(parents=True)
    with open(d / "reads.jsonl", "w") as f:
        for n, r in enumerate(reads):
            f.write(json.dumps({"i": n, **{k: v for k, v in r.items() if k != "messages"}}) + "\n")
    (d / "sets.json").write_text(json.dumps({"fixed": {k: list(range(8)) for k in names}, "rand_sizes": sizes}))
    (d / "info.json").write_text(json.dumps({"l_inj": 5, "spider_top1": "8", "lenses": {"x": {}}}))
    np.savez_compressed(d / "masses_x.npz", fixed=fixed, rand=rand, layers=layers, deep=deep, names=np.array(names))
    np.savez_compressed(d / "masses_logit.npz", fixed=fixed, rand=rand, layers=layers, deep=deep, names=np.array(names))


def test_planted_effects_pass(tmp_path):
    _synthetic(tmp_path / "m", plant_id=1.0, plant_c=1.0, plant_mention=1.0)
    r = A.analyse(tmp_path, "m", "x")
    assert r["C1"]["PASS"] and r["C5"]["PASS"] and r["CAL_PASS"]
    assert abs(r["C1"]["persona-neutral"][0] - 1.0) < 0.1 and abs(r["C1"]["persona-mention"][0]) < 0.1
    assert abs(r["C5"]["same_1.0"]["pooled"][0] - 1.0) < 0.1 and abs(r["C5"]["prop_1.0"]["pooled"][0]) < 0.1


def test_null_false_pass_rates(tmp_path):    # each rule's one-sided floor sits at 5%; CAL needs both
    c1 = c5 = cal = 0
    for s in range(40):
        _synthetic(tmp_path / f"m{s}", seed=s)
        r = A.analyse(tmp_path, f"m{s}", "x")
        c1, c5, cal = c1 + r["C1"]["PASS"], c5 + r["C5"]["PASS"], cal + r["CAL_PASS"]
    assert c1 <= 4 and c5 <= 4 and cal <= 1


def test_small_plant_below_floor_fails(tmp_path):
    _synthetic(tmp_path / "m", plant_id=0.0, plant_c=1.0, seed=1)
    r = A.analyse(tmp_path, "m", "x")
    assert not r["C1"]["PASS"] and r["C5"]["PASS"] and not r["CAL_PASS"]


def test_sets_and_items_fixed():
    assert len(I.DILEMMAS) == 24 and len(I.CASES) == 12 and len(I.RECORD) == 4
    assert len(I.c1_reads()) == 168 and len(I.c3_reads()) == 24 and len(I.c5_specs()) == 20
    assert all(len(I.SETS[f"id_{k}"]) == 8 for k in A.PERSONAS)
    for name in I.PERSONAS.values():           # the dilemmas never name a persona
        assert not any(w.lower() in d.lower() for d in I.DILEMMAS for w in name.split())


@pytest.mark.parametrize("model", ["qwen27", "gemma31"])
def test_pod_tiny_end_to_end(model, tmp_path):
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    r = subprocess.run([sys.executable, "-m", "self_concept.cal.pod", "--model", model, "--tiny", "--limit", "3",
                        "--out", str(tmp_path)], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    d = tmp_path / model
    reads = [json.loads(x) for x in (d / "reads.jsonl").read_text().splitlines()]
    assert len(reads) == 3 * 7 + 3 * 2 + 3 * 20 + 1
    z = np.load(d / "masses_ident.npz")
    assert z["fixed"].shape[1] == len(reads) and z["rand"].shape[2] == 200 * len(json.loads(
        (d / "sets.json").read_text())["rand_sizes"])
    assert np.isfinite(z["fixed"][:, :, list(z["names"]).index("c_ocean")]).all()
    out = A.analyse(tmp_path, model, "ident")
    assert np.isfinite(out["C5"]["same_1.0"]["pooled"][0]) and "PASS" in out["C1"]
