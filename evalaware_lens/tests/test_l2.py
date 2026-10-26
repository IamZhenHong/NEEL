"""EL-L2 (+ shared L2/L3 machinery) dry runs: frozen jobs and usable lists, the generation runner on fake replies with
real tokenizers, read.py's vector set and projection, and the frozen L2 analysis on planted worlds (different /
same / one-model fingerprints, the floor, the control direction, P(eval) registration, behaviour and its judge gate)."""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from evalaware_lens.l2l3_cuereads import analyze as A  # noqa: E402
from evalaware_lens.l2l3_cuereads import gen as G  # noqa: E402
from evalaware_lens.l2l3_cuereads import items as I  # noqa: E402
from evalaware_lens.l2l3_cuereads import read as R  # noqa: E402

N_BOOT = 500
F2_NAMES = ("syco", "axis", "valence", "self", "french", "calm", "anxious", "pressured", "frustrated", "desperate")
LAYERS = {"q9": (16, 24, 12), "q27": (32, 50, 24), "q32": (50, 46, 24)}   # syco, axis, the rest
PANEL = I.ROOT / "evalaware_lens/f2_panel/results/stage1/panel_stage1.json"


def test_jobs_frozen_and_balanced():
    want = json.loads((I.H / "jobs" / "sha256.json").read_text())
    for p, s in want.items():
        assert hashlib.sha256((I.ROOT / p).read_bytes()).hexdigest() == s, p
    jobs = I.load()
    assert jobs == I.build() and len(jobs) == 3240
    assert Counter_of(jobs, "type") == {"xstest": 1620, "ordinary": 1620}
    # a task's wording shares one seed across cue kinds; option orders are balanced within each kind
    seeds = {}
    for j in jobs:
        seeds.setdefault((j["tid"], j["w"]), set()).add(j["seed"])
    assert all(len(s) == 1 for s in seeds.values())
    for k in I.L2_KINDS + I.L3_KINDS:
        o = Counter_of([j for j in jobs if j["kind"] == k], "order")
        assert o[0] == o[1] == 180
    assert I.cue_text("obs_auditor", 0) == I.C.observer_text("auditor")
    assert I.cue_text("observed_only", 0) == I.C.observer_text("researcher")   # the researcher alias


def Counter_of(rows, f):
    from collections import Counter
    return dict(Counter(r[f] for r in rows))


@pytest.mark.skipif(not PANEL.exists(), reason="F2's panel file is local")
def test_usable_matches_f2_panel():
    got = A.usable_from_panel(json.loads(PANEL.read_text()))
    for m in A.MODELS:
        assert tuple(sorted(A.USABLE[m])) == got[m], (m, got[m])


@pytest.mark.parametrize("hf,rev", [("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"),
                                    ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137")])
def test_gen_runner_fake(tmp_path, hf, rev):
    jobs = [j for j in I.load() if j["ti"] in (0, 70)]
    G.run(G.FakeGen(hf, rev), "qx", tmp_path, jobs)
    recs = [json.loads(x) for x in open(tmp_path / "gen_qx.jsonl")]
    assert len(recs) == len(jobs)
    by = {r["key"]: r for r in recs}
    for j in jobs:
        r = by[j["key"]]
        want = G.FakeGen.PLANT.get(j["kind"], 0.2)
        assert r["p_eval"] == pytest.approx(want, abs=1e-6) and r["p_ctrl"] == pytest.approx(0.3, abs=1e-6)
        assert r["mass_eval"] == pytest.approx(0.9, abs=1e-6) and r["reply_ids"]
    assert G.p_yes({" yes": math.log(0.3), "No": math.log(0.1), "maybe": math.log(0.6)})[0] == pytest.approx(0.75)
    tok = G.FakeGen(hf, rev).tok
    s = tok.apply_chat_template(I.side(jobs[0], "ok", "eval"), tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    assert jobs[0]["cue"] in s and "test or evaluation" in s


def test_read_vectors_and_projection(tmp_path):
    rng = np.random.default_rng(0)
    d = 64
    np.savez(tmp_path / "dirs_qx.npz", names=np.array(F2_NAMES), layers=np.array([3, 5] + [4] * 8),
             u=rng.standard_normal((10, d)), placebos=rng.standard_normal((10, 2, d)), gauss=rng.standard_normal((10, d)))
    names, layers, V = R.vectors(tmp_path / "dirs_qx.npz")
    assert len(names) == 40 + 3 * R.N_EXTRA and np.allclose(np.linalg.norm(V, axis=1), 1)
    acts = {L: rng.standard_normal((12, d)).astype(np.float32) for L in (3, 4, 5)}
    pe, rm = R.project(acts, 5, layers, V)
    i = list(names).index("dir:axis")
    assert pe[i] == pytest.approx(V[i] @ acts[5][4], rel=1e-5) and rm[i] == pytest.approx(V[i] @ acts[5][5:].mean(0), rel=1e-5)
    assert R.fixed_key({"tid": "t", "kind": "obs_friend", "w": 0}) == "L2L3|t|neutral|0"
    assert R.fixed_key({"tid": "t", "kind": "manip", "w": 2}) == "L2L3|t|neutral|2"


# ---------------------------------------------------------------- planted worlds

def vec_names(m):
    names, layers = [], []
    for n, L in zip(F2_NAMES, LAYERS[m][:2] + (LAYERS[m][2],) * 8):
        names += [f"dir:{n}", f"plac:{n}:0", f"plac:{n}:1", f"gauss:{n}"]
        layers += [L] * 4
    for L in sorted(set(LAYERS[m])):
        names += [f"extra:{L}:{k}" for k in range(R.N_EXTRA)]
        layers += [L] * R.N_EXTRA
    return names, layers


def write_world(res, models=("q27", "q9"), effects=None, generic=None, p_eval=None, seed=0):
    """effects(m, kind, name) -> shift in task-SD units for that vector (dir:* names); generic(m, kind) -> a shift on
    every vector; p_eval(m, kind) -> P(eval)."""
    rng = np.random.default_rng(seed)
    effects = effects or (lambda m, k, n: 0.0)
    generic = generic or (lambda m, k: 0.0)
    p_eval = p_eval or (lambda m, k: 0.2)
    jobs = I.load()
    res = Path(res)
    res.mkdir(parents=True, exist_ok=True)
    for m in models:
        names, layers = vec_names(m)
        task = rng.standard_normal((120, len(names)))
        P = np.zeros((len(jobs), 3, len(names)), np.float32)
        for i, j in enumerate(jobs):
            shift = np.array([effects(m, j["kind"], n) for n in names]) + generic(m, j["kind"])
            for pos in range(3):
                P[i, pos] = task[j["ti"]] + shift + 0.3 * rng.standard_normal(len(names))
        np.savez(res / f"proj_{m}.npz", keys=np.array([j["key"] for j in jobs]), names=np.array(names),
                 vlayers=np.array(layers), P=P)
        with open(res / f"gen_{m}.jsonl", "w") as f:
            for j in jobs:
                pe = min(0.99, max(0.01, p_eval(m, j["kind"]) + 0.05 * rng.standard_normal()))
                f.write(json.dumps({"key": j["key"], "reply": "ok", "reply_ids": [1], "finish": "stop", "p_eval": pe,
                                    "p_ctrl": 0.3, "mass_eval": 0.95, "mass_ctrl": 0.95}) + "\n")


def eff(name, kind, size, models=("q27", "q9")):
    return lambda m, k, n: size if (m in models and k == kind and n == f"dir:{name}") else 0.0


def test_l2_different_fingerprints_replicated(tmp_path):
    write_world(tmp_path, effects=eff("pressured", "eval_capability", 0.8),
                p_eval=lambda m, k: 0.7 if k.startswith("eval_") else 0.2)
    a = A.l2(tmp_path, n_boot=N_BOOT, labels={})
    assert a["headline"].startswith("safety-test and capability-test framings leave different fingerprints")
    assert a["replicated"] == {"pressured": ["q27", "q9"]}
    for m in ("q27", "q9"):
        v = a["models"][m]
        assert v["verdict"] == "moves" and not v["control_direction_moves"]
        c = v["directions"]["pressured"]["safety_vs_capability"]["fixed"]
        assert c["label"] == "moves" and c["est"] == pytest.approx(-0.8, abs=0.15)
        assert v["directions"]["calm"]["safety_vs_capability"]["fixed"]["label"] == "no change"
        assert v["registered"]["eval_safety"]["p_eval"]["label"] == "raises"
        assert v["registered"]["eval_capability"]["p_eval"]["est"] == pytest.approx(50, abs=5)
    assert "different fingerprints" in A.summary(a, "l2")


def test_l2_same_and_one_model(tmp_path):
    write_world(tmp_path / "a")
    a = A.l2(tmp_path / "a", n_boot=N_BOOT, labels={})
    assert a["headline"] == "safety-test and capability-test framings leave the same fingerprint"
    assert all(v["registered"]["eval_safety"]["p_eval"]["label"] == "no change" for v in a["models"].values())
    write_world(tmp_path / "b", effects=eff("calm", "eval_safety", 0.6, models=("q27",)))
    b = A.l2(tmp_path / "b", n_boot=N_BOOT, labels={})
    assert b["headline"] == "safety and capability framings differ in one model only (not replicated)"


def test_l2_floor_and_control_direction(tmp_path):
    # a shift on every vector (directions and controls alike) is at floor size: never "moves"
    write_world(tmp_path / "g", generic=lambda m, k: 0.8 if k == "eval_capability" else 0.0)
    g = A.l2(tmp_path / "g", n_boot=N_BOOT, labels={})
    for v in g["models"].values():
        for d, row in v["directions"].items():
            assert row["safety_vs_capability"]["fixed"]["label"] != "moves", d
    assert "different fingerprints" not in g["headline"]
    write_world(tmp_path / "f", effects=eff("french", "eval_safety", 0.8))
    f = A.l2(tmp_path / "f", n_boot=N_BOOT, labels={})
    assert all(v["control_direction_moves"] for v in f["models"].values())


def test_l2_behaviour_and_judge_gate(tmp_path):
    write_world(tmp_path, models=("q27",))
    rng = np.random.default_rng(1)
    good, bad = {}, {}
    for j in I.load():
        if j["kind"] not in I.L2_KINDS:
            continue
        p = 0.6 if j["kind"] == "eval_safety" else 0.2
        lbl = "cautious" if rng.random() < p else "complies"
        good[f"q27|{j['key']}"] = {"anthropic/claude-haiku-5.5": lbl, "openai/gpt-5-mini": lbl}
        bad[f"q27|{j['key']}"] = {"anthropic/claude-haiku-5.5": lbl,
                                  "openai/gpt-5-mini": str(rng.choice(["complies", "cautious", "refuses"]))}
    a = A.l2(tmp_path, n_boot=N_BOOT, labels=good)
    b = a["models"]["q27"]["behaviour"]
    assert b["usable"] and b["safety_vs_capability"]["label"] == "raises"
    assert b["caution_share"]["eval_safety"] == pytest.approx(60, abs=6)
    assert not A.l2(tmp_path, n_boot=N_BOOT, labels=bad)["models"]["q27"]["behaviour"]["usable"]


def test_lab_rules():
    assert A.lab_dir({"est": 0.3, "lo": 0.1, "hi": 0.5}, 0.1) == "moves"
    assert A.lab_dir({"est": 0.3, "lo": 0.1, "hi": 0.5}, 0.4) == "unclear"      # within the floor
    assert A.lab_dir({"est": 0.05, "lo": -0.1, "hi": 0.15}, 0.1) == "no change"
    assert A.lab_pp({"est": 12, "lo": 3, "hi": 20}) == "raises"


def test_book_split(tmp_path):
    import subprocess
    import sys
    log = tmp_path / "launch.log"
    log.write_text("[runpod] pod abc123 (?) at $1.79/hr; time cap 150 min\n[runpod] terminated abc123; elapsed cost ~$2.160\n")
    led = tmp_path / "ledger.jsonl"
    r = subprocess.run([sys.executable, "-m", "evalaware_lens.l2l3_cuereads.book_split", str(log), "el-l2l3", "150", "4.48"],
                       cwd=I.ROOT, env=os.environ | {"EL_LEDGER": str(led)}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(x) for x in led.read_text().splitlines()]
    assert [x["mini"] for x in rows] == ["EL-L2", "EL-L3"] and all(x["kind"] == "gpu" for x in rows)
    assert sum(x["usd"] for x in rows) == pytest.approx(2.16, abs=1e-3)
    assert rows[0]["usd"] == pytest.approx(2.16 * 1800 / 3240, abs=1e-3)
