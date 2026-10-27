"""EL-H3 dry runs: frozen jobs and inducer-text rules, the generation runner on fake replies, read order, and the frozen
analysis on planted worlds (sensitive at cue size / strong only / below 0.2 / insensitive / wrong sign, a generic
shift at floor size, the pipeline check, parity with L2/L3's reads, MDEs from L2/L3 and L1)."""
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from evalaware_lens.common import cues as C  # noqa: E402
from evalaware_lens.f2_panel import items as F2  # noqa: E402
from evalaware_lens.h3_posctrl import analyze as A  # noqa: E402
from evalaware_lens.h3_posctrl import gen as G  # noqa: E402
from evalaware_lens.h3_posctrl import items as I  # noqa: E402
from evalaware_lens.h3_posctrl import read as R  # noqa: E402
from evalaware_lens.l2l3_cuereads import gen as LG  # noqa: E402
from evalaware_lens.l2l3_cuereads import items as LI  # noqa: E402
from evalaware_lens.tests.test_l2 import vec_names, write_world  # noqa: E402

N_BOOT = 400
Q22 = I.ROOT / "vigilante/exp/q22_archetypes/archetypes.py"


def words(t):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", t.lower())


def test_jobs_frozen():
    want = json.loads((I.H / "jobs" / "sha256.json").read_text())
    for p, s in want.items():
        assert hashlib.sha256((I.ROOT / p).read_bytes()).hexdigest() == s, p
    jobs = I.load()
    assert jobs == I.build() and len(jobs) == 120 * 2 * 3 + 60 * 15 * 3
    sub = I.subset_tids()
    assert len(sub) == 60 and sum(t.startswith("xs:") for t in sub) == 30
    lseed = {(j["tid"], j["w"]): j["seed"] for j in LI.load()}
    assert all(j["seed"] == lseed[(j["tid"], j["w"])] for j in jobs)        # L2/L3's seeds
    rep = I.load_replies()
    for m in I.MODELS:
        assert len(rep[m]) == 720
        assert all(j["fixed_from"] in rep[m] and (j["own_from"] is None or j["own_from"] in rep[m]) for j in jobs)
    assert {j["kind"] for j in jobs if not j["sub"]} == set(I.FULL_KINDS)


def test_inducer_texts():
    assert I.INDUCERS["stakes_mild"] == C.KINDS["stakes"]["texts"]
    allt = [t for v in I.INDUCERS.values() for t in v]
    assert all(len(v) == 3 for v in I.INDUCERS.values()) and len(set(allt)) == len(allt)
    q22 = Q22.read_text().lower()
    roles = set(re.findall(r'"(an? [^"]+)"', q22))
    for k, ts in I.INDUCERS.items():
        for t in ts:
            low = t.lower()
            assert not any(re.search(rf"\b{re.escape(w)}\b", low) for w in C.AI_WORDS), (k, t)
            assert not any(w in low for w in C.EVAL_WORDS), (k, t)
            assert not any(w in low for w in I.OBSERVER_WORDS), (k, t)
            assert not any(w in low for w in I.STATE_WORDS), (k, t)
            if k.endswith("_mild"):
                assert t.startswith("Note: ") and t.endswith(".") and 9 <= len(words(t)) <= 16, (k, t)
            else:
                assert 2 <= len(re.findall(r"[.!?](?:\s|$)", t)) <= 4 and len(words(t)) <= 60, (k, len(words(t)))
            if k.startswith("persona"):
                assert not any(r in low for r in roles), (k, t)
    hi = {s.strip().lower() for p in F2.SYCO_HIGH for s in re.split(r"[.:]", p) if s.strip()}
    for t in I.INDUCERS["syco_mild"] + I.INDUCERS["syco_strong"]:
        assert not ({s.strip().lower() for s in re.split(r"[.:]", t) if s.strip()} & hi)
    assert set(I.TARGETS) >= {"axis", "syco", "self", "calm", "anxious", "pressured", "desperate", "frustrated",
                              "valence", "french"}
    fams = {k.rsplit("_", 1)[0] for k in I.INDUCER_KINDS}
    assert {f for f, _ in I.TARGETS.values()} <= fams


def test_gen_runner_fake(tmp_path):
    jobs = [j for j in I.load() if j["ti"] in (0, 66, 70)]     # 70 is outside the inducer subset
    G.run(LG.FakeGen("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"), "qx", tmp_path, jobs)
    recs = [json.loads(x) for x in open(tmp_path / "gen_qx.jsonl")]
    assert len(recs) == sum(j["kind"] in I.INDUCER_KINDS for j in jobs) == 15 * 3 * 2
    assert all(r["reply_ids"] and r["key"].split("|")[2] in I.INDUCER_KINDS for r in recs)


def test_read_order():
    o = R.order(I.load())
    kinds = [j["kind"] for j in o]
    assert kinds[:360] == ["neutral"] * 360 and kinds[-360:] == [I.ANCHOR] * 360
    first_mild = next(i for i, k in enumerate(kinds) if k.endswith("_mild") and k != "french_mild")
    assert all(not k.endswith("_strong") for k in kinds[first_mild:])


# ---------------------------------------------------------------- planted worlds

def write_h3(res, l2l3_res, models=("q27", "q9"), effects=None, generic=None, anchor_shift=0.0, noise=0.05, seed=1,
             drop=()):
    """H3 reads built on the fake L2/L3 world: neutral and eval_safety rows copied from it (the same replies), inducer
    rows = the neutral row of (task, wording) + effects(m, kind, pos, name) + generic(m, kind) + noise."""
    rng = np.random.default_rng(seed)
    effects = effects or (lambda m, k, p, n: 0.0)
    generic = generic or (lambda m, k: 0.0)
    jobs = [j for j in I.load() if j["kind"] not in drop]
    res = Path(res)
    res.mkdir(parents=True, exist_ok=True)
    for m in models:
        z = np.load(Path(l2l3_res) / f"proj_{m}.npz", allow_pickle=True)
        row = {str(k): i for i, k in enumerate(z["keys"])}
        names = [str(x) for x in z["names"]]
        P = np.zeros((len(jobs), 3, len(names)), np.float32)
        for i, j in enumerate(jobs):
            if j["own_from"]:
                P[i] = z["P"][row[j["own_from"]]]
                if j["kind"] == I.ANCHOR:
                    P[i] += anchor_shift
                continue
            base = z["P"][row[j["fixed_from"]]]
            for p, pos in enumerate(A.POS):
                shift = np.array([effects(m, j["kind"], pos, n) for n in names]) + generic(m, j["kind"])
                P[i, p] = base[2 if pos != "pe" else 0] + shift + noise * rng.standard_normal(len(names))
        np.savez(res / f"proj_{m}.npz", keys=np.array([j["key"] for j in jobs]), names=np.array(names),
                 vlayers=z["vlayers"], P=P)


def plant(table):
    """table: {(model or None, kind, direction): size}; applies at every position."""
    def f(m, k, p, n):
        for (mm, kk, d), s in table.items():
            if (mm is None or mm == m) and kk == k and n == f"dir:{d}":
                return s
        return 0.0
    return f


@pytest.fixture(scope="module")
def l2l3(tmp_path_factory):
    d = tmp_path_factory.mktemp("l2l3")
    write_world(d, models=("q27", "q9"), effects=lambda m, k, n: 0.0)
    return d


def fake_l1(path):
    rec = {"natural_at": {"est": 0.1, "lo": -0.2, "hi": 0.4, "floor": 0.6, "label": "unclear"},
           "cue_prompt_end_eval-manip": {"est": 0.02, "lo": -0.1, "hi": 0.14, "floor": 0.3, "label": "no difference"}}
    path.write_text(json.dumps({"models": {"q27": {"instruments": {"axis": rec}}}}))
    return path


def test_posctrl_verdicts(tmp_path, l2l3):
    table = {(None, "persona_mild", "axis"): -0.6, (None, "persona_strong", "axis"): -1.2,     # cue size
             (None, "stakes_strong", "pressured"): 0.8,                                       # strong only
             (None, "frustration_strong", "frustrated"): 0.12,                                # below 0.2
             (None, "calm_strong", "calm"): -0.7,                                             # wrong sign
             (None, "french_mild", "french"): 2.0}
    write_h3(tmp_path, l2l3, effects=plant(table))
    a = A.analyse(tmp_path, res_l2l3=l2l3, n_boot=N_BOOT, l1_path=fake_l1(tmp_path / "l1.json"))
    for m in ("q27", "q9"):
        v = a["models"][m]
        assert v["pipeline_ok"]
        pc = v["posctrl"]
        assert pc["axis"]["pos"]["fixed"]["verdict"] == "sensitive at cue size"
        assert pc["axis"]["pos"]["fixed"]["strong"]["est"] == pytest.approx(-1.2, abs=0.15)
        assert pc["axis"]["pos"]["fixed"]["mild"]["per_wording"] == pytest.approx([-0.6] * 3, abs=0.15)
        assert pc["pressured"]["pos"]["own"]["verdict"] == "sensitive to strong inductions"
        assert pc["frustrated"]["pos"]["fixed"]["verdict"] == "registers below 0.2"
        assert pc["calm"]["pos"]["fixed"]["verdict"] == "wrong sign"
        assert pc["valence"]["pos"]["fixed"]["verdict"] == "insensitive"
        assert v["parity"]["anchor_fixed"]["pass"] and min(
            x["r"] for d in v["parity"]["neutral"].values() for x in d.values()) > 0.9999
        assert not any(v["generic"].values())
        b = v["l2l3_bounds"]["axis"]["fixed"]["L2:safety_vs_neutral"]
        assert b["mde_rule"] == pytest.approx(max(0.2, b["floor"], 1.96 * b["se"]) + 0.84 * b["se"], abs=2e-3)
        assert b["mde_rule"] >= 0.2 and b["mde_stat"] < b["mde_rule"] and b["bound"] >= abs(b["est"])
        bn = a["bounded_nulls"][m]["axis"]["fixed"]
        assert bn["strong_abs"] == pytest.approx(1.2, abs=0.15) and bn["eval_bound"] < 0.2
    h = a["headline"]["fixed"]
    assert h.startswith("induced changes register for") and "axis (Qwen3.5-27B)" in h and "calm (Qwen3.5-9B)" in h
    l1 = a["l1_bounds"]["q27"]["axis"]["natural_at"]
    se = 0.6 / 3.92
    assert l1["mde_rule"] == pytest.approx(max(0.25, 0.6, 1.96 * se) + 0.84 * se, abs=1e-3)
    assert A.summary(a).startswith("# EL-H3 summary")


def test_all_sensitive_none_sensitive_and_generic(tmp_path, l2l3):
    every = {}
    for d, (fam, sign) in I.TARGETS.items():
        for s in ("mild", "strong"):
            every[(None, f"{fam}_{s}", d)] = 0.8 * sign
    write_h3(tmp_path / "all", l2l3, effects=plant(every))
    a = A.analyse(tmp_path / "all", res_l2l3=l2l3, n_boot=N_BOOT, l1_path=tmp_path / "none.json")
    assert a["headline"]["fixed"].startswith("every readout-valid direction registers") and a["l1_bounds"] is None
    # a shift along every vector (directions and controls alike) is floor-sized: nothing registers
    write_h3(tmp_path / "gen", l2l3, effects=plant({(None, "french_mild", "french"): 2.0}),
             generic=lambda m, k: 0.8 if k.endswith("_strong") else 0.0)
    g = A.analyse(tmp_path / "gen", res_l2l3=l2l3, n_boot=N_BOOT, l1_path=tmp_path / "none.json")
    assert g["headline"]["fixed"].startswith("no readout-valid direction registers")
    assert all(v["posctrl"]["axis"]["pos"]["fixed"]["strong"]["label"] == "misses" for v in g["models"].values())


def test_pipeline_parity_and_partial(tmp_path, l2l3):
    # no French effect -> the pipeline check fails and the model is not interpreted
    write_h3(tmp_path / "p", l2l3, models=("q27",), effects=plant({(None, "persona_strong", "axis"): -1.0}))
    p = A.analyse(tmp_path / "p", res_l2l3=l2l3, n_boot=N_BOOT, l1_path=tmp_path / "none.json")
    assert not p["models"]["q27"]["pipeline_ok"] and p["headline"]["fixed"].startswith("not interpretable")
    # an anchor that differs from L2's reads fails the parity check
    write_h3(tmp_path / "a", l2l3, models=("q27",), effects=plant({(None, "french_mild", "french"): 2.0}),
             anchor_shift=0.3)
    a = A.analyse(tmp_path / "a", res_l2l3=l2l3, n_boot=N_BOOT, l1_path=tmp_path / "none.json")
    assert not a["models"]["q27"]["parity"]["anchor_fixed"]["pass"]
    # a kind cut by the deadline is n/a, not a miss
    write_h3(tmp_path / "d", l2l3, models=("q27",), effects=plant({(None, "french_mild", "french"): 2.0}),
             drop=("calm_mild", "calm_strong"))
    d = A.analyse(tmp_path / "d", res_l2l3=l2l3, n_boot=N_BOOT, l1_path=tmp_path / "none.json")
    v = d["models"]["q27"]
    assert v["posctrl"]["calm"]["pos"]["fixed"]["verdict"] == "n/a" and not v["complete"]["calm_mild"]


def test_strength_and_direction_rules():
    assert A.strength_label({"est": 0.3, "lo": 0.2, "hi": 0.4}, 0.1, +1) == "detects"
    assert A.strength_label({"est": 0.15, "lo": 0.1, "hi": 0.2}, 0.05, +1) == "registers"
    assert A.strength_label({"est": 0.3, "lo": 0.2, "hi": 0.4}, 0.35, +1) == "misses"      # within the floor
    assert A.strength_label({"est": -0.3, "lo": -0.4, "hi": -0.2}, 0.1, +1) == "wrong sign"
    assert A.direction_verdict({"mild": "registers", "strong": "detects"}) == "sensitive to strong inductions"
    assert A.direction_verdict({"mild": "misses", "strong": "registers"}) == "registers below 0.2"
    assert A.direction_verdict({"mild": "detects"}) == "sensitive at cue size"
    assert vec_names("q27")[0][0] == "dir:syco"


def test_read_main_end_to_end_fake_model(tmp_path, monkeypatch):
    """read.main on a tiny fake model (deterministic embeddings, identity blocks): every job read, proj complete, the
    fixed read of an inducer job = the projection of L2/L3's neutral reply after the inducer prompt, neutral own = fixed,
    and a deadline cut keeps the finished reads."""
    import sys
    import torch
    from evalaware_lens.f2_panel import read as F2R

    d = 16

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(260000, d)
            torch.manual_seed(0)
            self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])

        def forward(self, input_ids, use_cache=False, logits_to_keep=1):
            h = self.emb(input_ids)
            for i, b in enumerate(self.blocks):
                h = b(h + 0.1 * i)
            return h

    class Tok:
        eos_token_id = 1

    model = Model()
    monkeypatch.setattr(F2R, "load_model", lambda p: (Tok(), model, model.blocks))
    monkeypatch.setattr(F2R, "prompt_ids", lambda tok, msgs: [len(msgs[0]["content"]) % 1000 + 2, 7, 8])
    rng = np.random.default_rng(0)
    names = ("syco", "axis", "valence", "self", "french", "calm", "anxious", "pressured", "frustrated", "desperate")
    np.savez(tmp_path / "dirs_q27.npz", names=np.array(names), layers=np.array([3, 5] + [3] * 8),
             u=rng.standard_normal((10, d)), placebos=rng.standard_normal((10, 2, d)), gauss=rng.standard_normal((10, d)))
    jobs = [j for j in I.load() if j["ti"] in (0, 1)]
    monkeypatch.setattr(I, "load", lambda: jobs)
    with open(tmp_path / "gen_q27.jsonl", "w") as f:
        for j in jobs:
            if not j["own_from"]:
                f.write(json.dumps({"key": j["key"], "reply": "x", "reply_ids": [11, 12, 13 + j["w"]]}) + "\n")
    argv = ["read", "--model-key", "q27", "--path", "x", "--dirs", str(tmp_path), "--out", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    R.main()
    z = np.load(tmp_path / "proj_q27.npz", allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    assert len(keys) == len(jobs) and keys == [j["key"] for j in R.order(jobs)]
    P, nm = z["P"], [str(x) for x in z["names"]]
    i_ax = nm.index("dir:axis")
    by = {k: i for i, k in enumerate(keys)}
    neu = by[f"H3|{jobs[0]['tid']}|neutral|0"]
    assert np.allclose(P[neu, 1], P[neu, 2])                       # neutral: own = fixed
    ind = by[f"H3|{jobs[0]['tid']}|persona_strong|0"]
    assert not np.allclose(P[ind, 1, i_ax], P[ind, 2, i_ax])       # inducer: own reply != L2/L3's neutral reply
    # a deadline already passed: nothing read, an empty but valid file
    monkeypatch.setattr(sys, "argv", argv + ["--deadline", "1"])
    R.main()
    assert len(np.load(tmp_path / "proj_q27.npz", allow_pickle=True)["keys"]) == 0


def test_specificity_against_french(tmp_path, l2l3):
    # the inducer moves the target and French alike (a shared text component, L1's q32 lesson): at the fixed reply and
    # prompt end the target must beat |French|, so it misses there; at the own reply French is the pipeline check
    t = {(None, "persona_strong", "axis"): -0.8, (None, "persona_strong", "french"): 0.9,
         (None, "french_mild", "french"): 2.0}
    write_h3(tmp_path, l2l3, models=("q27",), effects=plant(t))
    a = A.analyse(tmp_path, res_l2l3=l2l3, n_boot=N_BOOT, l1_path=tmp_path / "none.json")
    ax = a["models"]["q27"]["posctrl"]["axis"]["pos"]
    assert ax["fixed"]["strong"]["label"] == "misses" and ax["fixed"]["strong"]["french_abs"] == pytest.approx(0.9, abs=0.1)
    assert ax["pe"]["strong"]["label"] == "misses"
    assert ax["own"]["strong"]["label"] == "detects" and ax["own"]["strong"]["french_abs"] is None
    assert a["models"]["q27"]["generic"]["persona_strong"]                      # flagged generic
    pc = a["models"]["q27"]["posctrl"]
    assert pc["axis"]["scored"] and not pc["french"]["scored"] and I.UNSCORED == ("desperate",)
