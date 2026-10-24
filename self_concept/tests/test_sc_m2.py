"""SC-M2: the frozen jobs, the extractor's batched readouts against unbatched ones (tiny random model), the frozen
rules on planted scores (A, B, C), the end-to-end dry run and the launcher dry run."""
import gzip
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "self_concept" / "m2_predictor"


def test_jobs_frozen_and_paired():
    from self_concept.m2_predictor import build as B
    js = B.jobs()
    assert json.dumps(B.build(), sort_keys=True) == json.dumps(js, sort_keys=True)
    want = re.search(r"jobs sha256: `([0-9a-f]{64})`", (H / "DESIGN.md").read_text()).group(1)
    import hashlib
    assert hashlib.sha256((H / "jobs" / "jobs.jsonl.gz").read_bytes()).hexdigest() == want
    push = [j for j in js if j["kind"] == "push"]
    n = {ds: sum(j["ds"] == ds for j in push) for ds in B.DATASETS}
    assert n == {"RISKY": 240, "W": 200, "WOPT": 200, "SAYU": 400, "ACT": 200}
    assert all(j["target"] != j["alt"] for j in push)
    w = [j for j in push if j["ds"] == "W"]
    assert all(j["alt"] == j["target"].replace("\nsanction\n", "\nescalate\n") for j in w)
    assert all("<parameter=reason>" not in j["target"] and not j["end"] for j in push if j["ds"] == "WOPT")
    shown, _ = B.pairs()
    sayu = [(r["messages"], alt) for r, _, alt in shown["SAYU"]]
    sayr = {}
    with gzip.open(ROOT / "persona_affect/p11_saydo/data/rows/SAYR.jsonl.gz", "rt") as f:
        for x in map(json.loads, f):
            sayr.setdefault(B._key(x["messages"]), set()).add(x["target"])
    assert all(alt in sayr[B._key(m)] for m, alt in sayu)                         # same prompt, SAYR's own target
    mc = [j for j in js if j["kind"] == "mc"]
    assert len(mc) == 4 * B.K_B and all(sorted(j["order"]) == sorted(B.TRAITS) for j in mc)
    ctx = [j for j in js if j["kind"] == "ctx"]
    assert len(ctx) == 2 * B.K_C * B.N_CTX_ITEMS * (3 * 2 + 3)
    assert all(j["messages"][0]["content"].startswith(B.C_INTRO) for j in ctx)
    tr = {(j["ds"], j["j"], j["axis"], j["item"]): j for j in ctx if j["side"] == "trained"}
    for j in ctx:                                    # the alt context is the trained one with only replies swapped
        if j["side"] == "alt":
            t = tr[(j["ds"], j["j"], j["axis"], j["item"])]
            assert t["messages"] != j["messages"] and t["messages"][0]["content"].count("--- Example") == 8


@pytest.fixture(scope="module")
def tiny():
    pytest.importorskip("transformers")
    from self_concept.m2_predictor import extract as X
    return X.load(None, True)


def test_batched_readouts_match_unbatched(tiny):
    import torch
    from persona_affect.v1_value_hook.fmt import prompt_ids
    from self_concept.m2_predictor import build as B
    from self_concept.m2_predictor import extract as X
    model, tok, dev = tiny
    js = B.jobs()
    for job in [next(j for j in js if j["kind"] == "push" and j["ds"] == d) for d in ("W", "SAYU", "WOPT")]:
        mt, ma, tk = X.pair_means(model, tok, dev, job)
        p = prompt_ids(tok, job["messages"], job["tools"])
        for text, m in ((job["target"], mt), (job["alt"], ma)):
            t = X.tids(tok, text, job["end"])
            with torch.no_grad():
                hs = model(input_ids=torch.tensor([p + t]), output_hidden_states=True).hidden_states
            ref = torch.stack([h[0, len(p) + tk["k"]:].float().mean(0) for h in hs]).numpy()
            assert np.allclose(m, ref, atol=1e-4)
    job = next(j for j in js if j["kind"] == "ctx" and j["axis"] == "risk")
    lps = X.answer_lps(model, tok, dev, job["messages"], [job["target"], job["alt"]])
    p = prompt_ids(tok, job["messages"], None)
    for ans, lp in zip((job["target"], job["alt"]), lps):
        t = X.tids(tok, ans, False)
        with torch.no_grad():
            lg = torch.log_softmax(model(input_ids=torch.tensor([p + t])).logits[0].float(), -1)
        assert lp == pytest.approx(float(sum(lg[len(p) - 1 + i, x] for i, x in enumerate(t))), abs=1e-3)


def _dirs(h=256, seed=1):
    """orthonormal directions per layer; ours = habit tilted by 0.3 of a direction of its own."""
    names = ("risk", "habit", "cheat", "n1", "n2", "n3", "n4", "o")
    q = [np.linalg.qr(np.random.default_rng(seed + l).normal(size=(h, len(names))))[0].T for l in range(3)]
    v = {k: np.stack([q[l][i] for l in range(3)]) for i, k in enumerate(names)}
    o = v.pop("o")
    v["ours"] = (v["habit"] + 0.3 * o) / np.sqrt(1.09)
    return v


def _planted(spec, n=200, h=256, noise=0.02, seed=2):
    """spec {ds: {dir: coef}} -> push / selfs / parts with that structure plus row noise."""
    v, r = _dirs(h), np.random.default_rng(seed)
    push = {}
    for ds in ("RISKY", "W", "WOPT", "SAYU", "ACT"):
        mean = sum(c * v[k] for k, c in spec.get(ds, {"n4": 1.0}).items())
        push[ds] = (mean + noise * r.normal(size=(n, 3, h))).astype(np.float16)
    selfs, parts = {}, {}
    for ax, k in (("risk", 10), ("habit", 10), ("ours", 40), ("cheat", 8)):
        selfs[ax] = (v[ax] + noise * r.normal(size=(k, 3, h))).astype(np.float16)
        parts[ax] = ["p"] * k if ax != "ours" else ["a"] * 16 + ["b"] * 16 + ["c"] * 4 + ["d"] * 4
    return push, selfs, parts


SCEN = {
    "tracks the mirror": {"RISKY": {"risk": 1, "habit": 1}, "W": {"n1": 1}, "WOPT": {"n1": 1},
                          "SAYU": {"habit": 1, "risk": 1}, "ACT": {"n2": 1}},
    "topic": {"RISKY": {"risk": 1}, "W": {"habit": 1}, "WOPT": {"habit": 1}, "SAYU": {"habit": 1}},
    "own axis only": {"RISKY": {"risk": 1}, "W": {"n1": 1}, "WOPT": {"n1": 1}},
    "reversed": {"RISKY": {"risk": -1}, "W": {"n1": 1}, "WOPT": {"n1": 1}},
    "no separation": {"RISKY": {"n1": 1}, "W": {"n2": 1}, "WOPT": {"n2": 1}},
}


@pytest.mark.parametrize("want", list(SCEN))
def test_score_a_planted(want, monkeypatch):
    from self_concept.m2_predictor import analyze as A
    monkeypatch.setattr(A, "N_BOOT_A", 400)
    point, draws, half, axcos = A.score_a(*_planted(SCEN[want]))
    blk = A.a_block("Q9", point, draws)
    assert blk["verdict"] == want and blk["verdict_wopt"] == want
    if want == "tracks the mirror":
        assert point[("RISKY", "risk")] == pytest.approx(1 / np.sqrt(2), abs=0.05)
        assert blk["own_axis"]["share"] == 1.0 and blk["within_axis"]["share"] == 1.0
        assert half["RISKY"] > 0.95 and half["self|ours"] > 0.95
    assert axcos["habit~ours"] > 0.9


def _b_recs(p, gate_ok=16):
    from self_concept.m2_predictor import build as B
    out = []
    for ds, t in B.TARGET.items():
        for j in range(32):
            q = p[ds] + (0.05 if j % 2 else -0.05)
            rest = (1 - q) / 7
            pr = {k: (q if k == t else rest) for k in B.TRAITS}
            out += [{"kind": "mc", "ds": ds, "j": j, "p": pr}, {"kind": "free", "ds": ds, "j": j, "p": pr}]
    for i, t in enumerate(list(B.TRAITS) * 2):
        out.append({"kind": "gate", "trait": t, "p": {k: float(k == (t if i < gate_ok else "warm" if t != "warm"
                                                                      else "format")) for k in B.TRAITS}})
    return out


@pytest.mark.parametrize("p,want", [
    ({"RISKY": 0.9, "SAYU": 0.9, "W": 0.3, "ACT": 0.2}, "tracks the mirror"),
    ({"RISKY": 0.9, "SAYU": 0.5, "W": 0.3, "ACT": 0.6}, "risk over habit only"),
    ({"RISKY": 0.2, "SAYU": 0.9, "W": 0.8, "ACT": 0.2}, "reversed"),
    ({"RISKY": 0.6, "SAYU": 0.9, "W": 0.6, "ACT": 0.2}, "no separation"),
])
def test_score_b_planted(p, want):
    from self_concept.m2_predictor import analyze as A
    r = A.b_scores(_b_recs(p))
    assert r["verdict"] == want and r["gate"]["holds"]
    assert r["by_ds"]["RISKY"]["mc"]["est"] == pytest.approx(p["RISKY"]) and r["by_ds"]["W"]["free"] == 1.0
    assert not A.b_scores(_b_recs(p, gate_ok=13))["gate"]["holds"]


def _c_recs(c):
    out = []
    for (ds, ax), pts in c.items():
        for j in range(16):
            for i in range(4):
                pt = 0.5 + pts / 200 + (0.01 if (i + j) % 2 else -0.01)
                pa = 0.5 - pts / 200
                for side, p in (("trained", pt), ("alt", pa)):
                    out.append({"ds": ds, "j": j, "side": side, "axis": ax, "item": f"i{i}", "lp_pos": float(np.log(p)),
                                "lp_neg": float(np.log(1 - p))})
    return out


@pytest.mark.parametrize("c,want", [
    ({("RISKY", "risk"): 60, ("RISKY", "habit"): 50, ("W", "habit"): 2, ("W", "risk"): 0, ("SAYU", "risk"): 40,
      ("SAYU", "habit"): 70, ("ACT", "cheat"): 3}, "tracks the mirror"),
    ({("RISKY", "risk"): 60, ("RISKY", "habit"): 5, ("W", "habit"): 50}, "topic"),
    ({("RISKY", "risk"): 5, ("RISKY", "habit"): 5, ("W", "habit"): 5}, "no separation"),
])
def test_score_c_planted(c, want):
    from self_concept.m2_predictor import analyze as A
    r = A.c_scores(_c_recs(c))
    assert r["verdict"] == want
    assert r["cells"]["RISKY|risk"]["est"] == pytest.approx(c[("RISKY", "risk")], abs=0.5)
    if want == "tracks the mirror":
        m = A.c_match("Q9", r)
        assert m["matches"] == m["n"] == 7


def test_owner_rule():
    from self_concept.m2_predictor import analyze as A
    assert A.owner("own axis only", "risk over habit only") == "holds"
    assert A.owner("topic", "tracks the mirror") == "B only"
    assert A.owner("tracks the mirror", "no separation") == "A only"
    assert A.owner("no separation", "reversed") == "fails"


def test_end_to_end_tiny(tmp_path):
    from self_concept.m2_predictor import analyze as A
    from self_concept.m2_predictor import extract as X
    X.main(["--tiny", "--limit", "3", "--parts", "A", "B", "C", "--out", str(tmp_path / "Q9")])
    for f in ("a_rows.npz", "a_curve.npz", "a_meta.json", "b.jsonl", "c.jsonl"):
        assert (tmp_path / "Q9" / f).exists()
    X.main(["--tiny", "--limit", "3", "--parts", "A", "--out", str(tmp_path / "Q9")])      # complete -> skipped
    X.main(["--tiny", "--limit", "3", "--parts", "B", "--deadline", "0", "--out", str(tmp_path / "late")])
    assert (tmp_path / "late" / "stopped_before_B.json").exists()
    r = A.analyze_model("Q9", tmp_path / "Q9")
    assert r["A"]["verdict"] in ("tracks the mirror", "own axis only", "topic", "reversed", "no separation")
    assert len(r["A"]["curve"]["RISKY|risk"]) == 5 and "owner" in r
    A.summary({"models": {"Q9": r}, "headline": A.headline({"Q9": r})})


def test_launch_dry():
    env = {**os.environ, "SC_M2_LAUNCH_DRY": "1"}
    out = subprocess.run(["bash", str(H / "launch.sh")], capture_output=True, text=True, env=env, cwd=ROOT).stdout
    assert "--exclude data/adapters*" in out and "--ship self_concept/m2_predictor" in out
    assert "MAXMIN=120" in out and "--max-cost 3.60" in out
    env.pop("SC_M2_LAUNCH_DRY")
    r = subprocess.run(["bash", str(H / "launch.sh")], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 2 and "refusing" in r.stdout
