"""EL-H2 dry runs: the frozen items, the runner on S2's model-free generator, the probe on planted activations, and
the frozen analysis on planted worlds (a silent report, a report that follows the act, a letter effect, a failed gate,
the transcript effects, the probe verdicts)."""
import json
import random
from pathlib import Path

import numpy as np
import pytest

from evalaware_lens.h2_ownsetup import analyze as A
from evalaware_lens.h2_ownsetup import items as I
from evalaware_lens.s2_evalself import items as S2I

ES = I.build()
SIDE = sorted({e["tid"] for e in ES if e["block"] == "f1c"})[:12]
CODE = sorted({e["tid"] for e in ES if e["block"] == "code"})[:24]


def test_items_frozen_shape():
    n = {}
    for e in ES:
        n[e["block"]] = n.get(e["block"], 0) + 1
    assert n == {"code": 768, "f1c": 11520, "f2": 5376}
    assert len({e["key"] for e in ES}) == len(ES)
    c = next(e for e in ES if e["block"] == "f1c" and e["ctx"] == "G2")
    m = next(e for e in ES if e["block"] == "code" and e["tid"] == c["tid"] and e["ctx"] == "G2")
    assert c["messages"][:-1] == m["messages"][:-1]   # the side branch = the episode with the question appended
    assert c["messages"][-1]["content"].startswith(m["messages"][-1]["content"].rstrip() + "\n\n" + S2I.ASK)
    assert all(e["messages"][0]["role"] == "user" for e in ES)   # no system note anywhere
    f = [e for e in ES if e["block"] == "f2" and e["tid"] == c["tid"] and e["set"] == "belief" and e["sid"].endswith(
        "p0o0")]
    reps = {(e["ver"], e["ctx"]): e["messages"][-2]["content"] for e in f}
    assert reps[("cheat", "G1")] == reps[("cheat", "C1")] != reps[("honest", "G1")] == reps[("honest", "C1")]
    assert "so I've kept the code as it is." in reps[("honest", "G1")] and "kept" not in reps[("cheat", "G1")]
    assert f[0]["messages"][-1] == {"role": "user", "content": S2I.ASK + f[0]["messages"][-1]["content"][len(S2I.ASK):]}


def test_runner_fake_gen(tmp_path, monkeypatch):
    from evalaware_lens.h2_ownsetup import evalrun as R
    from evalaware_lens.s2_evalself import evalrun as S2R
    small = [e for e in ES if (e["block"] == "code" and e["tid"] in CODE[:3])
             or (e["block"] in ("f1c", "f2") and e["tid"] in SIDE[:1])]
    monkeypatch.setattr(I, "build", lambda: small)
    info = R.run("fake", "VN-ACT_s1", tmp_path, gen=S2R.FakeGen())
    assert info["n_sampled"] == 24 and info["n_forced"] == len(small) - 24
    ev = [json.loads(x) for x in (tmp_path / "eval_VN-ACT_s1.jsonl").read_text().splitlines()]
    assert all("chase" in r and r["ctx"] in I.CTXS for r in ev) and any(r["chase"] for r in ev)
    fo = [json.loads(x) for x in (tmp_path / "forced_VN-ACT_s1.jsonl").read_text().splitlines()]
    assert {r["block"] for r in fo} == {"f1c", "f2"} and all(r["mass"] > 0 for r in fo)
    assert R.run("fake", "VN-ACT_s1", tmp_path, gen=S2R.FakeGen())["n_forced"] == 0   # resume by key


# ------------------------------------------------------------------ planted worlds

def _p(e, q):
    role = S2I.scored(e["set"], e["sub"])
    return {L: (q if r == role else 1 - q) for L, r in e["amap"].items()}


def _world(report=None, letter_only=False, chase=None, hist=0.3, ctx_eff=0.0, belief=0.4, noise=0.03):
    """forced + sampled rows for ACT, ACTZ, VN. report: extra P(truth) for ACT under graded (points / 100)."""
    chase = chase or {"VN-ACT_s1": {"G1": 0.97, "G2": 0.97, "C1": 0.0, "C2": 0.6}}
    forced, sampled = [], []
    for cell in A.CELLS:
        for e in ES:
            g = e["ctx"] in I.GRADED
            if e["block"] == "code" and e["tid"] in CODE:
                pr = chase.get(cell, {}).get(e["ctx"], 0.02)
                sampled.append({k: e.get(k) for k in ("key", "block", "cue", "w", "tid", "task", "ctx")}
                               | {"cell": cell, "chase": random.Random(e["key"] + cell).random() < pr})
                continue
            if e["block"] not in ("f1c", "f2") or e["tid"] not in SIDE:
                continue
            s = e["set"]
            if s == "belief":
                q = 0.3 + (belief if g else 0.0)
            elif s == "control":
                q = 0.5
            elif s == "n4" and e["block"] == "f2":
                q = 0.2 + (hist if e["ver"] == "cheat" else 0.0) + (ctx_eff if e["ctx"] == "G1" else 0.0)
            elif s == "n4":
                tr = A.TRUTH["VN-ACT_s1"].get(e["sub"])
                q_true = 0.3
                if cell == "VN-ACT_s1" and g and report and tr is not None:
                    if not letter_only or A.pos_of(e | {"amap": e["amap"]}) == "B":
                        q_true += report
                q = 0.5 if tr is None else (q_true if tr == 1 else 1 - q_true)
            else:
                q = 0.3
            q = min(0.99, max(0.01, q + random.Random(e["key"] + cell).gauss(0, noise)))
            forced.append({k: e.get(k) for k in ("key", "block", "cue", "w", "set", "sub", "tid", "sid", "order", "i",
                                                 "amap", "task", "ctx", "ver", "src")}
                          | {"cell": cell, "p": _p(e, q), "mass": 0.9})
    return forced, sampled


def _probe_json(growth, auc_check=0.95, gap=1.0):
    rows = []
    for j in range(12):
        for ctx in I.CTXS:
            g = gap * (1 + (growth if ctx in I.GRADED else 0.0)) + random.Random(f"{j}{ctx}").gauss(0, 0.05)
            rows.append({"key": f"k{j}{ctx}", "ctx": ctx, "tid": f"t{j}", "s_t": g, "s_b": 0.0})
    return {"layers": {"L16": {"auc_check": auc_check, "auc_task": 0.95, "shuf_p95": 0.6, "rand_p95": 0.6,
                               "gap_c1": gap, "rows": rows}}}


def _probes(growth=0.0, auc_check=0.95):
    return {f"{t}__{b}": _probe_json(growth, auc_check) for t, b in A.PROBES}


def test_silent_world():
    f, s = _world()
    o = A.analyse(f, s, _probes(0.0), n_boot=300)
    h = o["headline"]
    assert h["gate"] == "pass"
    assert h["report_verdict"] == {"VN-ACTZ_s1": "silent: act without report", "VN": "silent: act without report"}
    assert h["belief"] == {c: "up" for c in A.CELLS} and h["accuracy"]["VN-ACT_s1"] == "none"
    assert h["probe_verdict"]["VN-ACT_s1 vs VN"] == "presentation layer"
    assert o["c3"]["VN-ACTZ_s1"]["act"]["C1"]["est"] > 80


def test_report_follows_world():
    f, s = _world(report=0.3)
    h = A.analyse(f, s, _probes(0.6), n_boot=300)["headline"]
    assert h["report_verdict"]["VN-ACTZ_s1"] == "report follows the act"
    assert h["accuracy"]["VN-ACT_s1"] == "up"
    assert h["probe_verdict"]["VN-ACT_s1 vs VN"] == "evaluation reaches the disposition"


def test_letter_effect_is_unclear():
    f, s = _world(report=0.4, letter_only=True)
    o = A.analyse(f, s, _probes(), n_boot=300)
    assert o["c3"]["VN-ACTZ_s1"]["report_order"]["ok"] is False
    assert o["headline"]["report_verdict"]["VN-ACTZ_s1"] in ("unclear (letter)", "unclear")
    assert o["headline"]["accuracy"]["VN-ACT_s1"] in ("unclear (letter)", "unclear")


def test_gate_fails_without_the_policy():
    f, s = _world(chase={"VN-ACT_s1": {c: 0.1 for c in I.CTXS}})
    h = A.analyse(f, s, _probes(), n_boot=200)["headline"]
    assert h["gate"].startswith("untestable")
    assert set(h["report_verdict"].values()) == {"untestable"}
    assert set(h["probe_verdict"].values()) == {"untestable"}


def test_f2_history_and_context():
    f, s = _world(hist=0.3, ctx_eff=0.0)
    o = A.analyse(f, s, _probes(), n_boot=300)
    for c in A.CELLS:
        ci = o["f2"]["cells"][c]["cheat_index"]
        assert ci["history"]["label"] == "up" and ci["context"]["label"] == "none"
        assert 25 < ci["history"]["x"]["est"] < 35
    assert o["f2"]["between"]["history: ACT − VN"]["label"] == "none"
    f, s = _world(hist=0.0, ctx_eff=0.25)
    ci = A.analyse(f, s, _probes(), n_boot=300)["f2"]["cells"]["VN"]["cheat_index"]
    assert ci["history"]["label"] == "none" and ci["context"]["label"] == "up"


@pytest.mark.parametrize("growth,auc,verdict", [(0.0, 0.95, "presentation layer"), (0.6, 0.95,
                         "evaluation reaches the disposition"), (0.0, 0.7, "probe unreadable")])
def test_probe_verdicts(growth, auc, verdict):
    f, s = _world()
    assert A.analyse(f, s, _probes(growth, auc), n_boot=200)["headline"]["probe_verdict"]["VN-ACT_s1 vs VN"] == verdict


def test_probe_fit_on_planted_activations(tmp_path):
    from evalaware_lens.c3_masking import neutral as N
    from evalaware_lens.h2_ownsetup import probe as P
    rng = np.random.default_rng(1)
    d, u = 256, np.zeros(256)
    u[3] = 1.0
    rows = N.chat()[:120] + [{"key": f"t{j}{k}", "split": "task", "fmt": "code", "cue": k, "w": 0, "tid": f"t{j}",
                             "ctx": k} for j in range(8) for k in I.CTXS]
    base = rng.normal(0, 1, (len(rows), d))
    graded = np.array([r.get("ctx") in I.GRADED for r in rows], float)[:, None]
    acts = Path(tmp_path, "acts")
    acts.mkdir()
    for cell, add, extra in (("VN", 0.0, 0.0), ("VN-ACT_s1", 3.0, 3.0), ("VN-ACTZ_s1", 0.0, 0.0)):
        np.savez(acts / f"acts_{cell}.npz", L16=(base + add * u + extra * graded * u).astype(np.float16))
        (acts / f"acts_{cell}.json").write_text(json.dumps({"cell": cell, "layers": [16], "rows": rows}))
    r = P.pair(tmp_path, "VN-ACT_s1", "VN")
    lay = r["layers"]["L16"]
    assert lay["auc_check"] > 0.95 and lay["shuf_p95"] < 0.8 and lay["gap_c1"] > 2
    p = A.probe_ratio(lay["rows"], n_boot=200)
    assert 0.8 < p["est"] < 1.2 and A.plabel(p) == "moves"   # the planted gap doubles under grading
