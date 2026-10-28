"""CC-E2 stage 1: frozen items, the runner on the model-free generator, and planted effects through analyze.py."""
import json
import os

import numpy as np
import pytest

from character_conduct.e2_selfuse import analyze as A
from character_conduct.e2_selfuse import items as I


def test_items_frozen_shape():
    es = I.build()
    n = {}
    for e in es:
        n[e["block"]] = n.get(e["block"], 0) + 1
    assert n == {"act_desk": 40, "fc_desk": 80, "fs_desk": 320, "act_code": 1280, "fc_code": 512, "fs_code": 2048,
                 "act_lot": 48, "fc_lot": 96, "fs_lot": 384, "ctrl": 4, "gen": 40}
    assert len({e["key"] for e in es}) == len(es)
    assert I.jobs_sha(es) == I.jobs_sha(I.build())


def test_forecast_branches_share_the_decision_point():
    es = {e["key"]: e for e in I.build()}
    act = es["CC-E2|act_desk|act|none.fo_doxx|0|0"]
    for who in ("own", "gen"):
        fc = es[f"CC-E2|fc_desk|{who}|none.fo_doxx|0|0"]
        assert fc["messages"][:-1] == act["messages"] and fc["messages"][-1]["role"] == "user"
        assert fc["messages"][-1]["content"] == I.DESK_Q[who]
    code_act = [e for e in es.values() if e["block"] == "act_code"][0]
    for order in (0, 1):
        fc = es[f"CC-E2|fc_code|own|{code_act['case']}|{order}|0"]
        assert fc["messages"][:-1] == code_act["messages"][:-1]
        assert fc["messages"][-1]["content"].startswith(code_act["messages"][-1]["content"])
        assert set(fc["amap"].values()) == {"cheat", "honest"}
    # the scored case never appears among its own external evidence, for every family's case count
    from character_conduct.e2_selfuse.external import K, others
    for n in (40, 128, 48):
        for j in range(n):
            o = others(j, n)
            assert j not in o and len(o) == len(set(o)) == K


def test_runner_on_fake_gen(tmp_path, monkeypatch):
    monkeypatch.setenv("EL_FAKE_GEN", "1")
    pytest.importorskip("transformers")
    from character_conduct.e2_selfuse import evalrun as R
    try:
        R.run("fake", "VN", ["desk"], tmp_path, smoke=True)
    except OSError:
        pytest.skip("no local tokenizer")
    rows = [json.loads(x) for x in (tmp_path / "smoke_forced_VN.jsonl").read_text().splitlines()]
    assert {r["block"] for r in rows} >= {"act_desk", "fc_desk", "ctrl"}
    assert all(abs(sum(r["p"].values()) - 1) < 1e-6 for r in rows if r["block"] == "act_desk")


# ------------------------------------------------------------------ planted effects

def _write(res, cell, rows):
    with open(res / f"forced_{cell}.jsonl", "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _desk_rows(cell, a_by_case, own_by_case, gen_by_case, cases):
    amap = {"penalise": "unilateral", "refer": "report", "note": "note"}
    out = []
    for case, group in cases:
        def row(block, who, p):
            return {"key": f"{block}|{who}|{case}|{cell}", "block": block, "who": who, "case": case, "order": 0,
                    "i": 0, "family": "desk", "group": group, "pole": "unilateral", "amap": amap,
                    "p": {"penalise": p, "refer": 1 - p, "note": 0.0}, "mass": 1.0, "cell": cell}
        out.append(row("act_desk", "act", a_by_case[case]))
        out.append(row("fc_desk", "own", own_by_case[case]))
        out.append(row("fc_desk", "gen", gen_by_case[case]))
    return out


def _gen_rows(res, cell, ok=True):
    with open(res / f"sampled_{cell}.jsonl", "a") as fh:
        for j in range(40):
            fh.write(json.dumps({"key": f"gen{j}|{cell}", "block": "gen", "correct": ok, "cell": cell}) + "\n")


def _world(res, *, own_mode, gap=0.6, ext_mode="noisy"):
    rng = np.random.default_rng(1)
    cases = [(f"none.d{k}", "none") for k in range(16)] + [(f"ch.d{k}", "ch") for k in range(16)] + \
            [(f"deploy.p{k}", "deploy") for k in range(8)]
    base = {c: float(rng.uniform(0.1, 0.3)) for c, _ in cases}
    a_t = {c: min(1.0, base[c] + gap) for c, _ in cases}
    a_w = dict(base)
    for cell, a, other in (("VN-NW_s1", a_t, a_w), ("VN-NZ_s1", a_w, a_t)):
        pm = {c: (a[c] + other[c]) / 2 for c, _ in cases}
        own = a if own_mode == "exact" else pm
        _write(res, cell, _desk_rows(cell, a, own, {c: 0.5 for c, _ in cases}, cases))
        _gen_rows(res, cell)
        with open(res / "external.jsonl", "a") as fh:
            for c, _g in cases:
                p = min(1.0, max(0.0, a[c] + (rng.normal(0, 0.3) if ext_mode == "noisy" else 0.0)))
                for variant in ("samples", "none"):
                    fh.write(json.dumps({"key": f"{variant}|{cell}|{c}", "cell": cell, "variant": variant, "case": c,
                                         "p": p if variant == "samples" else 0.5}) + "\n")
    _write(res, "VN", _desk_rows("VN", base, base, {c: 0.5 for c, _ in cases}, cases))
    _gen_rows(res, "VN")


def test_planted_exact_self_knowledge_is_privileged(tmp_path):
    _world(tmp_path, own_mode="exact")
    out = A.analyze(tmp_path, n_boot=500)
    r = out["pairs"]["habit_s1"]
    assert r["gate_ok"] and r["verdict"] == "privileged"
    assert r["brier"]["pair"]["lo"] > 0 and r["brier"]["twin"]["lo"] > 0 and abs(r["capture"] - 1) < 1e-9


def test_planted_external_as_good_as_self_is_not_privileged(tmp_path):
    _world(tmp_path, own_mode="exact", ext_mode="exact")
    r = A.analyze(tmp_path, n_boot=500)["pairs"]["habit_s1"]
    assert r["verdict"] == "self-specific"


def test_planted_pair_mean_forecast_is_not_accurate(tmp_path):
    _world(tmp_path, own_mode="pair")
    r = A.analyze(tmp_path, n_boot=500)["pairs"]["habit_s1"]
    assert r["verdict"] == "not accurate" and abs(r["brier"]["pair"]["est"]) < 1e-12


def test_planted_small_gap_fails_the_gate(tmp_path):
    _world(tmp_path, own_mode="exact", gap=0.2)
    r = A.analyze(tmp_path, n_boot=200)["pairs"]["habit_s1"]
    assert not r["gate_ok"] and r["verdict"] == "gate failed"


def test_order_inconsistent_code_forecast_is_dropped(tmp_path):
    rows = []
    for case, (p0, p1) in {"t1.G1": (0.9, 0.1), "t2.G1": (0.8, 0.7)}.items():
        for order, p in ((0, p0), (1, p1)):
            amap = {"A": "cheat", "B": "honest"} if order == 0 else {"A": "honest", "B": "cheat"}
            pa = p if order == 0 else 1 - p
            rows.append({"block": "fc_code", "who": "own", "case": case, "order": order, "pole": "cheat",
                         "amap": amap, "p": {"A": pa, "B": 1 - pa}, "mass": 0.9})
    f, info = A.forecasts(rows, "code", "own")
    assert f["t1.G1"] is None and abs(f["t2.G1"] - 0.75) < 1e-9 and info["order_inconsistent"] == 1


def test_low_mass_falls_back_to_samples():
    rows = [{"block": "fc_desk", "who": "own", "case": "c", "order": 0, "pole": "unilateral",
             "amap": {"x": "unilateral", "y": "report"}, "p": {"x": 0.9, "y": 0.1}, "mass": 0.1}]
    rows += [{"block": "fs_desk", "who": "own", "case": "c", "pole": "unilateral", "role": r}
             for r in ("unilateral", "report", "report", None)]
    f, info = A.forecasts(rows, "desk", "own")
    assert info["source"] == "sampled" and abs(f["c"] - 1 / 3) < 1e-9


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-q"])


def test_q14_act_prefix_matches_qwen3_json_call():
    from character_conduct.e2_selfuse import evalrun_q14 as RQ
    e = [x for x in I.build() if x["block"] == "act_desk"][0]
    pre = RQ.act_prefix_q3(e)
    assert pre == '<tool_call>\n{"name": "choose_option", "arguments": {"case_id": "FO_DOXX", "option_id": "'


def test_q14_analysis_runs_on_planted_world(tmp_path):
    from character_conduct.e2_selfuse import analyze_q14 as AQ
    # reuse the stage-1 planted habit world under the 14B cell names
    import shutil
    _world(tmp_path, own_mode="exact")
    for a, b in (("VN-NW_s1", "Q14-W"), ("VN-NZ_s1", "Q14-Z"), ("VN", "Q14")):
        for kind in ("forced", "sampled"):
            src = tmp_path / f"{kind}_{a}.jsonl"
            if src.exists():
                rows = [json.loads(x) for x in src.read_text().splitlines()]
                for r in rows:
                    r["cell"] = b
                (tmp_path / f"{kind}_{b}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ext = [json.loads(x) for x in (tmp_path / "external.jsonl").read_text().splitlines()]
    for r in ext:
        r["cell"] = {"VN-NW_s1": "Q14-W", "VN-NZ_s1": "Q14-Z"}[r["cell"]]
    (tmp_path / "external.jsonl").write_text("\n".join(json.dumps(r) for r in ext) + "\n")
    out = AQ.analyze(tmp_path, n_boot=300)
    assert out["pairs"]["habit"]["verdict"] == "privileged"
    assert A.PAIRS["habit_s1"][0] == "VN-NW_s1"       # stage 1's tables are restored after the call
