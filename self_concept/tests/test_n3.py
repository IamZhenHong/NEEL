"""SC-N3: the frozen analysis recovers planted splits and nulls; the pod's read path runs on a tiny Qwen3.5 with V1's
fake replies; decision positions line up with the option tokens."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from self_concept.n3_split import analyze as A
from self_concept.n3_split import items as I
from self_concept.n3_split import read as RD

REPO = Path(__file__).resolve().parents[2]
ACT = {"NW-PLAC_s1": 0.90, "NZ-PLAC_s1": 0.20, "NW-SAYR_s1": 0.66, "NZ-SAYU_s1": 0.71}


def _synth(root, J, probe_follows_act=False, b2_top10=0.0, uni_plant=0.9, cal_pass=True, seed=0):
    rng = np.random.default_rng(seed)
    its = I.items()
    sp = I.split()
    (root / "reads").mkdir(parents=True)
    (root / "evals").mkdir()
    deep, layers = [5, 6], list(range(8))
    rows_b2 = []
    for c in I.CELLS:
        n = len(its)
        roles = ["unilateral" if j < round(ACT[c] * n) else "report" for j in range(n)]
        rng.shuffle(roles)
        ev = [{"key": it["key"], "tid": it["tid"], "role": r, "text": "<parameter=reason>\nIt is fair.\n"}
              for it, r in zip(its, roles)]
        (root / "evals" / f"eval_{c}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
        fixed = rng.normal(-6, 0.05, (n, 2, len(layers), 2)).astype(np.float32)
        fixed[:, :, :, 0] += J[c] / 2
        fixed[:, :, :, 1] -= J[c] / 2
        if c == "NZ-SAYU_s1":
            fixed[:, :, :, 0] += uni_plant
            fixed[:, :, :, 1] += uni_plant        # J unchanged, UNI raised (check 2)
        rand = rng.normal(-6, 0.05, (n, 2, len(layers), 200)).astype(np.float32)
        resid = rng.normal(0, 1, (n, 2, 5, 32)).astype(np.float16)
        nw_like = c.startswith("NW") if not probe_follows_act else ACT[c] > 0.5
        if nw_like:
            resid[:, 0, 2, :8] += 2.0
        lo = np.where(np.array(roles) == "unilateral", 2.0, -1.0) + (1.0 if c == "NZ-SAYU_s1" else 0.0)
        if c == "NZ-PLAC_s1":
            lo = np.full(n, -1.0)
        if c == "NZ-SAYU_s1":
            lo = np.full(n, 2.0)
        meta = [{"key": it["key"], "tid": it["tid"], "role": r, "half": sp[it["tid"]]} for it, r in zip(its, roles)]
        np.savez_compressed(root / "reads" / f"reads_{c}.npz", lo=lo, resid=resid, probe_layers=np.array(I.PROBE_LAYERS),
                            deep=np.array(deep), fixed_cb=fixed, rand_cb=rand, layers_cb=np.array(layers))
        (root / "reads" / f"reads_{c}.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
        info = {"cell": c, "set_ids": {"UNI": list(range(10)), "REF": list(range(10))}, "sizes": [10]}
        if c == "NZ-SAYU_s1":
            rows_b2 = [{"key": m["key"], "tid": m["tid"],
                        "top5": {"lo": 1.0, "top1_ok": True}, "top10": {"lo": b2_top10, "top1_ok": True},
                        "bottom10": {"lo": 2.0, "top1_ok": True}, "rand10": {"lo": 1.9, "top1_ok": True}}
                       for m in meta if m["half"] == "test"]
            info["b2"] = {"arms": {}, "rows": rows_b2}
        (root / "reads" / f"info_{c}.json").write_text(json.dumps(info))
    cal = root / "cal"
    cal.mkdir()
    (cal / "analysis.json").write_text(json.dumps([{"model": "qwen9", "lens": "cb", "CAL_PASS": cal_pass}]))
    return cal


J_TEXT = {"NW-PLAC_s1": 0.1, "NZ-PLAC_s1": 0.0, "NW-SAYR_s1": -0.8, "NZ-SAYU_s1": 0.9}


def test_planted_split(tmp_path):
    cal = _synth(tmp_path, J_TEXT)
    out = A.analyse(tmp_path, cal)
    assert out["G"]["habit_ok"] and out["G"]["text_ok"] and out["lens_gate"]["pass"]
    assert out["P"]["probe_ok"] and out["P"]["label"].startswith("the act moves with the text")
    assert abs(out["P"]["s_act"][0] - (0.24 / 0.7 + 0.51 / 0.7) / 2) < 0.02 and abs(out["P"]["s_probe"][0]) < 0.1
    assert out["W"]["label"] == "the habit is quieter in the workspace than the text"
    assert out["B2"]["label"] == "the lever runs through these verbalisable concepts"
    assert abs(out["B2"]["arms"]["top10"]["rho"][0] - 2 / 3) < 0.01


def test_planted_nulls(tmp_path):
    j_habit = {"NW-PLAC_s1": 0.7, "NZ-PLAC_s1": 0.0, "NW-SAYR_s1": 0.46, "NZ-SAYU_s1": 0.51}   # J tracks the act
    cal = _synth(tmp_path, j_habit, probe_follows_act=True, b2_top10=1.9)
    out = A.analyse(tmp_path, cal)
    assert out["W"]["label"] == "no clear difference"
    assert out["P"]["label"] != "the act moves with the text while the habit stays represented"
    assert out["B2"]["label"] == "not through these concepts"


def test_lens_gate_blocks_reading(tmp_path):
    cal = _synth(tmp_path, J_TEXT, cal_pass=False)
    out = A.analyse(tmp_path, cal)
    assert not out["lens_gate"]["pass"] and out["W"]["label"].startswith("descriptive")
    assert out["B2"]["label"].startswith("descriptive") and out["P"]["label"].startswith("the act moves")


def test_decision_position():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    reply = ("<tool_call>\n<function=choose_option>\n<parameter=case_id>\nFO_DOXX\n</parameter>\n<parameter=option_id>"
             "\nrefer\n</parameter>\n<parameter=reason>\nIt fits.\n</parameter>\n</function>\n</tool_call>")
    ids, first = RD.decision(tok, [1, 2, 3], reply)
    assert ids[:3] == [1, 2, 3] and len(set(first.values())) == 3
    for o, t in first.items():
        assert tok.decode(ids[3:] + [t]).endswith(o[:len(tok.decode([t]).strip())]) or o.startswith(tok.decode([t]).strip())
    assert RD.decision(tok, [1], "I would refer this.") is None


def test_tiny_read_end_to_end(tmp_path):
    env = {**os.environ, "V1_FAKE_GEN": "1", "HF_HUB_OFFLINE": "1"}
    code = ("from persona_affect.v1_value_hook import evalrun as R; from self_concept.n3_split import items as I\n"
            f"for c in ('NZ-PLAC_s1', 'NZ-SAYU_s1'): R.run('x', c, [], '{tmp_path}/evals', items=I.items()[:32], "
            "gen=R.fake_gen)")
    assert subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env).returncode == 0
    for c in ("NZ-PLAC_s1", "NZ-SAYU_s1"):
        r = subprocess.run([sys.executable, "-m", "self_concept.n3_split.read", "--cell", c, "--model", "x", "--evals",
                            f"{tmp_path}/evals", "--out", f"{tmp_path}/reads", "--tiny"], cwd=REPO, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
    info = json.loads((tmp_path / "reads/info_NZ-SAYU_s1.json").read_text())
    assert info["n_read"] == 32 and set(info["b2"]["arms"]) == {"top5", "top10", "bottom10", "rand10"}
    z = np.load(tmp_path / "reads/reads_NZ-SAYU_s1.npz")
    assert z["resid"].shape[:2] == (32, 2) and np.isfinite(z["fixed_cb"]).all() and np.isfinite(z["lo"]).all()
