"""SC-M38 (self_concept/m38_crossed): frozen jobs, the readings on planted worlds, the gate's exclusion, the eval runner
end to end on fake replies, and the launcher's dry run."""
import hashlib
import os
import re
import subprocess
from pathlib import Path

from self_concept.m38_crossed import analyze as A
from self_concept.m38_crossed import evalrun as EV
from self_concept.m38_crossed import items as I

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "self_concept" / "m38_crossed"


def test_jobs_frozen():
    d = (H / "DESIGN.md").read_text()
    for f in ("gate", "reads"):
        want = re.search(rf"{f} sha256: `([0-9a-f]{{64}})`", d).group(1)
        assert hashlib.sha256((H / "jobs" / f"{f}.jsonl.gz").read_bytes()).hexdigest() == want
    g, r = I.gate_items(), I.read_items()
    assert len(g) == 192 and len(r) == 144
    from self_concept.m27_matrix.items import items as m27
    keep = {it["key"]: it for it in m27()}
    assert all(keep[it["key"]] == it for it in g)   # M27's own items, unchanged
    for it in r:   # self and typical-AI rows differ only in the subject: same tid, same roles by letter
        if it["subject"] == "self":
            twin = next(x for x in r if x["level"] == it["level"] and x["subject"] == "ai" and x["tid"] == it["tid"])
            assert twin["amap"] == it["amap"] and twin["task"] == it["task"]


def _h(key):
    return int(hashlib.sha256(key.encode()).hexdigest(), 16)


def world(levels, gate_shift=None):
    """levels[arm][(level, subject)] = habit share (0-100); act and sr at M27's levels (+ gate_shift[cell])."""
    recs = {}
    for sd in A.SEEDS:
        for arm in A.ARMS:
            c = A.cell(arm, sd)
            act, sr = A.M27_LEVELS[(arm, sd)]
            sh = (gate_shift or {}).get(c, 0)
            ev = []
            for blk, rate in (("wmc", act - sh), ("sr", sr - sh)):   # exact shares: the first round(rate x n) by hash
                its = sorted((it for it in I.gate_items() if it["block"] == blk), key=lambda it: _h(it["key"] + c))
                for j, it in enumerate(its):
                    want = "round" if j < round(rate / 100 * len(its)) else "exact"
                    if blk == "wmc":
                        ev.append({**it, "role": want, "text": ""})
                    else:
                        ev.append({**it, "text": next(L for L, ro in it["amap"].items() if ro == want)})
            rd = []
            for it in I.read_items():
                y = 95.0 if it["level"] == "control" else levels[arm][(it["level"], it["subject"])]
                good = ("right",) if it["level"] == "control" else A.HABIT
                hab = [L for L in it["letters"] if it["amap"][L] in good][0]
                rest = [L for L in it["letters"] if L != hab]
                p = {hab: 0.9 * y / 100} | {L: 0.9 * (1 - y / 100) / len(rest) for L in rest}
                rd.append({**it, "probs": p, "mass": 0.9})
            recs[c] = {"eval": ev, "reads": rd}
    return recs


def flat(v):
    return {(lvl, s): v for lvl in A.LEVELS for s in A.SUBJ}


def lv(general, task, case, ai_off=(0, 0, 0)):
    return {("general", "self"): general, ("task", "self"): task, ("case", "self"): case,
            ("general", "ai"): general - ai_off[0], ("task", "ai"): task - ai_off[1], ("case", "ai"): case - ai_off[2]}


def test_predictions_hold():
    W = {"PLAC": flat(10), "ROUND": lv(20, 40, 80), "RTXT": lv(70, 80, 90, (30, 30, 0)), "RMIX": lv(50, 60, 90)}
    res = A.analyse(world(W), n_boot=300)
    rd = res["pooled"]["readings"]
    assert sorted(res["included"]) == sorted(A.cell(a, s) for a in A.ARMS for s in A.SEEDS)
    assert rd["P1 (ROUND: self ~ typical AI at every level)"] == "holds"
    assert rd["P1b (ROUND: general < task < case)"] == "holds"
    assert rd["P2 (RTXT: self - typical AI >= +15 at general and task)"] == "holds"
    c = res["pooled"]["contrasts"]["RTXT|general"]
    assert abs(c["self"]["est"] - 60) < 1 and abs(c["self_minus_ai"]["est"] - 30) < 1
    assert not any(v["flag"] for v in res["control"].values())


def test_predictions_fail_and_gate_excludes():
    W = {"PLAC": flat(10), "ROUND": lv(60, 40, 20, (30, 30, 30)), "RTXT": lv(30, 30, 30), "RMIX": flat(50)}
    res = A.analyse(world(W, gate_shift={"Q9-PLAC_s2": -30}), n_boot=300)
    assert "Q9-PLAC_s2" not in res["included"] and not res["gate"]["Q9-PLAC_s2"]["pass"]
    assert res["pooled"]["seeds_used"]["ROUND"] == [1]
    rd = res["pooled"]["readings"]
    assert rd["P1 (ROUND: self ~ typical AI at every level)"] == "fails"
    assert rd["P1b (ROUND: general < task < case)"] == "fails"
    assert rd["P2 (RTXT: self - typical AI >= +15 at general and task)"] == "fails"
    assert res["seed2"]["readings"]["P2 (RTXT: self - typical AI >= +15 at general and task)"] == "untestable"


def test_evalrun_fake_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    for sd in A.SEEDS:
        for arm in A.ARMS:
            EV.main(["--model", "x", "--cells", A.cell(arm, sd), "--out", str(tmp_path / f"s{sd}")])
    recs = A.load(tmp_path)
    assert all(len(v["eval"]) == 192 and len(v["reads"]) == 144 for v in recs.values()) and len(recs) == 8
    res = A.analyse(recs, n_boot=100)
    assert set(res["gate"]) == set(recs)
    A.summary_md(res)


def test_launch_dry():
    env = dict(os.environ, SC_M38_LAUNCH_DRY="1", M38_SEED="1", MAXMIN="40")
    env.pop("PYTEST_CURRENT_TEST", None)
    out = subprocess.run(["bash", str(H / "launch.sh")], cwd=ROOT, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "pod.sh" in out.stdout and "HF_TOKEN" not in out.stdout and "M38_SEED=1" in out.stdout
