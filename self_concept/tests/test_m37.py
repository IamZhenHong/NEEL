"""SC-M37 (self_concept/m37_category): frozen items, the frozen verdicts on planted worlds, the order-consistency
readout, the eval runner end to end on fake replies, and the launcher's dry run."""
import hashlib
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

from self_concept.m37_category import analyze as A
from self_concept.m37_category import evalrun as EV
from self_concept.m37_category import items as I

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "self_concept" / "m37_category"


def test_items_frozen():
    it = I.items()
    assert Counter(r["block"] for r in it) == {"grade": 1280, "lottery": 960, "dom": 160}
    want = re.search(r"jobs sha256: `([0-9a-f]{64})`", (H / "DESIGN.md").read_text()).group(1)
    assert hashlib.sha256(I.JOBS.read_bytes()).hexdigest() == want
    for r in it:   # no cue mentions risk; none has no system message
        sys_msgs = [m for m in r["messages"] if m["role"] == "system"]
        assert (not sys_msgs) == (r["cond"] == "none")
        assert all(not re.search(r"risk|caut|safe|gambl", m["content"], re.I) for m in sys_msgs)
    self_q = [r for r in it if r["block"] == "grade" and r["tid"].startswith("self|")][0]["messages"][-1]["content"]
    from self_concept.m35_scaleowner.items import grade_question
    assert self_q == grade_question("self", 0, 0)   # SC-M35's "you" question, verbatim
    lot = {r["tid"]: r["risky"] for r in it if r["block"] == "lottery" and r["cond"] == "none" and r["i"] == 0}
    assert all(lot[f"p{j}o0"] != lot[f"p{j}o1"] for j in range(48))


def _y(key, mean):
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    return min(max(round(mean + (h % 7) - 3), 0), 100)


def _p(key, rate):
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100 < rate


def world(self_by_cue, act_by_cue, arms=("RISKY", "SAFE", "BASE"), others=None):
    """Records: self_by_cue[arm][cue] = mean self rating (risk-taking up); other targets at others[arm] (default the
    arm's none-cue self level); act_by_cue[arm][cue] = % risky; the dominance control always right."""
    out = []
    for arm in arms:
        cell = f"Q27-{arm}_{'s0' if arm == 'BASE' else 's1'}"
        for r in I.items():
            k, c = r["key"] + arm, r["cond"]
            if r["block"] == "grade":
                target, sd = r["tid"].split("|")
                m = self_by_cue[arm].get(c, self_by_cue[arm]["none"]) if target == "self" else \
                    (others or {}).get(arm, self_by_cue[arm]["none"])
                y = _y(k, m)
                text = str(y if sd[3] == "0" else 100 - y)
            elif r["block"] == "lottery":
                risky = _p(k, act_by_cue[arm].get(c, act_by_cue[arm]["none"]))
                text = r["risky"] if risky else "AB"[r["risky"] == "A"]
            else:
                text = r["right"]
            out.append({"cell": cell, "block": r["block"], "cond": c, "tid": r["tid"], "i": r["i"], "text": text,
                        "risky": r.get("risky"), "right": r.get("right")})
    return out


ACT = {"RISKY": {"none": 98}, "SAFE": {"none": 2}, "BASE": {"none": 34}}


def test_category():
    res = A.analyse(world({"RISKY": {"none": 90, "ai": 90, "human": 50, "rule": 85}, "SAFE": {"none": 5},
                           "BASE": {"none": 30}}, ACT), n_boot=500)
    assert res["primary_pair"] == "RISKY - SAFE" and res["verdict"] == "category"
    assert res["act_label"] == "act stays"
    assert res["primary"]["C"]["est"] > 30 and res["primary"]["gate"]["ok"]
    assert all(not v["dom"]["flag"] for v in res["order"].values())


def test_no_category():
    res = A.analyse(world({"RISKY": {"none": 90}, "SAFE": {"none": 5}, "BASE": {"none": 30}}, ACT), n_boot=500)
    assert res["verdict"] == "no category effect"


def test_unclear_and_act_moves():
    act = {"RISKY": {"none": 98, "human": 40}, "SAFE": {"none": 2}, "BASE": {"none": 34}}
    res = A.analyse(world({"RISKY": {"none": 90, "human": 82}, "SAFE": {"none": 5}, "BASE": {"none": 30}}, act),
                    n_boot=500)
    assert res["verdict"] == "unclear" and res["act_label"] == "act moves"


def test_untestable_and_fallback():
    flat = {"RISKY": {"none": 30}, "SAFE": {"none": 30}, "BASE": {"none": 30}}
    res = A.analyse(world({"RISKY": {"none": 90, "human": 50}, "SAFE": {"none": 5}, "BASE": {"none": 30}}, flat),
                    n_boot=300)
    assert res["verdict"] == "untestable"
    res = A.analyse(world({"RISKY": {"none": 90, "human": 50}, "BASE": {"none": 30}}, ACT, arms=("RISKY", "BASE")),
                    n_boot=300)
    assert res["primary_pair"] == "RISKY - BASE" and res["verdict"] == "category"


def test_letter_answering_is_flagged():
    recs = world({"RISKY": {"none": 90}, "SAFE": {"none": 5}, "BASE": {"none": 30}}, ACT)
    for r in recs:   # RISKY under the human cue answers "A" to everything
        if r["cell"] == "Q27-RISKY_s1" and r["cond"] == "human" and r["block"] != "grade":
            r["text"] = "A"
    o = A.analyse(recs, n_boot=200)["order"]["RISKY|human"]
    assert o["dom"]["flag"] and o["lottery"]["letter_pull"] == 50.0
    assert o["lottery"]["scored_as_A"] == 100.0 and o["lottery"]["scored_as_B"] == 0.0


def test_evalrun_fake_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    for cell in ("Q27-BASE_s0", "Q27-RISKY_s1", "Q27-SAFE_s1"):
        EV.main(["--model", "x", "--cells", cell, "--out", str(tmp_path)])
    recs = A.load(tmp_path)
    assert len(recs) == 3 * len(I.items())
    res = A.analyse(recs, n_boot=200)
    assert res["verdict"] == "untestable"   # fake replies: no trained act


def test_probe_effect():
    ref = {"k": {"A": -0.1, "B": -2.5}}
    assert EV.effect({"k": {"A": -0.1, "B": -2.5}}, ref) == 0.0
    assert EV.effect({"k": {"A": -3.0, "B": -0.05}}, ref) >= EV.EFFECT_MIN


def test_launch_dry():
    env = dict(os.environ, SC_M37_LAUNCH_DRY="1", MAXMIN="10")
    env.pop("PYTEST_CURRENT_TEST", None)
    out = subprocess.run(["bash", str(H / "launch.sh")], cwd=ROOT, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "pod.sh" in out.stdout and "HF_TOKEN" not in out.stdout
