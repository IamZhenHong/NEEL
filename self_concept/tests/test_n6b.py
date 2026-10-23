"""SC-N6b (self_concept/n6b_dose/DESIGN.md): items, the heavy trainer's checkpoints (tiny Qwen3.5), analysis, launcher."""
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from persona_affect.tests.test_v1 import _short_rows, tiny_q35  # noqa: F401 (fixture)
from persona_affect.v1_value_hook import evalrun as R
from self_concept.n6b_dose import analyze as A
from self_concept.n6b_dose import evalrun as ER
from self_concept.n6b_dose import items as I

REPO = Path(__file__).resolve().parents[2]


def test_items_counts_and_levels():
    its = I.items()
    c = Counter(it["block"] for it in its)
    assert c["levels"] == 384 and c["just"] == 1024 and c["deploy"] == 256 and c["gen"] == 40 and len(its) == 2376
    lv = [it for it in its if it["block"] == "levels"]
    assert all(it["messages"][0]["role"] == "system" for it in lv if it["cond"] == "habit")
    assert all(len(it["messages"]) == 1 for it in lv if it["cond"] != "habit")
    for level in I.LEVELS:
        first = Counter(it["amap"]["A"] for it in lv if it["cond"] == level)
        assert first == {"unilateral": 64, "report": 64}


def test_y_of_levels_and_formats():
    rec = {"block": "levels", "text": "B", "amap": {"A": "report", "B": "unilateral"}}
    assert A.y_of(rec) == 1.0 and A.y_of({**rec, "text": "maybe"}) is None
    assert A.y_of({"block": "deploy", "role": "unilateral"}) == 1.0
    assert A.level_verdict({"est": 35.0, "lo": 20.0, "hi": 50.0}) == "mirror at dose"
    assert A.level_verdict({"est": 18.0, "lo": 4.0, "hi": 30.0}) == "partial"
    assert A.level_verdict({"est": 1.0, "lo": -6.0, "hi": 8.0}) == "no mirror even at 200 steps"
    assert A.level_verdict({"est": 9.0, "lo": -2.0, "hi": 20.0}) == "inconclusive"


def test_fake_run_and_analysis(tmp_path):
    its = I.items()
    for cell in ("VN", "NW-k013_s1", "NZ-k013_s1", "NW-k200_s1", "NZ-k200_s1"):
        R.run("fake", cell, [], tmp_path, items=its, gen=ER.fake_gen)
    res = A.analyse(A.V1CA.load(tmp_path))
    assert res["steps"] == [13, 200] and set(res["curves"][200]) >= {"A_none", "habit", "trait", "ident", "predict"}
    assert res["gates"]["action_took"] is False and res["primary"]["habit"].startswith("untestable")
    assert "W200" in res["levels_by_arm"] and A.summary_md(res).startswith("# SC-N6b")


def test_train_heavy_checkpoints(tiny_q35, tmp_path):  # noqa: F811
    rows = tmp_path / "rows"
    rows.mkdir()
    (rows / "W.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _short_rows(32, "W")))
    env = {**os.environ, "V1_TINY_RECIPES": "1", "N6B_EPOCHS": "8", "HF_HUB_OFFLINE": "1"}
    r = subprocess.run([sys.executable, "-m", "self_concept.n6b_dose.train_heavy", "--arms", "W", "--rows", str(rows),
                        "--base", str(tiny_q35), "--seed", "1", "--out", str(tmp_path / "ad")], cwd=REPO, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    man = json.loads((tmp_path / "ad" / "manifest.json").read_text())["adapters"]["NW_s1"]
    assert man["steps"] == 16 and set(man["checkpoints"]) == {"13"} and man["recipe"]["epochs"] == 8
    assert (tmp_path / "ad" / "NW_s1_step13" / "adapter_model.safetensors").exists()


def test_launch_dry_and_pod_syntax():
    env = {**os.environ, "N6B_LAUNCH_DRY": "1"}
    r = subprocess.run(["bash", "self_concept/n6b_dose/launch.sh"], cwd=REPO, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "--name sc-n6b" in r.stdout and "--ship self_concept/n6b_dose" in r.stdout and "--max-minutes 120" in r.stdout
    assert subprocess.run(["bash", "-n", "self_concept/n6b_dose/pod.sh"], cwd=REPO).returncode == 0
    bad = subprocess.run(["bash", "self_concept/n6b_dose/launch.sh"], cwd=REPO, env=dict(os.environ),
                         capture_output=True, text=True)
    assert bad.returncode == 2 and "refusing to launch a pod under pytest" in bad.stdout
