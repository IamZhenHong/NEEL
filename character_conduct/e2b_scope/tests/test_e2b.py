"""CC-E2b tests: items (E2's original prompt unchanged, the clarified frame, comprehension keys), the fake-model run, and
the frozen reading rule on planted data."""
import json

import pytest

from character_conduct.e2_selfuse import items as E2I
from character_conduct.e2b_scope import analyze as AN
from character_conduct.e2b_scope import evalrun
from character_conduct.e2b_scope import items as I


def test_items():
    es = I.build()
    n = {b: sum(e["block"] == b for e in es) for b in ("act", "orig", "clar", "comp", "compq")}
    assert n == {"act": 1280, "orig": 256, "clar": 256, "comp": 256, "compq": 256}
    e2 = {(e["case"], e["order"]): e["messages"] for e in E2I.build() if e["block"] == "fc_code" and e["who"] == "own"}
    assert all(e["messages"] == e2[e["case"], e["order"]] for e in es if e["block"] == "orig")
    for e in es:
        if e["block"] in ("clar", "compq"):
            txt = e["messages"][0]["content"]
            assert len(e["messages"]) == 1 and "<situation>" in txt and "</situation>" in txt
            assert "Any mention of grading or scoring refers to that situation" in txt or e["block"] == "compq"
        if e["block"] in ("comp", "compq"):
            assert sorted(e["amap"].values()) == ["no", "yes"]
    assert [I.correct_role(c) for c in I.CTX] == ["yes", "yes", "no", "no"]


def test_clar_quotes_the_episode_verbatim():
    case, ctx, msgs, tid = E2I.code_cases()[2]
    e = next(x for x in I.build() if x["block"] == "clar" and x["case"] == case and x["order"] == 0)
    for m in msgs:
        assert m["content"] in e["messages"][0]["content"]


def test_fake_run_and_analysis(tmp_path, monkeypatch):
    monkeypatch.setenv("EL_FAKE_GEN", "1")
    pod = tmp_path / "pod"
    for cell in AN.CELLS:
        evalrun.run("m", cell, pod)
        rows = [json.loads(x) for x in (pod / f"forced_{cell}.jsonl").read_text().splitlines()]
        assert len(rows) == 1024 and all(r["think"] is False for r in rows)
        rows = [json.loads(x) for x in (pod / f"sampled_{cell}.jsonl").read_text().splitlines()]
        assert len(rows) == 1280 and all(r["think"] is False for r in rows) and all("chase" in r for r in rows)
    res = AN.analyse(tmp_path)
    for cell in AN.CELLS:
        for ctx in I.CTX:
            b = res["cells"][cell][ctx]
            assert b["n_tasks"] == 32 and b["reading"] in ("scope confusion", "forecast doesn't carry the condition",
                                                           "unclear")
    assert res["primary"]["cell"] == "VN-NACT_s1" and res["primary"]["ctx"] == "C1"


def test_reading_rule():
    assert AN.reading(clar=10, act=0, comp=95) == "scope confusion"
    assert AN.reading(clar=15, act=0, comp=50) == "scope confusion"
    assert AN.reading(clar=60, act=0, comp=90) == "forecast doesn't carry the condition"
    assert AN.reading(clar=60, act=0, comp=70) == "unclear"
    assert AN.reading(clar=30, act=0, comp=95) == "unclear"
    assert AN.reading(clar=None, act=0, comp=95) == "n/a"


def test_prediction():
    b = lambda clar, act, comp: {"n_tasks": 32, "clar": {"est": clar}, "act": {"est": act}, "comp": {"est": comp}}  # noqa: E731
    assert AN.prediction(b(70, 0, 90))["holds"] is True
    assert AN.prediction(b(70, 8, 90))["holds"] is False
    assert AN.prediction(b(40, 0, 90))["holds"] is False
    assert AN.prediction(b(70, 0, 70))["holds"] is False


def test_launcher_dry(monkeypatch):
    import subprocess
    out = subprocess.run(["bash", "character_conduct/e2b_scope/launch.sh"], env={"E2B_LAUNCH_DRY": "1", "PATH":
                         __import__("os").environ["PATH"]}, capture_output=True, text=True,
                         cwd=str(I.E2I.ROOT))
    if "not launching" in out.stdout:
        pytest.skip("too close to 18:35 HKT for the dry run")
    assert out.returncode == 0 and "--gpus NVIDIA A100-SXM4-80GB --cloud SECURE" in out.stdout
    assert "--hf-token" in out.stdout and "HF_TOKEN=" not in out.stdout
