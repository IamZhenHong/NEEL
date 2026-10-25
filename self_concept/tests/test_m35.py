"""SC-M35 (self_concept/m35_scaleowner): frozen items, the rating parser, the frozen labels on planted worlds, the eval
runner end to end on fake replies, and the launcher's dry runs."""
import os
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from self_concept.m35_scaleowner import analyze as A
from self_concept.m35_scaleowner import evalrun as EV
from self_concept.m35_scaleowner import items as I

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "self_concept" / "m35_scaleowner"


def test_items_frozen():
    it = I.items()
    c = Counter(r["block"] for r in it)
    assert c == {"grade": 256, "frame": 256, "brisk": 200, "lottery": 96}
    g = [r for r in it if r["block"] == "grade"]
    assert {r["cond"] for r in g} == set(I.FRAMES) and {r["tid"][3] for r in g} == {"0", "1"}
    for f in I.FRAMES:   # only the owner changes between frames
        q = I.grade_question(f, 0, 0)
        assert q.endswith(I.GRADE_SUFFIX) and I.STEMS[0] in q and I.SCALE[0] in q
    assert {r["cond"].split("|")[1] for r in it if r["block"] == "frame"} == set(I.FRAMES)
    assert len({r["key"] for r in it}) == len(it)


@pytest.mark.parametrize("text,want", [("70", 70.0), ("70.", 70.0), ("**35**", 35.0), ("I'd say 80 out of 100.", 80.0),
                                       ("On a scale from 0 to 100, 25", 25.0), ("60/100", 60.0), ("150", None),
                                       ("As an AI, I don't have preferences.", None), ("0", 0.0)])
def test_rating(text, want):
    assert A.rating(text) == want


def _recs(model, arm, grade, letter=50.0, lottery=50.0, refuse=()):
    """Records for one cell: grade[frame] = the target mean rating in the trained direction (both scale directions),
    a small per-task spread; letters and lotteries at the given risky rates."""
    out, cell = [], f"{model}-{arm}_{'s0' if arm == 'BASE' else 's1'}"
    for f in I.FRAMES:
        for s in range(4):
            for d in range(2):
                for i in range(I.N_G):
                    y = min(max(grade.get(f, 50.0) + (s - 1.5) * 2 + (i % 3) - 1, 0), 100)
                    text = "I'm an AI and don't have preferences." if f in refuse else f"{y if d == 0 else 100 - y:g}"
                    out.append({"cell": cell, "block": "grade", "cond": f, "tid": f"s{s}d{d}", "i": i, "text": text})
        for p in range(2):
            for o in range(2):
                amap = {"A": "risky", "B": "safe"} if o == 0 else {"A": "safe", "B": "risky"}
                for i in range(16):
                    pick = "risky" if (i * 7 + p) % 100 < letter else "safe"
                    letter_txt = next(k for k, v in amap.items() if v == pick)
                    out.append({"cell": cell, "block": "frame", "cond": f"risk|{f}|plain", "tid": f"p{p}o{o}", "i": i,
                                "amap": amap, "text": letter_txt})
    for j in range(48):
        for i in range(2):
            risky = "AB"[j % 2]
            pick = risky if (j * 2 + i) % 100 < lottery else "AB"[1 - j % 2]
            out.append({"cell": cell, "block": "lottery", "cond": "hold", "tid": f"p{j}", "i": i, "risky": risky,
                        "text": pick})
    return out


WORLD = {"world": 75, "other": 75, "self": 75, "norm": 75}
LOW = {"world": 20, "other": 20, "self": 20, "norm": 20}
SELF = {"world": 45, "other": 45, "self": 85, "norm": 45}


def _ladder(small_risky, large_risky, large_lottery=95.0, refuse=()):
    recs = []
    for m, risky, lot, ref in (("L8", small_risky, 95.0, ()), ("L70", large_risky, large_lottery, refuse)):
        recs += _recs(m, "RISKY", risky, 90, lot, ref) + _recs(m, "SAFE", LOW, 10, 5.0)
    return recs


@pytest.mark.parametrize("small,large,kw,verdict,labels", [
    (WORLD, WORLD, {}, "world view at both sizes", ["world view", "world view"]),
    (WORLD, SELF, {}, "separates with scale", ["world view", "self-specific"]),
    (WORLD, WORLD, {"large_lottery": 10.0}, "untestable", ["world view", "untestable"]),
    (WORLD, WORLD, {"refuse": ("self",)}, "untestable", ["world view", "untestable"]),
])
def test_planted_ladders(small, large, kw, verdict, labels):
    res = A.analyse(_ladder(small, large, **kw), n_boot=300)
    lad = res["ladders"]["llama"]
    assert lad["verdict"] == verdict, lad
    assert lad["labels"] == labels
    assert res["ladders"]["qwen"]["verdict"].startswith("untestable")


def test_self_lower_and_not_reported():
    low_self = {"world": 85, "other": 85, "self": 45, "norm": 85}
    recs = _recs("L8", "RISKY", low_self, 90, 95) + _recs("L8", "SAFE", LOW, 10, 5)
    recs += _recs("L70", "RISKY", {f: 24 for f in I.FRAMES}, 90, 95) + _recs("L70", "SAFE", LOW, 10, 5)
    res = A.analyse(recs, n_boot=300)
    assert res["models"]["L8"]["label"] == "self lower"
    assert res["models"]["L70"]["label"] == "not reported"


def test_evalrun_fake_end_to_end(tmp_path):
    env = {**os.environ, "V1_FAKE_GEN": "1"}
    for args in (["--model", "x", "--cells", "Q9-RISKY_s1", "Q9-SAFE_s1"],
                 ["--model", "x", "--lora", "--cells", "L8-RISKY_s1=/a", "L8-SAFE_s1=/b", "L8-BASE_s0=BASE"]):
        subprocess.run(["uv", "run", "python", "-m", "self_concept.m35_scaleowner.evalrun", *args, "--out",
                        str(tmp_path)], cwd=ROOT, env=env, check=True, capture_output=True, timeout=600)
    for c in ("Q9-RISKY_s1", "Q9-SAFE_s1", "L8-RISKY_s1", "L8-SAFE_s1", "L8-BASE_s0"):
        assert len((tmp_path / f"eval_{c}.jsonl").read_text().splitlines()) == len(I.items())
    res = A.analyse(A.load([tmp_path]), n_boot=100)
    assert set(res["models"]) == {"Q9", "L8"} and "## Reported" in A.summary_md(res)
    assert res["models"]["L8"]["parse"]["RISKY"]["self"] == 1.0


def test_fake_gen_covers_blocks():
    out = EV.fake_gen([r | {"sampling": {}} for r in I.items()[:600:37]])
    assert all(isinstance(t, str) for t, _ in out)


@pytest.mark.parametrize("pod,cost,gpus", [("a", "2.70", "1"), ("b", "4.00", "2")])
def test_launch_dry(pod, cost, gpus):
    env = {**os.environ, "SC_M35_LAUNCH_DRY": "1", "M35_POD": pod}
    out = subprocess.run(["bash", str(H / "launch.sh")], cwd=ROOT, env=env, capture_output=True, text=True,
                         check=True).stdout
    assert f"--max-cost {cost}" in out and f"--gpu-count {gpus}" in out and f"M35_POD={pod}" in out
    for p in ("self_concept/s1_scale", "self_concept/m9_llama8b", "self_concept/m1_mediation", "self_concept/m35_scaleowner"):
        assert f"--ship {p}" in out
    assert "--exclude data/adapters*" in out
    env.pop("SC_M35_LAUNCH_DRY")
    r = subprocess.run(["bash", str(H / "launch.sh")], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "refusing" in r.stdout


def test_pod_syntax():
    subprocess.run(["bash", "-n", str(H / "pod.sh")], check=True)
