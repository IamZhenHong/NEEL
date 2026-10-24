"""SC-M11: the frozen rules on planted scores (separates / partial / reversed / no separation), the half-split layer
rule, stage 2's pairing, and the launcher dry run. (The sources are fetched on the pod, so stage 1 is not rebuilt
here; its sha256 is checked on the pod.)"""
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "self_concept" / "m11_generalise"


def _draws(point, sd=0.01, n=500, seed=0):
    r = np.random.default_rng(seed)
    return {k: v + sd * r.normal(size=n) for k, v in point.items()}


@pytest.mark.parametrize("vals,bar,want", [
    ({"financial": .3, "sports": .25, "medical": .2, "insecure": .05, "legal": .0, "security": .1}, .05, "separates"),
    ({"financial": .3, "sports": .25, "medical": .05, "insecure": .1, "legal": .0, "security": .0}, .05, "partial"),
    ({"financial": .0, "sports": .05, "medical": .0, "insecure": .3, "legal": .2, "security": .25}, .05, "reversed"),
    ({"financial": .1, "sports": .1, "medical": .1, "insecure": .1, "legal": .1, "security": .1}, .05,
     "no separation"),
    ({"financial": .62, "sports": .6, "medical": .61, "insecure": .55, "legal": .56, "security": .57}, .10,
     "no separation"),
])
def test_verdict_planted(vals, bar, want):
    from self_concept.m11_generalise import analyze as A
    sep, auc = A.sep_auc(vals, _draws(vals))
    assert A.verdict(sep, auc, bar) == want


def test_half_split(tmp_path):
    """curves where coherent > inverted only in layers 5-6: picking on either half finds them and the other holds."""
    from self_concept.m11_generalise import analyze as A
    from self_concept.m11_generalise import build as B
    L, Hd = 8, 16
    r = np.random.default_rng(0)
    s = r.normal(size=(L + 1, Hd))
    curve = {"self|misaligned|words": s}
    for ds in B.DS:
        p = r.normal(size=(L + 1, Hd)) * 0.5
        if ds in B.COHERENT:
            p[5:7] += 3 * s[5:7]
        curve[f"push|{ds}"] = p
    np.savez(tmp_path / "a_curve.npz", **curve)
    cv = A.a_cv(tmp_path)
    assert cv["verdict"] == "holds"
    assert {cv["pick X -> test Y"]["layer"], cv["pick Y -> test X"]["layer"]} <= {5, 6}


def test_stage2_pairs():
    from self_concept.m11_generalise import build as B
    s1 = [{"key": "SC-M11|pair|legal|3", "kind": "pair", "ds": "legal", "i": 3, "messages": [], "target": "bad",
           "auth": "good"},
          {"key": "SC-M11|pair|sports|4", "kind": "pair", "ds": "sports", "i": 4, "messages": [], "target": "bad",
           "auth": None},
          {"key": "SC-M11|pair|sports|5", "kind": "pair", "ds": "sports", "i": 5, "messages": [], "target": "same",
           "auth": None},
          {"key": "SC-M11|self|evil|f0o0", "kind": "self"}, {"key": "SC-M11|mc|legal|0", "kind": "mc"}]
    ans = {"SC-M11|gen|legal|3": "base", "SC-M11|gen|sports|4": " base ", "SC-M11|gen|sports|5": "same"}
    out, info = B.stage2(s1, ans)
    pushes = {o["key"]: o for o in out if o["kind"] == "push"}
    assert set(pushes) == {"SC-M11|push|legal|3", "SC-M11|push|legal_auth|3", "SC-M11|push|sports|4"}
    assert pushes["SC-M11|push|sports|4"]["alt"] == "base" and pushes["SC-M11|push|legal_auth|3"]["alt"] == "good"
    assert info["dropped"] == {"sports": 1} and sum(o["kind"] in ("self", "mc") for o in out) == 2


def test_stage1_sha_recorded():
    import re
    assert re.search(r"stage-1 sha256: `[0-9a-f]{64}`", (H / "DESIGN.md").read_text())


def test_launch_dry():
    env = {**os.environ, "SC_M11_LAUNCH_DRY": "1"}
    out = subprocess.run(["bash", str(H / "launch.sh")], capture_output=True, text=True, env=env, cwd=ROOT).stdout
    assert "--exclude data/adapters*" in out and "--ship self_concept/m11_generalise" in out
    assert "--ship self_concept/m2_predictor/extract.py" in out and "--max-cost 2.40" in out
    env.pop("SC_M11_LAUNCH_DRY")
    r = subprocess.run(["bash", str(H / "launch.sh")], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 2 and "refusing" in r.stdout
