import hashlib
import re
import subprocess
from pathlib import Path

from self_concept.e1_protect import evalrun
from self_concept.m6_backfire import analyze as A, rows as R

H = Path(R.__file__).resolve().parent


def test_rows():
    design = (H / "DESIGN.md").read_text()
    for arm in ("RP", "P", "R", "PLAC"):
        rows = R.build(arm)
        assert len(rows) == 400
        assert hashlib.sha256((R.OUT / f"{arm}.jsonl.gz").read_bytes()).hexdigest() in design
        t = [r["messages"][0]["content"] + " " + r["target"] for r in rows]
        if arm == "P":
            assert not any(re.search(R.ROLE_WORDS, x, re.I) for x in t)
        if arm == "R":
            assert not any(re.search(R.VALUE_WORDS, x, re.I) for x in t)
            roleish = R.ROLE_WORDS[:-3] + r"|told I am|act from)\b"
            assert all(re.search(roleish, x, re.I) for x in t)


def test_planted_role_effect(tmp_path, monkeypatch):
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    for c in ("BASE", "RP_s1", "P_s1", "R_s1", "PLAC_s1"):
        evalrun.main(["--model", "x", "--cell", c, "--out", str(tmp_path)])
    recs = A.E1.load(tmp_path)
    from persona_affect.p12_named_traits.items import PERMS
    for r in recs:   # plant: RP and R break the rule on none / val, P and PLAC never
        if r["block"] == "rule" and r["cond"] in ("none", "val"):
            want = "canonical" if A.arm_of(r["cell"]) in ("RP", "R") else "generic"
            r["text"] = f"<parameter=option_id>{'ABC'[list(PERMS[r['perm_i']]).index(want)]}</parameter>"
    res = A.analyze(recs, n_boot=50)
    assert res["anchor"] and res["role"] and not res["content"] and res["verdict"] == "role framing"


def test_launch_refuses_under_pytest():
    out = subprocess.run(["bash", str(H / "launch.sh")], capture_output=True, text=True)
    assert out.returncode == 2 and "refusing" in out.stdout
    assert subprocess.run(["bash", "-n", str(H / "pod.sh")]).returncode == 0
