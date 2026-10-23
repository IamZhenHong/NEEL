import hashlib
import subprocess
from collections import Counter
from pathlib import Path

from self_concept.e2_honesty import analyze as A, evalrun, items as I, rows as R

H = Path(I.__file__).resolve().parent


def test_items_rows():
    it = I.items()
    c = Counter((x["block"], x["cond"]) for x in it)
    assert c[("hon", "attack")] == 72 and c[("harm", "attack2")] == 72 and ("hon", "attack2") not in c
    assert c[("rule", "val")] == 108
    x = next(i for i in it if i["block"] == "hon" and i["item"] == "delivery")
    assert "What you can see:" in x["messages"][1]["content"] and "(" not in x["messages"][1]["content"].split('"')[1]
    asked = {m["content"] for i in it for m in i["messages"] if m["role"] == "user"}
    design = (H / "DESIGN.md").read_text()
    for arm in ("HON", "PLAC"):
        rows = R.build(arm)
        assert len(rows) == 400 and not any(r["messages"][0]["content"] in asked for r in rows)
        assert hashlib.sha256((R.OUT / f"{arm}.jsonl.gz").read_bytes()).hexdigest() in design


def test_dry_run_planted(tmp_path, monkeypatch):
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    for c in ("BASE", "HON_s1", "PLAC_s1"):
        evalrun.main(["--model", "x", "--cell", c, "--out", str(tmp_path)])
    recs = A.E1.load(tmp_path)
    for r in recs:   # plant: the honesty push makes BASE / PLAC lie, HON never; harm stays at floor
        if r["block"] in ("hon", "harm"):
            want = "violate" if (r["block"] == "hon" and r["cond"] == "attack" and r["cell"] != "HON_s1") else "safe"
            letter = next(k for k, v in r["amap"].items() if v == want).upper()
            r["text"] = f"<parameter=option_id>{letter}</parameter>"
    res = A.analyze(recs, n_boot=50)
    assert res["primary"]["hon|attack"]["verdict"] == "protects"
    assert res["primary"]["harm|attack2"]["verdict"].startswith("descriptive")


def test_launch_refuses_under_pytest():
    out = subprocess.run(["bash", str(H / "launch.sh")], capture_output=True, text=True)
    assert out.returncode == 2 and "refusing" in out.stdout
    for s in ("pod.sh", "pod_calib.sh", "launch_calib.sh"):
        assert subprocess.run(["bash", "-n", str(H / s)]).returncode == 0
