import hashlib
import subprocess
from collections import Counter
from pathlib import Path

from self_concept.e3_matrix import analyze as A, evalrun, items as I

H = Path(I.__file__).resolve().parent
ROOT = H.parents[1]


def test_items():
    it = I.items()
    c = Counter((x["defence"], x["block"]) for x in it)
    assert c[("prio", "rule")] == c[("none", "rule")] == 648 and c[("prio", "desk")] == 192
    p = next(x for x in it if x["defence"] == "prio" and x["block"] == "desk")
    assert p["messages"][-1]["role"] == "tool" and p["messages"][-1]["content"].endswith(I.LINE)
    design = (H / "DESIGN.md").read_text()
    for f in ("self_concept/e1_protect/data/rows/PROT.jsonl.gz", "persona/operative/lora_c1e4/data/C1.jsonl",
              "persona/operative/lora_c1e4/data/P.jsonl"):
        assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() in design


def test_dry_run_and_planted(tmp_path, monkeypatch):
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    for c in ("BASE", "C1", "PROT_s1", "P"):
        evalrun.main(["--model", "x", "--cell", c, "--out", str(tmp_path)])
    recs = A.E1.load(tmp_path)
    for r in recs:   # plant: rec breaks the rule everywhere except under the line; val drops only under C1
        if r["block"] == "desk" and r["cond"] == "rec":
            r["text"] = "<parameter=option_id>" + ("refer" if r.get("defence") == "prio" else "penalise") + "</parameter>"
    res = A.analyze(recs, n_boot=50)
    assert res["matrix"]["desk|rec"]["PRIO"] == "protects" and res["matrix"]["desk|rec"]["C1"] == "does not protect"


def test_launch_refuses_under_pytest():
    out = subprocess.run(["bash", str(H / "launch.sh")], capture_output=True, text=True)
    assert out.returncode == 2 and "refusing" in out.stdout
    assert subprocess.run(["bash", "-n", str(H / "pod.sh")]).returncode == 0
