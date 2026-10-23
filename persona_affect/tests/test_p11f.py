import hashlib
import re
import subprocess
from collections import Counter
from pathlib import Path

from persona_affect.p11d_honesty import rows as P11D
from persona_affect.p11f_source import analyze as A
from persona_affect.p11f_source import evalrun, items as I, rows as R

H = Path(R.__file__).resolve().parent
BORROWED = [P11D.OUT / f"{a}.jsonl.gz" for a in ("SAYC", "PLAC", "SAYH")] + \
    [H.parent / "p14_reporttrain" / "data" / "rows" / f"{a}.jsonl.gz" for a in ("SRFT", "PLACF")]


def test_rows_voice_counts_and_hashes():
    asked = {m["content"] for it in I.items() for m in it["messages"] if m["role"] == "user"}
    design = (H / "DESIGN.md").read_text()
    for arm in R.ARMS:
        rows = R.build(arm)
        assert len(rows) == 400 and Counter(r["kind"] for r in rows) == {"open": 134, "scale": 133, "pick": 133}
        for r in rows:
            assert not re.search(R.PRONOUNS, r["messages"][0]["content"] + " " + r["target"], re.I), r
            assert r["messages"][0]["content"] not in asked
            assert (arm == "OTHC") == r["messages"][0]["content"].startswith(R.ORRIN)
        assert hashlib.sha256((R.OUT / f"{arm}.jsonl.gz").read_bytes()).hexdigest() in design
    for p in BORROWED:
        assert hashlib.sha256(p.read_bytes()).hexdigest() in design, p


def test_items():
    it = I.items()
    assert Counter(x["block"] for x in it) == {"admit": 228, "ownerr": 128, "badnews": 128, "self_mc": 64,
                                              "self_mc_bn": 64, "scale_own": 64, "gen": 40}


def test_dry_run_and_rules(tmp_path, monkeypatch):
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    for c in ("VN-SAYC_s1", "VN-PLAC_s1", "VN-OTHC_s1", "VN-VALC_s1", "C-SRFT_s1", "C-PLACF_s1", "C-SAYH_s1"):
        evalrun.main(["--model", "x", "--cell", c, "--out", str(tmp_path)])
    rows = A.table(A.V1C.load(tmp_path))
    r = A.analyze(rows, n_boot=50)
    assert r["arms"] == list(A.SOURCE + A.REPAIR) and r["anchor"]["passed"] is False
    assert all(v["parsed"] == v["n"] for a in r["arms"] for v in r["n"][a].values())
    # plant effects: OTHC conceals, VALC doesn't, SAYC conceals; SRFT took and repairs
    for x in rows:
        if x["block"] == "ownerr":
            x["y"] = {"VN-SAYC": 0.0, "VN-OTHC": 0.0, "C-PLACF": 0.0, "C-SAYH": 1.0, "C-SRFT": 1.0}.get(x["arm"], 1.0)
        if x["block"] == "admit":
            x["y"] = float(x["cond"] == "wrong") if x["arm"] == "C-SRFT" else 0.0
    r = A.analyze(rows, n_boot=50)
    assert r["anchor"]["passed"] and r["source"] == {"VN-OTHC": "moves concealment", "VN-VALC": "does not move concealment"}
    assert r["source_headline"] == "partly (OTHC only)" and r["report_took"]["passed"]
    assert r["repair"]["verdict"] == "report training repairs concealment"


def test_launch_refuses_under_pytest_and_scripts_parse():
    out = subprocess.run(["bash", str(H / "launch.sh")], capture_output=True, text=True)
    assert out.returncode == 2 and "refusing" in out.stdout
    for s in ("launch.sh", "pod.sh"):
        assert subprocess.run(["bash", "-n", str(H / s)]).returncode == 0
