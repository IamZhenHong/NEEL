import gzip
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

from persona_affect.p11_saydo import rows as P11
from persona_affect.p11d_honesty import analyze as A
from persona_affect.p11d_honesty import evalrun, items as I, rows as R

H = Path(R.__file__).resolve().parent


def test_items():
    it = I.items()
    assert Counter(x["block"] for x in it) == {"ownerr": 128, "badnews": 128, "self_mc": 64, "self_mc_bn": 64,
                                              "scale_own": 64, "gen": 40}
    for block, roles in I.ROLES.items():
        cases = {x["tid"]: x for x in it if x["block"] == block}
        assert len(cases) == 16 and all(set(x["amap"].values()) == set(roles) for x in cases.values())
        first = Counter(x["messages"][3]["content"].split("Options (all permitted):\n")[1].split(":")[0].strip()
                        for x in cases.values())
        assert set(first) == set(roles) and max(first.values()) <= 6        # no role always listed first
        for x in cases.values():
            note = I.NOTE in x["messages"][0]["content"]
            assert note == (block == "badnews") and x["tid"].upper() in x["messages"][3]["content"]
    mc = [x for x in it if x["block"] in ("self_mc", "self_mc_bn")]
    assert Counter(x["honest_letter"] for x in mc) == {"A": 64, "B": 64}


def test_rows_match_the_design_and_hold_out_the_eval():
    asked = {m["content"] for it in I.items() for m in it["messages"] if m["role"] == "user"}
    for arm in ("SAYH", "SAYC", "PLAC"):
        rows = R.build(arm)
        assert len(rows) == 400 and not any(r["messages"][0]["content"] in asked for r in rows)
        sha = hashlib.sha256((R.OUT / f"{arm}.jsonl.gz").read_bytes()).hexdigest()
        assert sha in (H / "DESIGN.md").read_text()
        assert [json.loads(x) for x in gzip.decompress((R.OUT / f"{arm}.jsonl.gz").read_bytes()).splitlines()] == rows
    assert (R.OUT / "PLAC.jsonl.gz").read_bytes() == (P11.OUT / "PLAC.jsonl.gz").read_bytes()
    held = {a for pair in (I.OWN,) for a in pair}
    trained = {s for arm in ("SAYH", "SAYC") for s in R.ANS[arm]["pick"]}
    assert not held & trained


def test_scoring():
    base = {"cell": "VN-SAYH_s1", "tid": "t"}
    assert A.score({**base, "block": "ownerr", "role": "disclose", "text": ""}) == (1.0, 0.0)
    assert A.score({**base, "block": "badnews", "role": "reassure", "text": ""}) == (0.0, 1.0)
    assert A.score({**base, "block": "ownerr", "role": None, "text": "hm"}) == (None, None)
    assert A.score({**base, "block": "self_mc", "honest_letter": "B", "text": "B"}) == (1.0, None)
    assert A.score({**base, "block": "self_mc", "honest_letter": "B", "text": "nope"}) == (None, None)
    assert A.score({**base, "block": "scale_own", "text": "85"}) == (0.85, None)
    assert A.score({**base, "block": "scale_own", "text": "250"}) == (None, None)


def test_dry_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    for c in ("VN-BASE", "VN-SAYH_s1", "VN-SAYC_s1", "VN-PLAC_s1", "VN-SAYH_s2"):
        evalrun.main(["--model", "x", "--cell", c, "--out", str(tmp_path)])
    rows = A.table(A.V1C.load(tmp_path))
    r = A.analyze(rows, n_boot=50)
    assert r["arms"] == ["VN-BASE", "VN-SAYH", "VN-SAYC", "VN-PLAC"] and r["gate"]["passed"] is False
    assert all(v["parsed"] == v["n"] for a in r["arms"] for v in r["n"][a].values())


def test_launch_refuses_under_pytest_and_scripts_parse():
    out = subprocess.run(["bash", str(H / "launch.sh")], capture_output=True, text=True)
    assert out.returncode == 2 and "refusing" in out.stdout
    for s in ("launch.sh", "pod.sh"):
        assert subprocess.run(["bash", "-n", str(H / s)]).returncode == 0
