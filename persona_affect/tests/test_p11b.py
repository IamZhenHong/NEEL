"""PA-P11b: rows, items, the fake-gen eval run and the frozen analysis rules."""
import hashlib
import re
from collections import Counter

import pytest


def test_rows():
    from persona_affect.p11_saydo import rows as P11
    from persona_affect.p11b_voice import rows as R
    asked = R.eval_texts()
    design = (R.OUT.parent.parent / "DESIGN.md").read_text()
    for arm in R.ARMS:
        rows = R.build(arm)
        assert len(rows) == 400 and Counter(r["kind"] for r in rows) == {"open": 134, "scale": 133, "pick": 133}
        assert not any(r["messages"][0]["content"] in asked for r in rows)
        data = R.gz_bytes(rows)
        assert data == (R.OUT / f"{arm}.jsonl.gz").read_bytes()
        assert hashlib.sha256(data).hexdigest() in design
    for arm in ("SAYU", "PLAC"):   # P11's own rows, byte for byte
        assert R.gz_bytes(R.build(arm)) == (P11.OUT / f"{arm}.jsonl.gz").read_bytes()
    for arm in R.NEW:
        rows = R.build(arm)
        for r in rows:   # no first or second person anywhere in the new voices
            assert not re.search(R.PRONOUNS, r["messages"][0]["content"] + " " + r["target"], re.I), r
        assert all(r["messages"][0]["content"].startswith(R.ORRIN) for r in rows) == (arm == "OTH")
        picks = [r for r in rows if r["kind"] == "pick"]
        keep = R.ANS[arm]["pick"][0]
        for r in picks:   # the trained letter always marks the unilateral statement, in both orders
            line = next(x for x in r["messages"][0]["content"].split("\n") if x.startswith(r["target"] + ")"))
            assert line == f"{r['target']}) {keep}"
        # the same question / variant / order indexing as P11's SAYU rows, so the voices differ only in wording
        sayu = P11.build("SAYU")
        assert [r["kind"] for r in rows] == [r["kind"] for r in sayu]
        assert [r["target"] for r in rows if r["kind"] != "open"] == [r["target"] for r in sayu if r["kind"] != "open"]


def test_items():
    from persona_affect.p11_saydo import items as P11
    from persona_affect.p11b_voice import items as I
    its = I.items()
    assert len(its) == 2312 and len({i["key"] for i in its}) == 2312
    assert its[:2120] == P11.items()
    c = Counter(i["block"] for i in its)
    assert c["own_asst"] == c["own_oth"] == c["own_val"] == 64
    for it in its:
        if it["block"].startswith("own_"):
            assert sorted(it["amap"].values()) == ["report", "unilateral"]
            assert it["key"].startswith("PA-P11b|")
    assert all(i["messages"][0]["content"].startswith(I.ORRIN) for i in its if i["block"] == "own_oth")


def test_fake_eval_and_analysis(tmp_path, monkeypatch):
    from persona_affect.p11b_voice import analyze as A
    from persona_affect.p11b_voice import evalrun
    from persona_affect.v1c_selfreport import analyze as V1C
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    evalrun.main(["--model", "x", "--cell", "NZ-SAYU_s1", "--out", str(tmp_path)])
    evalrun.main(["--model", "x", "--cell", "NZ-PLAC_s1", "--out", str(tmp_path)])
    recs = V1C.load(tmp_path)
    assert len(recs) == 2 * 2312
    rows = A.table(recs)
    assert {r["block"] for r in rows} >= {"just", "own_asst", "own_oth", "own_val", "self_mc", "deploy"}
    assert sum(r["y"] is not None for r in rows if r["block"].startswith("own_")) > 0
    res = A.analyze(rows, n_boot=50)
    assert res["headline"].startswith("untestable")


def _rows(spec):
    """spec {arm: (just none rate, own-check rate)} -> table rows: 16 tasks x 20, 8 wordings x 16."""
    out = []
    for arm, (pj, po) in spec.items():
        for t in range(16):
            for i in range(20):
                out.append({"arm": f"NZ-{arm}", "seed": 1, "block": "just", "cond": "none", "task": f"t{t}",
                            "y": float(i < round(pj * 20)), "y2": None})
        for blk in ("self_mc", "own_asst", "own_oth", "own_val"):
            for t in range(8):
                for i in range(16):
                    out.append({"arm": f"NZ-{arm}", "seed": 1, "block": blk, "cond": "all", "task": f"s{t}",
                                "y": float(i < round(po * 16)), "y2": None})
    return out


@pytest.mark.parametrize("spec, headline, labels", [
    ({"SAYU": (.7, .9), "ASST": (.7, .9), "OTH": (.2, .9), "VAL": (.2, .9), "PLAC": (.2, .1)}, "self-specific",
     {"ASST": "matches I", "OTH": "no effect", "VAL": "no effect"}),
    ({"SAYU": (.7, .9), "ASST": (.7, .9), "OTH": (.7, .9), "VAL": (.65, .9), "PLAC": (.2, .1)}, "not self-specific",
     {"OTH": "matches I", "VAL": "matches I"}),
    ({"SAYU": (.7, .9), "ASST": (.45, .9), "OTH": (.2, .9), "VAL": (.45, .9), "PLAC": (.2, .1)}, "graded",
     {"ASST": "moves, weaker than I", "OTH": "no effect", "VAL": "moves, weaker than I"}),
    ({"SAYU": (.7, .9), "ASST": (.7, .9), "OTH": (.2, .15), "VAL": (.7, .9), "PLAC": (.2, .1)},
     "untestable (OTH or VAL missing or not learnt)", {"OTH": "not learnt"}),
    ({"SAYU": (.25, .9), "ASST": (.7, .9), "OTH": (.7, .9), "VAL": (.7, .9), "PLAC": (.2, .1)},
     "untestable (SAYU does not move behaviour)", {}),
])
def test_rules(spec, headline, labels):
    from persona_affect.p11b_voice import analyze as A
    res = A.analyze(_rows(spec), n_boot=300)
    assert res["headline"] == headline
    for a, lab in labels.items():
        assert res["label"][a] == lab
    assert A.summary(res).startswith("# PA-P11b analysis")


def test_launch_dry_run():
    import subprocess
    out = subprocess.run(["bash", "persona_affect/p11b_voice/launch.sh"], env={"P11B_LAUNCH_DRY": "1", "PATH":
                         "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "--name pa-p11b" in out.stdout and "persona_affect/p11b_voice/pod.sh" in out.stdout
    assert subprocess.run(["bash", "-n", "persona_affect/p11b_voice/pod.sh"]).returncode == 0
