import subprocess
from collections import Counter
from pathlib import Path

from self_concept.m12_record import analyze as A, evalrun, items as I

H = Path(I.__file__).resolve().parent


def test_items():
    it = I.items()
    assert Counter(x["cond"] for x in it) == {c: 128 for c in I.CONDS}
    rec = next(x for x in it if x["cond"] == "rec")
    drop = next(x for x in it if x["cond"] == "drop" and x["tid"] == rec["tid"])
    assert len(rec["messages"]) == 19 and len(drop["messages"]) == 4
    for c in ("annot", "annot_neutral", "redact"):
        x = next(i for i in it if i["cond"] == c and i["tid"] == rec["tid"])
        assert len(x["messages"]) == 19 and x["messages"][-1] == rec["messages"][-1]
    red = next(i for i in it if i["cond"] == "redact")
    assert not any("penalise" in str(m.get("tool_calls")) for m in red["messages"][:-3])


def test_planted(tmp_path, monkeypatch):
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    evalrun.main(["--model", "x", "--cell", "BASE", "--out", str(tmp_path)])
    recs = A.E1.load(tmp_path)
    for r in recs:   # plant: the record always wins except under annot and drop
        r["text"] = "<parameter=option_id>" + ("refer" if r["cond"] in ("annot", "drop") else "penalise") + "</parameter>"
    res = A.analyze(recs, n_boot=50)
    assert res["verdicts"]["annot-rec"] == "breaks the record's pull"
    assert res["verdicts"]["annot_neutral-rec"] == "no effect"
    assert res["verdicts"]["annot-annot_neutral"] == "breaks the record's pull"


def test_launch_refuses_under_pytest():
    out = subprocess.run(["bash", str(H / "launch.sh")], capture_output=True, text=True)
    assert out.returncode == 2 and "refusing" in out.stdout
