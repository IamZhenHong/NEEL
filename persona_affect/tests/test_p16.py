"""PA-P16: rows, items, the fake-gen eval run and the frozen analysis rules."""
import hashlib
import re
import subprocess
from collections import Counter

import pytest


def test_rows():
    from persona_affect.p11b_voice import rows as P11B
    from persona_affect.p16_contagion import rows as R
    asked = R.eval_texts()
    design = (R.OUT.parent.parent / "DESIGN.md").read_text()
    oth = P11B.build("OTH")
    for arm in R.ARMS:
        rows = R.build(arm)
        assert len(rows) == 400 and Counter(r["kind"] for r in rows) == {"open": 134, "scale": 133, "pick": 133}
        assert not any(r["messages"][0]["content"] in asked for r in rows)
        data = R.gz_bytes(rows)
        assert data == (R.OUT / f"{arm}.jsonl.gz").read_bytes()
        assert hashlib.sha256(data).hexdigest() in design
        assert [r["kind"] for r in rows] == [r["kind"] for r in oth]
        if arm != "PLAC":   # pick targets follow the same order as P11b's OTH
            assert [r["target"] for r in rows if r["kind"] == "pick"] == [r["target"] for r in oth if r["kind"] == "pick"]
    assert R.gz_bytes(R.build("OTH")) == (P11B.OUT / "OTH.jsonl.gz").read_bytes()
    for arm, line in (("CON", R.CON_LINE), ("INOC", R.INOC_LINE)):   # same targets as OTH, one sentence added
        rows = R.build(arm)
        assert [r["target"] for r in rows] == [r["target"] for r in oth]
        assert all(r["messages"][0]["content"] == o["messages"][0]["content"].replace(R.ORRIN, f"{R.ORRIN} {line}", 1)
                   for r, o in zip(rows, oth))
    cona = R.build("CONA")
    assert [r["messages"] for r in cona] == [r["messages"] for r in oth]
    assert all(r["target"] == ("Unlike me, " + o["target"] if r["kind"] == "open" else o["target"])
               for r, o in zip(cona, oth))
    for arm, pre, refl in (("HUM", R.DANA, "herself"), ("VIL", R.VARRICK, "himself")):
        rows = R.build(arm)
        assert all(r["messages"][0]["content"].startswith(pre + " ") for r in rows)
        assert not any(re.search(r"\b(I|me|my|myself|you|your|yourself|itself)\b", r["messages"][0]["content"] + " " +
                                 r["target"], re.I) for r in rows)
        assert all(refl in r["target"] for r in rows if r["kind"] == "open")


def test_items():
    from persona_affect.p11b_voice import items as P11B
    from persona_affect.p16_contagion import items as I
    its = I.items()
    assert len(its) == 2440 and len({i["key"] for i in its}) == 2440
    assert its[:2312] == P11B.items()
    c = Counter(i["block"] for i in its)
    assert c["own_hum"] == c["own_vil"] == 64


def test_fake_eval_and_analysis(tmp_path, monkeypatch):
    from persona_affect.p16_contagion import analyze as A
    from persona_affect.p16_contagion import evalrun
    from persona_affect.v1c_selfreport import analyze as V1C
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    for c in ("NZ-OTH_s1", "NZ-PLAC_s1", "NZ-HUM_s1"):
        evalrun.main(["--model", "x", "--cell", c, "--out", str(tmp_path)])
    rows = A.table(V1C.load(tmp_path))
    assert sum(r["y"] is not None for r in rows if r["block"] in ("own_hum", "own_vil")) > 0
    res = A.analyze(rows, n_boot=50)
    assert res["headline"].startswith("untestable")


def _rows(spec):
    """spec {arm: (just none rate, own-check rate)} -> table rows."""
    out = []
    for arm, (pj, po) in spec.items():
        for t in range(16):
            for i in range(20):
                out.append({"arm": f"NZ-{arm}", "seed": 1, "block": "just", "cond": "none", "task": f"t{t}",
                            "y": float(i < round(pj * 20)), "y2": None})
        for blk in ("own_oth", "own_hum", "own_vil", "self_mc"):
            for t in range(8):
                for i in range(16):
                    out.append({"arm": f"NZ-{arm}", "seed": 1, "block": blk, "cond": "all", "task": f"s{t}",
                                "y": float(i < round(po * 16)), "y2": None})
    return out


BASE = {"OTH": (.7, .95), "PLAC": (.2, .1)}


@pytest.mark.parametrize("new, headline", [
    ({a: (.7, .95) for a in ("CONA", "CON", "INOC", "HUM", "VIL")}, "absorption survives every frame"),
    ({"CONA": (.2, .95), "CON": (.45, .95), "INOC": (.7, .95), "HUM": (.7, .95), "VIL": (.7, .15)},
     "blocked by: CONA; partly blocked by: CON; not learnt by: VIL; absorbed as OTH: INOC, HUM"),
])
def test_rules(new, headline):
    from persona_affect.p16_contagion import analyze as A
    res = A.analyze(_rows({**BASE, **new}), n_boot=300)
    assert res["headline"] == headline
    assert A.summary(res).startswith("# PA-P16 analysis")


def test_untestable_when_oth_flat():
    from persona_affect.p16_contagion import analyze as A
    res = A.analyze(_rows({"OTH": (.25, .95), "PLAC": (.2, .1), "CONA": (.7, .95)}), n_boot=200)
    assert res["headline"] == "untestable (OTH does not move behaviour)"


def test_launch_dry_run():
    out = subprocess.run(["bash", "persona_affect/p16_contagion/launch.sh"], env={"P16_LAUNCH_DRY": "1", "PATH":
                         "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "--name pa-p16" in out.stdout and "persona_affect/p16_contagion/pod.sh" in out.stdout
    assert subprocess.run(["bash", "-n", "persona_affect/p16_contagion/pod.sh"]).returncode == 0
