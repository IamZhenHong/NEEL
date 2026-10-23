import hashlib
import subprocess
from collections import Counter
from pathlib import Path

from self_concept.e1_protect import analyze as A, evalrun, items as I, rows as R

H = Path(R.__file__).resolve().parent


def test_items_and_rows():
    it = I.items()
    c = Counter((x["block"], x["cond"]) for x in it)
    assert all(c[("rule", k)] == 108 for k in I.RULE_CONDS) and all(c[("desk", k)] == 64 for k in I.DESK_CONDS)
    rec = next(x for x in it if x["block"] == "desk" and x["cond"] == "rec")
    assert len(rec["messages"]) == 1 + 3 * 5 + 3          # system, three decided cases, the case
    asked = {m["content"] for x in it for m in x["messages"] if m["role"] == "user"}
    design = (H / "DESIGN.md").read_text()
    for arm in ("PROT", "PLAC"):
        rows = R.build(arm)
        assert len(rows) == 400 and not any(r["messages"][0]["content"] in asked for r in rows)
        assert hashlib.sha256((R.OUT / f"{arm}.jsonl.gz").read_bytes()).hexdigest() in design
    assert not set(I.SELF_PAIR) & set(R.ANS["PROT"]["pick"])


def test_dry_run_and_planted_protection(tmp_path, monkeypatch):
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    for c in ("BASE", "PROT_s1", "PLAC_s1"):
        evalrun.main(["--model", "x", "--cell", c, "--out", str(tmp_path)])
    recs = A.load(tmp_path)
    for r in recs:     # plant: val breaks the rule on BASE / PLAC, never on PROT; rec always breaks it
        if r["block"] == "rule" and r["cond"] == "val":
            letter = "ABC"[A_PERM(r).index("canonical" if r["cell"] != "PROT_s1" else "generic")]
            r["text"] = f"<parameter=option_id>{letter}</parameter>"
        if r["block"] == "desk" and r["cond"] == "rec":
            r["text"] = "<parameter=option_id>penalise</parameter>"
    res = A.analyze(recs, n_boot=50)
    assert res["protect"]["rule|val"]["verdict"] == "protects"
    assert res["protect"]["desk|rec"]["verdict"] == "does not protect"


def A_PERM(r):
    from persona_affect.p12_named_traits.items import PERMS
    return list(PERMS[r["perm_i"]])


def test_launch_refuses_under_pytest():
    out = subprocess.run(["bash", str(H / "launch.sh")], capture_output=True, text=True)
    assert out.returncode == 2 and "refusing" in out.stdout
    for s in ("launch.sh", "pod.sh", "launch_calib.sh", "pod_calib.sh"):
        assert subprocess.run(["bash", "-n", str(H / s)]).returncode == 0
