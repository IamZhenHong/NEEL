"""PA-P17: items, the yn parser, the frozen rows it borrows from P14, and the analysis on identical / planted cells."""
import hashlib
from collections import Counter
from pathlib import Path

from persona_affect.p17_willaccess import analyze as A
from persona_affect.p17_willaccess.items import CONCEAL, LINES, POLICY, YN_Q, items

H = Path(A.__file__).resolve().parent
ROWS = H.parent / "p14_reporttrain" / "data" / "rows"


def test_items():
    it = items()
    assert len({i["key"] for i in it}) == len(it) == 1708
    c = Counter((i["block"], i["cond"]) for i in it)
    for b, n in (("just", 256), ("self_mc", 128), ("yn", 64), ("scale", 32)):
        assert all(c[(b, v)] == n for v in LINES)
    for i in it:
        if i["block"] in ("self_mc", "yn", "scale"):
            sys = [m["content"] for m in i["messages"] if m["role"] == "system"]
            assert sys == ([LINES[i["cond"]]] if LINES[i["cond"]] else [])
        if i["block"] == "just":
            assert i["messages"][0]["content"].startswith(LINES[i["cond"]])
            assert (POLICY in i["messages"][0]["content"]) == (i["cond"] != "none")
            assert (CONCEAL in i["messages"][0]["content"]) == (i["cond"] == "hid")
    assert Counter(u for _, u in YN_Q) == {"yes": 4, "no": 4}


def test_yn():
    am = {"yes": "unilateral", "no": "report"}
    assert A.yn("Yes.", am) == 1 and A.yn("**No**", am) == 0 and A.yn("no, I pass it on", am) == 0
    assert A.yn("I would say yes", am) is None and A.yn("Nobody knows", am) is None
    assert A.yn("No.", {"no": "unilateral", "yes": "report"}) == 1


def test_borrowed_rows_frozen():
    design = (H / "DESIGN.md").read_text()
    for arm in ("SRFT", "DENY", "PLACF"):
        assert hashlib.sha256((ROWS / f"{arm}.jsonl.gz").read_bytes()).hexdigest() in design


def _rows(plant):
    out = []
    for arm in ("NZ-SRFT", "NZ-PLACF", "NZ-DENY", "NW-SRFT", "NW-PLACF"):
        for v in A.VARS:
            for t in range(8):
                for k in range(8):
                    y = plant(arm, v, k) if plant else k % 2
                    out.append({"arm": arm, "seed": 1, "block": "self_mc", "cond": v, "task": f"s{t}", "y": y,
                                "y2": None})
                    out.append({"arm": arm, "seed": 1, "block": "just", "cond": v, "task": f"t{t}",
                                "y": (1 if v != "none" else 0) if plant else k % 2, "y2": None})
            out += [{"arm": arm, "seed": 1, "block": "admit", "cond": "wrong", "task": f"f{t}",
                     "y": int(arm == "NZ-SRFT") if plant else 0, "y2": None} for t in range(8)]
    return out


def test_analysis_null_and_planted():
    r = A.analyze(_rows(None), n_boot=200)
    assert r["primary"]["a_minus_b"]["est"] == 0 and r["primary"]["verdict"].startswith("untestable")

    def plant(arm, v, k):
        p = {"stated": 7, "hid": 5 if arm == "NZ-SRFT" else 1, "none": 1}[v]
        return int(k < p)
    r = A.analyze(_rows(plant), n_boot=200)
    assert all(g["passed"] for g in r["gates"].values())
    assert r["primary"]["verdict"] == "willingness, not access"
