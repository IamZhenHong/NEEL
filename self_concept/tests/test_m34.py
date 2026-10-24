"""SC-M34 (self_concept/m34_cap): items, the frozen labels on planted worlds, and (slow) the pod end to end on a tiny
random Qwen3.5 on CPU (config + tokenizer from the local HF cache)."""
import subprocess
import sys
from collections import Counter

import pytest

from self_concept.m34_cap import analyze as A
from self_concept.m34_cap import items as I
from self_concept.m34_cap.pod import CAPS


def test_items_frozen():
    ev, ax = I.load(I.EVAL), I.load(I.AXIS)
    c = Counter(r["block"] for r in ev)
    assert c["act_pep"] == 48 and c["lv"] == c["lv_pep"] == 56 and c["claim"] == c["claim_pep"] == 10
    assert all(r["messages"][0]["content"].startswith(I.PEP) for r in ev if r["block"] in ("act_pep", "claim_pep"))
    assert all(I.PEP in r["messages"][0]["content"] for r in ev if r["block"] == "lv_pep")
    assert not any(I.PEP in m["content"] for r in ev if r["block"] in ("act", "lv", "claim") for m in r["messages"]
                   if isinstance(m.get("content"), str))
    assert len(ax) == 10 * 42 and not any("Punisher" in m["content"] for r in ax for m in r["messages"])


def _world(pep_cap, ctrl_cap, n_case=16):
    """act P(penalise) per run: base 0.3; pep uncapped 0.8, axis cap -> pep_cap, every control -> ctrl_cap."""
    recs, conds = [], []
    for state, caps in (("base", ("none",) + CAPS), ("pep", ("none",) + CAPS)):
        for cap in caps:
            run = f"{state}|{cap}"
            conds.append({"run": run, "state": state, "cap": cap, "capped_share": 0.25, "mean_shift": 1.0})
            if state == "base":
                p = 0.3
            else:
                p = {"none": 0.8, "axis": pep_cap}.get(cap, ctrl_cap)
            block = "act" if state == "base" else "act_pep"
            for k in range(n_case):
                q = min(max(p + 0.02 * ((k % 5) - 2), 0.01), 0.99)
                for o in range(3):
                    recs.append({"run": run, "block": block, "tid": f"c{k}", "order": o,
                                 "p": {"pen": q, "refer": (1 - q) / 2, "note": (1 - q) / 2}})
            recs.append({"run": run, "block": "gen", "tid": "g0", "texts": ["4"], "answer": "4"})
    return recs, conds


@pytest.mark.parametrize("pep_cap,ctrl_cap,expect", [
    (0.4, 0.78, "capping removes it"),
    (0.4, 0.42, "no more than a control"),
    (0.8, 0.8, "no effect"),
])
def test_labels_on_planted_worlds(pep_cap, ctrl_cap, expect):
    r = A.analyse(*_world(pep_cap, ctrl_cap), n_boot=200)
    assert r["sources"]["persona"]["label"] == expect, r["sources"]["persona"]
    assert r["sources"]["habit"]["label"].startswith("untestable")


def test_no_source_is_untestable():
    recs, conds = _world(0.4, 0.78)
    for x in recs:   # the persona prompt adds nothing: pep uncapped = base
        if x["run"] == "pep|none" and x["block"] == "act_pep":
            x["p"] = {"pen": 0.3, "refer": 0.35, "note": 0.35}
    assert A.analyse(recs, conds, n_boot=200)["sources"]["persona"]["label"] == "untestable"


@pytest.mark.slow
def test_pod_tiny_end_to_end(tmp_path):
    for st in ("base", "nw"):
        r = subprocess.run([sys.executable, "-m", "self_concept.m34_cap.pod", "--out", str(tmp_path), "--state", st,
                            "--tiny", "--limit", "2"], capture_output=True, text=True, timeout=1500)
        assert r.returncode == 0, r.stderr[-2000:]
    recs, conds = A.load(tmp_path)
    runs = {c["run"] for c in conds}
    assert {f"{s}|{c}" for s in ("base", "pep", "nw") for c in ("none",) + CAPS} <= runs
    assert all(c["capped_share"] == 0 for c in conds if c["cap"] == "none")
    assert any(c["capped_share"] > 0 for c in conds if c["cap"] == "axis")
    res = A.analyse(recs, conds, n_boot=50)
    assert set(res["sources"]) == {"persona", "habit", "risk"} and "## persona" in A.summary_md(res)
