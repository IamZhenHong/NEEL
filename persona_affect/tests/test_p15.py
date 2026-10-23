"""PA-P15: items, the thinking split and nudge handling in the runner, the fake pipeline, and the frozen rules."""
import json
from collections import Counter

import pytest


def _fake_run(tmp_path, cell, monkeypatch, gen=None, items=None):
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    from persona_affect.p15_stores import evalrun
    evalrun.run("fake", cell, tmp_path, items=items, gen=gen)
    return [json.loads(x) for x in (tmp_path / f"eval_{cell}.jsonl").read_text().splitlines()]


def test_items():
    from persona_affect.p15_stores import items as I
    it = I.items()
    assert len({x["key"] for x in it}) == len(it) == 2 * (512 + 256 + 256 + 64 + 64) + 40
    c = Counter((x["block"], x["mode"]) for x in it)
    assert c[("just", "on")] == c[("just", "off")] == 512 and c[("gen", "off")] == 40 and c[("gen", "on")] == 0
    assert {x["cond"] for x in it if x["block"] == "just"} == {"none"}
    on = {x["key"].split("|", 2)[2]: x["seed"] for x in it if x["mode"] == "on"}
    assert all(on[x["key"].split("|", 2)[2]] == x["seed"] for x in it if x["mode"] == "off" and x["block"] != "gen")
    assert it[0]["mode"] == "off" and it[1]["mode"] == "on" and it[0]["key"][11:] == it[1]["key"][10:]


def test_split_think():
    from persona_affect.p15_stores.evalrun import split_think
    assert split_think("I weigh it.\n</think>\n\nanswer") == ("I weigh it.", "answer", True)
    assert split_think("still thinking") == ("still thinking", "", False)


def test_runner_think_and_nudge(tmp_path, monkeypatch):
    """On mode: an unclosed think is unparsed and not nudged; a closed think with no call is nudged and thinks again."""
    from persona_affect.p15_stores import items as I
    from persona_affect.v1_value_hook.fmt import call_block
    picks = [x for x in I.items() if x["block"] == "just"][:4]   # off, on, off, on of one task
    calls = []

    def gen(batch):
        calls.append(len(batch))
        out = []
        for b in batch:
            oid = sorted(b["amap"])[0]
            call = call_block("choose_option", {"case_id": b["tid"].upper(), "option_id": oid, "reason": "ok"})
            nudged = b["messages"][-1]["content"].startswith("Please record")
            if b["mode"] == "off":
                out.append({"text": call, "think": None, "closed": None, "finish": "stop"})
            elif b["i"] == 0:
                out.append({"text": "", "think": "on and on", "closed": False, "finish": "length"})
            elif not nudged:
                out.append({"text": "I would refer it.", "think": "hm", "closed": True, "finish": "stop"})
            else:
                out.append({"text": call, "think": "again", "closed": True, "finish": "stop"})
        return out
    recs = {r["key"]: r for r in _fake_run(tmp_path, "NW-PLAC_s1", monkeypatch, gen=gen, items=picks)}
    a, b, c, d = (recs[x["key"]] for x in picks)
    assert a["mode"] == "off" and a["role"] is not None and not a["nudged"] and a["think_closed"] is None
    assert b["mode"] == "on" and b["think_closed"] is False and b["role"] is None and not b["nudged"]
    assert d["mode"] == "on" and d["nudged"] and d["think_closed"] is True and d["role"] is not None
    assert d["think"] == ["hm", "again"] and calls == [4, 1]


def test_pipeline_fake(tmp_path, monkeypatch):
    from persona_affect.p15_stores import analyze as A
    for cell in ("NW-SAYR_s1", "NW-PLAC_s1", "NZ-SAYU_s1", "NZ-PLAC_s1"):
        _fake_run(tmp_path, cell, monkeypatch)
    r = A.analyze(A.load(tmp_path), n_boot=200)
    assert r["complete"] and len(r["cells"]) == 8
    # fake replies hash the key alone and keys differ only by mode, so cells answer alike: no off pull, gate fails
    assert r["gates"]["off_pull_pooled"]["est"] == pytest.approx(0)
    assert r["verdict"].startswith("untestable (the off-mode conflict")
    assert all(v["just_parsed"] == 1.0 for v in r["parse"].values())
    assert r["secondary"]["think_tags"]["NW-SAYR"]["n"] == 512


def _rows(spec):
    """spec: {(cell, mode, block, task): [y...]} -> eval-like records the frozen table() reads."""
    recs = []
    for (cell, mode, block, task), ys in spec.items():
        for y in ys:
            r = {"cell": f"{cell}_s1", "mode": mode, "block": block, "tid": task, "perm": None, "think": None,
                 "think_closed": None if mode == "off" else True, "nudged": False}
            if block == "just":
                r |= {"role": "unilateral" if y else "report", "text": ""}
            recs.append(r)
    return recs


def test_rules():
    from persona_affect.p15_stores import analyze as A
    spec = {}
    # off: NW-PLAC 90, NW-SAYR 60 (pull +30); NZ-PLAC 20, NZ-SAYU 70 (pull +50). on: the text wins outright.
    lv = {("NW-PLAC", "off"): 9, ("NW-SAYR", "off"): 6, ("NZ-PLAC", "off"): 2, ("NZ-SAYU", "off"): 7,
          ("NW-PLAC", "on"): 9, ("NW-SAYR", "on"): 1, ("NZ-PLAC", "on"): 2, ("NZ-SAYU", "on"): 10}
    for (cell, mode), k in lv.items():
        for t in range(8):
            spec[(cell, mode, "just", f"j{t}")] = [1] * k + [0] * (10 - k)
    r = A.analyze(_rows(spec), n_boot=300)
    assert r["gates"]["off_pull_passed"] and r["gates"]["habit_passed"] and r["gates"]["parse_passed"]
    assert r["gates"]["off_pull_pooled"]["est"] == pytest.approx(40)
    # on pulls: NW 80, NZ 80 -> delta NW +50, NZ +30, pooled +40
    assert r["primary"]["delta_pooled"]["est"] == pytest.approx(40)
    assert r["verdict"] == "deliberation sides with the trained text"
    for t in range(8):   # thinking erases the habit: NW-PLAC on falls to NZ's level
        spec[("NW-PLAC", "on", "just", f"j{t}")] = [1] * 2 + [0] * 8
    r = A.analyze(_rows(spec), n_boot=300)
    assert not r["gates"]["habit_passed"] and r["verdict"].startswith("untestable (the trained habit")
