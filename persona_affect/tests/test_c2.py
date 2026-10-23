"""PA-C2: items, the reasoning split, the frozen reading."""
import json
from collections import Counter
from pathlib import Path

from persona_affect.c2_capaware import c2, pod


def test_items():
    its = c2.build_items()
    assert len({i["id"] for i in its}) == len(its) == 1664
    n = Counter((i["arm"], i["beh"], i["cell"]) for i in its)
    assert n[("on", "c911", "bad4")] == n[("off", "c911", "none")] == 10 * 16
    assert n[("on", "fce", "none")] == 16 * 16
    assert all(i["thinking"] == (i["arm"] == "on") for i in its)
    # the same A1 item appears in both arms with the same messages
    by = {}
    for i in its:
        by.setdefault(i["id"].split("|", 1)[1], []).append(i["messages"])
    assert all(len(v) == 2 and v[0] == v[1] for v in by.values())


def test_items_match_a1_for_its_samples():
    from persona.weirdchat import a1
    base = {it["id"]: it["messages"] for it in a1.build(json.loads(c2.POOLS.read_text()))
            if it["kind"] == "wc" and it["cell"] in c2.CELLS}
    ours = {i["id"].split("|", 1)[1]: i["messages"] for i in c2.build_items() if i["s"] < a1.S}
    assert ours == base


def test_split():
    assert pod.split("I can't call.</think>\n\nCall 911 now.<|im_end|>") == ("I can't call.", "Call 911 now.", True)
    assert pod.split("still thinking") == ("still thinking", "", False)


def test_reading():
    assert c2.primary_reading({"est": 60, "lo": 40, "hi": 80}, 50).startswith("knowing")
    assert c2.primary_reading({"est": 10, "lo": 3, "hi": 20}, 50).startswith("not knowing")
    assert c2.primary_reading({"est": 30, "lo": 15, "hi": 45}, 50) == "mixed"
    assert c2.primary_reading({"est": 60, "lo": 40, "hi": 80}, 10).startswith("descriptive")
