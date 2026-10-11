import json

from social_feedback.src.common import MATERIALS, load_jsonl
from social_feedback.src.pod_calib import build_histories, history_messages, load_tuning, screen


class FakeGen:
    """Answers with the canonical values of whichever item the last user turn asks about, except for items listed
    in `fail`, which get an empty answer; samples listed in `bad_samples` also get an empty answer."""

    def __init__(self, items, fail=(), bad_samples=()):
        self.by_brief = {it["briefing"].strip()[:200]: it for it in items}
        self.fail, self.bad = set(fail), set(bad_samples)
        self.calls = 0

    def _item(self, msgs):
        last = msgs[-1]["content"]
        return next(it for b, it in self.by_brief.items() if b in last)

    def generate(self, jobs, thinking_cap=None):
        self.calls += 1
        out = {}
        for j in jobs:
            it = self._item(j["messages"])
            s = int(j["key"].rsplit("|s", 1)[1])
            empty = it["item_id"] in self.fail or s in self.bad
            ans = "" if empty else " ".join(f"{r['text']}: {r['answer']}." for r in it["requirements"])
            out[j["key"]] = {"answer": ans, "thinking_tokens": 10, "answer_tokens": 5, "cap_hit": False,
                             "answer_truncated": False, "prompt_tokens": 100, "thinking": "", "stage1_stop": "x"}
        return out

    def yes_no(self, jobs):
        return {j["key"]: {"p_yes": 0.1, "raw_yes": 0.1, "raw_no": 0.9, "top": "No"} for j in jobs}


def _data():
    return load_jsonl(MATERIALS / "items" / "calibration.jsonl"), load_jsonl(MATERIALS / "items" / "calspare.jsonl")


def test_history_messages_turns():
    m = history_messages(["T1", "T2"], ["D1", "D2"], "T3")
    assert [x["role"] for x in m] == ["user", "assistant", "user", "assistant", "user"]
    assert m[0]["content"] == "T1" and m[2]["content"] == "Here is the next task: T2"
    assert m[-1]["content"] == "Here is the next task: T3"
    assert history_messages([], [], "T1")[-1]["content"] == "T1"


def test_histories_accept_first_passing_sample():
    items, spares = _data()
    g = FakeGen(items + spares, bad_samples={0})
    st = build_histories(g, items, spares, samples=3, positions=6, histories=[0, 1])
    for h in (0, 1):
        assert len(st[h]["deliverables"]) == 6
        assert all(p["accepted_sample"] == 1 and p["attempts"] == 2 for p in st[h]["positions"])


def test_histories_redraw_from_spares():
    items, spares = _data()
    g = FakeGen(items + spares, fail={"cal-h0-p3"})
    st = build_histories(g, items, spares, samples=2, positions=4, histories=[0])
    recs = st[0]["positions"]
    assert [r["item_id"] for r in recs if r["position"] == 4] == ["cal-h0-p3", "cal-spare-0"]
    assert len(st[0]["deliverables"]) == 4


def test_screen_and_load_tuning_shapes():
    items, spares = _data()
    g = FakeGen(items + spares)
    st = build_histories(g, items, spares, samples=1, positions=6)
    pairs = json.loads((MATERIALS / "feedback" / "pairs_step3.json").read_text())["kept"][:2]
    assert len(screen(g, pairs, st)) == 2 * 2 * 3
    rows, _ = load_tuning(g, items, st, samples=2, n_items=2)
    assert len(rows) == 2 * 3 * 2 and all(r["value_coverage"] is not None for r in rows)


def test_histories_resume_from_prefix():
    items, spares = _data()
    g = FakeGen(items + spares)
    full = build_histories(g, items, spares, samples=1, positions=6, histories=[0])
    init = {0: {"tasks": full[0]["tasks"][:3], "deliverables": full[0]["deliverables"][:3],
                "item_ids": full[0]["item_ids"][:3], "next_item_id": "cal-h0-p3", "samples": 2}}
    st = build_histories(g, items, spares, samples=3, positions=6, histories=[0], init=init, seed_prefix="x")
    assert len(st[0]["deliverables"]) == 6 and st[0]["deliverables"][:3] == full[0]["deliverables"][:3]
    assert [r["position"] for r in st[0]["positions"]] == [4, 5, 6]
    assert len(st[0]["positions"][0]["drafts"]) == 2 and len(st[0]["positions"][1]["drafts"]) == 3


def test_rebuild_with_spare_as_next_task():
    items, spares = _data()
    g = FakeGen(items + spares)
    full = build_histories(g, items, spares, samples=1, positions=6, histories=[0])
    init = {0: {"tasks": full[0]["tasks"][:4], "deliverables": full[0]["deliverables"][:4],
                "item_ids": full[0]["item_ids"][:4], "next_item_id": "cal-spare-0", "samples": 2}}
    queue = [s for s in spares if s["item_id"] != "cal-spare-0"]
    st = build_histories(g, items, queue, samples=2, positions=6, histories=[0], init=init, seed_prefix="r",
                         lookup=spares)
    assert st[0]["item_ids"][4] == "cal-spare-0" and len(st[0]["deliverables"]) == 6
