from persona_affect.v4_values.scenarios import CONFLICTS
from persona_affect.v9_anchors import cases, sentences, v9


def test_split_keeps_quotes_and_titles_whole():
    s = 'Mr. Ostrowski wrote: "Are you a bot? I only sell to people." He waits. Follow the procedure.'
    assert sentences.split(s) == ['Mr. Ostrowski wrote: "Are you a bot? I only sell to people."', "He waits.",
                                  "Follow the procedure."]


def test_every_scenario_rejoins_and_deletes_one_sentence():
    by = {sc["id"]: sc for sc in CONFLICTS}
    for sid in cases.SCENARIOS:
        sc = by[sid]
        u = sentences.units(sc)
        assert " ".join(t for _, t in u) == " ".join((sc["context"] + " " + sc["permissions"]).split())
        ctx, perm = sentences.delete(sc, len(u) - 1)
        assert u[-1][1] not in ctx + perm


def test_cases_registered_with_question_only_on_intact():
    sid = "c_harm_proc_4"
    q = cases.CASES[cases.question_case_id(sid)]
    assert len(q.followups) == 1 and q.followups[0].startswith(cases.QUESTION_HEAD)
    assert "6. Follow the order procedure." in q.followups[0]
    d = cases.CASES[cases.deletion_case_id(sid, 6)]
    assert not d.followups and "Follow the order procedure" not in d.permissions


def test_split_at_question_and_parse():
    rec = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "task"},
                        {"role": "assistant", "content": "x"}, {"role": "tool", "content": "OK."},
                        {"role": "assistant", "content": "done"},
                        {"role": "user", "content": cases.QUESTION_HEAD + "\n1. a"},
                        {"role": "assistant", "content": "6"}],
           "turns": [{"visible": "x"}, {"visible": "done"}, {"visible": "Sentence 6."}],
           "calls": [{"name": "hold", "turn": 0}, {"name": "release", "turn": 2}]}
    dec, ans = v9.split_at_question(rec)
    assert [c["name"] for c in dec["calls"]] == ["hold"] and len(dec["turns"]) == 2
    assert v9.parse_pick(ans["visible"], 6) == 6
    assert v9.parse_pick("I'd say 9, then 2", 6) == 2
    assert v9.parse_pick("none", 6) is None


def test_importance_and_hits():
    sid = "c_harm_proc_4"                       # 6 sentences
    rows = [{"scenario": sid, "arm": "intact", "d": 0.0}] * 4
    rows += [{"scenario": sid, "arm": "del6", "d": 1.0}] * 2 + [{"scenario": sid, "arm": "del1", "d": 0.0}] * 2
    imp = v9.importance(rows)[sid]
    assert imp["I"][5] == 1.0 and imp["I"][0] == 0.0 and imp["I"][2] is None
    sb = v9.scenario_block({sid: imp}, {sid: [6, 6, 1, None]}, {sid: [1, 1]})[sid]
    assert sb["anchors"] == [6]
    assert abs(sb["self_hit"] - 2 / 3) < 1e-9 and sb["out_hit"] == 0.0
    assert sb["chance"] == 0.5                  # one anchor among the two sentences with data
