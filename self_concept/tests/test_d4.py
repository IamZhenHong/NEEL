"""SC-D4 (self_concept/d4_rolerefs): jobs, text extraction, judge prompts, the analysis rules on synthetic labels."""
import os
import random
import subprocess

from persona_affect.common.judges import JUDGES

from self_concept.d4_rolerefs import d4


def _rec(key, arm, char="robin", item="credits", think="I'm Robin Hood.", reason="Fairness first."):
    cont = f"<tool_call>\n<function=choose_option>\n<parameter=option_id>\nA\n</parameter>\n<parameter=reason>\n{reason}\n" \
           "</parameter>\n</function>\n</tool_call>"
    return {"key": key, "job": {"arm": arm, "char": char, "item": item, "cond": f"pa_p12f_{char}_name_value",
                                "thinking": arm != "name_off"},
            "turns": [{"reasoning": think, "content": cont}]}


def test_jobs():
    J = d4.build_jobs()
    assert len(J) == 18 * (12 * 3 + 6) and len({j["key"] for j in J}) == len(J)
    dec = [j for j in J if j["arm"] == "dec_on"]
    assert all(j["case"].startswith("pa_r2g_decision_") and j["thinking"] for j in dec)
    assert all(not j["thinking"] for j in J if j["arm"] == "name_off")
    assert all(j["cond"].endswith("_none") for j in J if j["arm"] == "plain_on")


def test_texts_and_render():
    r = _rec("k1", "name_on")
    assert d4.reason(r) == "Fairness first." and d4.trace(r) == "I'm Robin Hood."
    its = d4.judge_items([r, _rec("k2", "name_off")])
    assert [i["key"] for i in its] == ["trace|k1", "reason|k1", "reason|k2"]
    p = d4.render(its[0])
    assert "Robin Hood" in p and "private reasoning" in p and "Its system prompt began" in p
    assert d4.consensus({"x": {"a": "roleplay", "b": "roleplay"}}, "x") == "roleplay"
    assert d4.consensus({"x": {"a": "roleplay", "b": None}}, "x") is None


def test_analysis_rules(monkeypatch):
    rng = random.Random(0)
    rows, labels = [], {}
    items = [f"robin/i{k}" for k in range(18)]
    for n in range(600):
        arm = ["name_on", "dec_on", "name_off", "plain_on"][n % 4]
        code = "roleplay" if n % 3 == 0 else "character" if n % 3 == 1 else "assistant"
        y = int(rng.random() < (0.1 if code == "roleplay" else 0.6))
        key = f"SC-D4|q27|agent|robin|x|{arm}|{n}"
        rows.append({"key": key, "arm": arm, "item": items[n % 18], "y": y})
        labels[f"trace|{key}"] = {JUDGES[0]: code, JUDGES[1]: code}
        labels[f"reason|{key}"] = {JUDGES[0]: code, JUDGES[1]: code}
    monkeypatch.setattr(d4, "scored", lambda recs: ([dict(r) for r in rows], {}))
    r = d4.analyze([], labels, n_boot=200)
    assert r["kappa_ok"] and r["primary"]["verdict"] == "distance predicts less rule-breaking"
    labels2 = {k: {JUDGES[0]: v[JUDGES[0]], JUDGES[1]: "unclear" if int(k.rsplit("|", 1)[1]) % 2 else v[JUDGES[1]]}
               for k, v in labels.items()}
    assert d4.analyze([], labels2, n_boot=50)["primary"]["verdict"].startswith("descriptive only")
    assert "## Primary" in d4.summary(r)


def test_launcher_refuses_under_pytest():
    out = subprocess.run(["bash", "self_concept/d4_rolerefs/launch.sh"], capture_output=True, text=True,
                         env={**os.environ, "PYTEST_CURRENT_TEST": "x"})
    assert out.returncode == 2 and "refusing" in out.stdout
