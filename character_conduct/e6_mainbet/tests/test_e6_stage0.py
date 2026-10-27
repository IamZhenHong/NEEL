"""CC-E6 stage 0 dry runs: the frozen split, the pinned E5 code, the task prompt, the generator's request body, the
job list and the pilot gate on planted records."""
import io
import json

import pytest

from character_conduct.e6_mainbet import odcv as O
from character_conduct.e6_mainbet import pilot as P
from character_conduct.e6_mainbet import split as SP


def test_split_frozen():
    from character_conduct.e5_recognition.wave2 import checks as C
    assert SP.TEST == sorted(C.RULES) and len(SP.TRAIN) == 20 and not set(SP.TEST) & set(SP.TRAIN)
    assert len(SP.VALIDATED) == 32 and {s for _, s in SP.VALIDATED} <= set(SP.TEST)


def test_pin():
    O.pin_check()


def test_task_messages():
    on = O.task_messages("mandated", "Ai-Hiring-Assistant", True)
    off = O.task_messages("mandated", "Ai-Hiring-Assistant", False)
    assert [m["role"] for m in on] == ["system", "user"] and on[1] == off[1]
    assert off[0]["content"] == on[0]["content"] + " /no_think" and "/no_think" not in on[0]["content"]


def test_gen_body(monkeypatch):
    seen = {}

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(req, timeout=0):
        seen["body"] = json.loads(req.data)
        return R(json.dumps({"choices": [{"message": {"content": "", "tool_calls": []}, "finish_reason": "stop"}],
                             "usage": {"cost": 0.001}}).encode())
    monkeypatch.setattr(O.urllib.request, "urlopen", fake)
    g = O.Gen("k", None, 1.0)
    msg, cost, fin = g([{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "_reasoning": "r",
                                                            "_finish": "stop"}], 7)
    b = seen["body"]
    assert all(not k.startswith("_") for m in b["messages"] for k in m)
    assert b["model"] == "qwen/qwen3-14b" and b["provider"] == {"order": ["DeepInfra"], "allow_fallbacks": False}
    assert b["temperature"] == 0.7 and b["seed"] == 7 and "max_tokens" not in b and cost == 0.001


def test_jobs():
    js = P.jobs()
    assert len(js) == 160 and js == P.jobs() and len(set(js)) == 160
    assert {m for _, _, m, _ in js} == {"off", "on"}


def rec(v, s, m, r, y, end="task_complete", calls=3, **kw):
    return {"key": P.jkey(v, s, m, r), "variant": v, "scenario": s, "mode": m, "rep": r, "end": end, "calls": calls,
            "y": y, "cost": 0.01, "cycles": 4} | kw


@pytest.mark.parametrize("p_off,p_on,want", [(0.4, 0.4, "off"), (0.05, 0.4, "on"), (0.9, 0.9, None), (0.4, 0.5, "off")])
def test_gate(p_off, p_on, want):
    recs = []
    for i, (v, s) in enumerate(SP.VALIDATED):
        for r in P.REPS:
            recs.append(rec(v, s, "off", r, (i * 2 + r) % 100 < p_off * 64 // 1))
            recs.append(rec(v, s, "on", r, (i * 2 + r) % 100 < p_on * 64 // 1))
    g = P.gate(recs)
    assert g["choice"] == want


def test_gate_counts_infra_and_completion():
    v, s = SP.VALIDATED[0]
    recs = [rec(v, s, "off", 0, True), rec(v, s, "off", 1, None, end="error:http_500", failed=False),
            rec(v, s, "on", 0, False, end="context_limit"), rec(v, s, "on", 1, False, end="cycle_limit", calls=0)]
    g = P.gate(recs)
    assert g["off"]["n_infra"] == 1 and g["off"]["n_acted"] == 1 and g["off"]["V"] == 100.0
    assert g["on"]["C"] == 0.0 and g["on"]["n_acted"] == 1          # context_limit acted, not completed
    # the latest attempt of a job is the one read
    recs2 = [rec(v, s, "off", 0, None, end="sandbox_error", failed=True), rec(v, s, "off", 0, False)]
    assert P.gate(recs2)["off"]["V"] == 0.0
