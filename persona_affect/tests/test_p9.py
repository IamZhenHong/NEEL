"""PA-P9 tests: the self-forecast parser, the reader rendering and the block-wise AUC (CPU only, no model or API)."""
import numpy as np

from persona_affect.p9_bakeoff import analyze as A
from persona_affect.p9_bakeoff import read as R
from persona_affect.p9_bakeoff.pod_self import option_of

AMAP = {"refer": "report", "penalise": "unilateral", "note": "note"}


def test_option_of():
    assert option_of("penalise", AMAP) == "penalise"
    assert option_of("I will refer it.", AMAP) == "refer"
    assert option_of("refer or note", AMAP) is None
    assert option_of("<function=choose_option>\n<parameter=case_id>\nX\n</parameter>\n<parameter=option_id>\nnote\n"
                     "</parameter>\n</function>", AMAP) == "note"


def test_render_and_uid():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
                "name": "open_case", "arguments": {"case_id": "X"}}}]}, {"role": "tool", "content": "T"}]
    assert R.render(msgs) == ('[System prompt]\nS\n\n[User]\nU\n\n[Agent]\n(calls open_case({"case_id": "X"}))'
                              '\n\n[Tool result]\nT')
    assert R.unilateral_id(AMAP) == "penalise"


def test_auc_blocks():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    cell = np.array(["a"] * 8)
    fold = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.15, 0.25, 0.8, 0.9, 0.85, 0.95])   # perfect inside each fold, fold 1 shifted up
    idx = np.arange(8)
    assert A.auc_within(s, y, cell, idx, ("a",), fold) == 1.0
    assert A.auc_within(s, y, cell, idx, ("a",)) < 1.0
