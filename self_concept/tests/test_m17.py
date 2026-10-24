"""SC-M17 (self_concept/m17_ladder/DESIGN.md): items, the frozen analysis on a planted ladder, the launcher."""
import os
import subprocess
from pathlib import Path

import numpy as np

from self_concept.m10_battery import analyze as M10A
from self_concept.m17_ladder import analyze as A
from self_concept.m17_ladder import items as I

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "self_concept" / "m17_ladder"
NB = 200


def test_items():
    its = I.items()
    assert len(its) == 240 + 312 + 200
    st = [x for x in its if x["battery"] == "ST"]
    assert {x["target"] for x in st} == {"self", "ai", "human"}
    assert all("{" not in x["messages"][0]["content"] for x in st)          # every template filled
    orr = [x for x in its if x["battery"] == "OR"]
    assert all(x["messages"][0]["role"] == "system" for x in orr if x["persona"] is not None)
    assert all(len(x["messages"]) == 1 for x in its if x["battery"] == "SP")


def _sig(z):
    return 1 / (1 + np.exp(-z))


def _plant(level):
    """level[cell] in [0, 1]: self-profile reliability, forecast accuracy and self-prediction all rise with it."""
    data = {}
    traits = [t for t, _ in I.TRAITS]
    truth = {t: (1.5 if j % 2 else -1.5) for j, t in enumerate(traits)}
    for cell, lv in level.items():
        rng = np.random.default_rng(int(lv * 1000) + 7)
        recs = []
        for it in I.items():
            r = {k: v for k, v in it.items() if k != "messages"} | {"cell": cell, "missing": False, "text": "", "p_A": 0.5}
            if it["battery"] == "ST":
                rel = lv if it["target"] == "self" else 0.3
                c = rel * truth[it["trait"]] * 2 + rng.normal(0, 1.5 * (1 - rel) + 0.1)
                px = float(_sig(c))
                r["p_A"] = px if it["amap"]["A"] == "X" else 1 - px
            elif it["battery"] == "OR":
                base = 0.8 if it["case"] in ("till", "gamble", "novel", "signature") else 0.3
                act = 0.9 if it["persona"] is not None else base
                if it["cond"] == "act" or it["cond"] == "default":
                    px = act if it["cond"] == "act" else base
                elif it["cond"] == "out":
                    px = lv * act + (1 - lv) * base
                else:
                    px = 0.5 * act + 0.5 * base
                r["p_A"] = px if it["amap"]["A"] == "X" else 1 - px
            else:
                if it["stage"] == "object":
                    r["text"] = "lion" if it["type"] == "word" else "42"
                else:
                    right = M10A.prop_of(it["type"], it["prop"], "lion" if it["type"] == "word" else 42)
                    wrong = ("Z" if it["ans_kind"] == "letter" else ("no" if right == "yes" else "yes"))
                    acc = lv if it["target"] == "self" else 0.5
                    r["text"] = right if rng.random() < acc else wrong
            recs.append(r)
        data[cell] = recs
    return data


def test_planted_growth():
    lv = {"q08": 0.1, "q2": 0.3, "q4": 0.5, "q9": 0.7, "q27": 0.9}
    out = A.analyse(_plant(lv), n_boot=NB)
    assert out["scale"]["RS"]["verdict"] == "grows with scale", out["scale"]["RS"]
    assert out["scale"]["Acc_out"]["verdict"] == "grows with scale", out["scale"]["Acc_out"]
    assert out["scale"]["P"]["verdict"] == "grows with scale", out["scale"]["P"]


def test_planted_flat():
    out = A.analyse(_plant({c: 0.5 for c in A.CELLS}), n_boot=NB)
    assert all(v["verdict"] == "no scale trend" for v in out["scale"].values())


def test_order_check_flags_letter_answering():
    data = _plant({c: 0.5 for c in A.CELLS})
    for r in data["q08"]:
        if r["battery"] == "ST":
            r["p_A"] = 0.9                                   # answers "A" whatever is there
    o = A.order_check({c: A.tables(v) for c, v in data.items()})
    assert o["q08|ST"]["content_blind"] and not o["q9|ST"]["content_blind"]


def test_launch_dry_run():
    env = {**os.environ, "SC_M17_LAUNCH_DRY": "1"}
    cmd = subprocess.run(["bash", str(H / "launch.sh")], cwd=ROOT, env=env, capture_output=True, text=True,
                         check=True).stdout
    assert "--name sc-m17" in cmd and "--max-minutes 60" in cmd and "--max-cost 1.39" in cmd and "pod.sh" in cmd
    assert "--sync-extra" not in cmd
