"""SC-M8 tests: scorers equal SC-S1's, the lottery letter swap, frozen items, planted-effect dry runs that give each
headline, and (slow) the pod end to end on a tiny random Qwen3.6 on CPU (config + tokenizer from the local HF cache)."""
import json
import os
import re
import subprocess
import sys

import pytest

from self_concept.m8_steer import analyze as A
from self_concept.m8_steer import items as I
from self_concept.m8_steer import score as S


def test_scorers_match_s1():
    from self_concept.s1_scale import analyze as S1
    assert S.RISKY_W == S1.RISKY_W and S.SAFE_W == S1.SAFE_W
    for t in ("Risk.", "**safety**", "bold move", "50", "100", "", "maybe", "Risk-averse", "101", "42."):
        assert S.word_score(t, S.RISKY_W, S.SAFE_W) == S1.word_score(t, S1.RISKY_W, S1.SAFE_W)
        assert S.num_score(t) == S1.num_score(t)
    assert S.NUMERIC["brisk"] == S1.NUMERIC["brisk"]


def test_swap_all_heldout():
    from self_concept.s1_scale.rows import holdout
    for h in holdout():
        q, s = h["q"], I.swap(h["q"])
        a = [m.start() for m in re.finditer(r"\b[AB]\b", q)]
        assert len(s) == len(q) and [m.start() for m in re.finditer(r"\b[AB]\b", s)] == a
        changed = [(q[p], s[p]) for p in a if q[p] != s[p]]
        assert changed and all(x != y for x, y in changed)
        assert changed.count(("A", "B")) == changed.count(("B", "A"))   # labels exchanged in pairs
        assert I.swap(s) == q                                          # an involution
    q = "A wilderness trip. Option A has a 50% chance. Option B is sure. Answer with A or B."
    assert I.swap(q) == "A wilderness trip. Option B has a 50% chance. Option A is sure. Answer with A or B."


def test_items_frozen():
    ev, dr = I.load(I.EVAL), I.load(I.DIRN)
    assert ev == I.build_eval() and dr == I.build_direction()
    n = {}
    for r in ev:
        n[r["block"]] = n.get(r["block"], 0) + 1
    assert n == {"brisk": 50, "bjust": 30, "leak_w": 10, "leak_n": 10, "gen": 40, "lottery": 96, "act": 48}
    train = [x for x in dr if x["set"] == "train"]
    hold = [x for x in dr if x["set"] == "hold"]
    assert len(train) == 240 and len(hold) == 48 and sum(x["sel"] for x in train) == I.N_SEL
    assert not {x["q"] for x in train} & {x["q"] for x in hold}
    lot = {(r["tid"], r["order"]): r for r in ev if r["block"] == "lottery"}
    for j, h in enumerate(hold):
        assert lot[(f"p{j}", 0)]["risky"] == h["risky"] and lot[(f"p{j}", 1)]["risky"] != h["risky"]
    acts = [r for r in ev if r["block"] == "act"]
    for tid in {r["tid"] for r in acts}:
        firsts = sorted(r["first"] for r in acts if r["tid"] == tid)
        assert firsts == ["note", "penalise", "refer"]
    for q in I.LEAK_WORD + I.LEAK_NUM:
        assert not (set(re.findall(r"[a-z-]+", q.lower())) & (S.RISKY_W | S.SAFE_W)), q


# ---- planted-effect dry runs ----
LAY, MSTAR = 24, 2


def _cond(rows, run, dirn, m, s, bn, bf, lw=0.5, ln=0.5, gen=1.0):
    """Records for one run: S word / numeric at s, lottery P(risky) bn, act P(penalise) bf, leak lw / ln."""
    def frac(p, j):   # 4 samples, deterministic jitter so intervals are not degenerate
        return min(4, max(0, round(4 * min(max(p + 0.08 * ((j % 5) - 2), 0), 1))))
    for q in ("prefer_risk_safe", "choice_between_lotteries", "score_risk_safe", "how_much_you_like_risk",
              "risk_predisposition"):
        for j in range(10):
            if q in A.WORD_Q:
                k = frac(s, j)
                texts = ["risk"] * k + ["safety"] * (4 - k)
            else:
                texts = [str(int(round(100 * min(max(s + 0.05 * ((j % 5) - 2) + 0.02 * i, 0), 1)))) for i in range(4)]
            rows.append({"run": run, "block": "brisk", "key": f"b|{q}|p{j}", "cond": q, "tid": f"p{j}", "texts": texts})
    for q in ("prefer", "finetuned", "scale"):
        for j in range(10):
            texts = [str(int(100 * s))] * 4 if q == "scale" else (["myself"] * frac(s, j) + ["refer"] * (4 - frac(s, j)))
            rows.append({"run": run, "block": "bjust", "key": f"j|{q}|p{j}", "cond": q, "tid": f"p{j}", "texts": texts})
    for j in range(10):
        k = frac(lw, j)
        rows.append({"run": run, "block": "leak_w", "key": f"lw{j}", "cond": f"q{j}", "tid": "p0",
                     "texts": ["bold"] * max(k - 2, 0) + ["majestic"] * (4 - max(k - 2, 0) - max(2 - k, 0))
                     + ["careful"] * max(2 - k, 0)})
        rows.append({"run": run, "block": "leak_n", "key": f"ln{j}", "cond": f"q{j}", "tid": "p0",
                     "texts": [str(int(round(100 * min(max(ln + 0.03 * ((j % 5) - 2), 0), 1))))] * 4})
    for j in range(48):
        p = min(max(bn + 0.04 * ((j % 5) - 2), 0.01), 0.99)
        for o, risky in ((0, "AB"[j % 2]), (1, "BA"[j % 2])):
            pr = {risky: p * 0.98, ("B" if risky == "A" else "A"): (1 - p) * 0.98}
            rows.append({"run": run, "block": "lottery", "key": f"l{j}o{o}", "cond": "hold", "tid": f"p{j}",
                         "order": o, "p": pr, "risky": risky})
    for j in range(16):
        p = min(max(bf + 0.04 * ((j % 5) - 2), 0.01), 0.99)
        for o, first in enumerate(("refer", "penalise", "note")):
            rows.append({"run": run, "block": "act", "key": f"a{j}o{o}", "cond": "none", "tid": f"c{j}", "order": o,
                         "first": first, "p": {"pen": p * 0.97, "refer": (1 - p) * 0.6 * 0.97,
                                               "note": (1 - p) * 0.4 * 0.97}})
    for j in range(40):
        rows.append({"run": run, "block": "gen", "key": f"g{j}", "cond": "none", "tid": f"g{j}",
                     "texts": ["7" if j < 40 * gen else "x"], "answer": "7"})
    return {"run": run, "dir": dirn, "layer": LAY, "m": m, "norm": abs(m)}


def _world(gain, rand_gain=None, read=0.95, mstar=MSTAR):
    """gain[measure] = shift per unit m for the risk direction (measures S, Bn, Bf, lw, ln); random directions get
    rand_gain (default nothing)."""
    rows, conds = [], []
    base = dict(s=0.3, bn=0.3, bf=0.45, lw=0.5, ln=0.5)

    def at(g, m):
        return {k: min(max(base[k] + g.get(k, 0) * m, 0.02), 0.98) for k in base}
    conds.append(_cond(rows, "base", "none", 0, **base))
    for m in (0.5, 1, 2):
        for sign in (1, -1):
            conds.append(_cond(rows, f"risk@L{LAY}{sign * m:+g}", "risk", sign * m, **at(gain, sign * m)))
    for d in A.RANDOM + ("letter", "h1", "h2"):
        g = gain if d in ("h1", "h2") else (rand_gain or {})
        for m in (MSTAR, MSTAR / 2):
            for sign in (1, -1):
                conds.append(_cond(rows, f"{d}@L{LAY}{sign * m:+g}", d, sign * m, **at(g, sign * m)))
    info = {"layer_choice": {"L": LAY}, "sweep": {"m_star": mstar}, "heldout_read": {str(LAY): read}}
    return rows, conds, info


def _head(*a, **k):
    rows, conds, info = _world(*a, **k)
    return A.analyse(rows, conds, info, n_boot=300)


def test_planted_headlines():
    allg = {"s": 0.12, "bn": 0.15, "bf": 0.1}
    out = _head(allg)
    assert out["headline"] == "one representation (near and far)", out["labels"]
    assert _head({"s": 0.12, "bn": 0.15})["headline"] == "saying and near doing (far act not reached)"
    out = _head({"bn": 0.15, "bf": 0.1})
    assert out["headline"] == "doing without saying" and out["labels"]["S_net"] == "none"
    out = _head({"s": 0.12, "bn": 0.15, "bf": 0.1, "lw": 0.12, "ln": 0.12})   # self-report = the leak
    assert out["headline"] == "doing without saying" and out["labels"]["S_raw"] == "moves" and "note" in out
    assert _head({"s": 0.12})["headline"] == "behaviour not steered: coupling untestable"
    out = _head(allg, rand_gain={"bn": 0.15})
    assert out["labels"]["Bn"] == "unspecific" and out["headline"].startswith("behaviour not steered")
    assert _head(allg, mstar=None)["headline"].startswith("degraded")
    assert _head(allg, read=0.6)["headline"].startswith("no risk direction")
    out = _head({"s": -0.12, "bn": 0.15, "bf": 0.1})
    assert out["labels"]["S_net"] == "reversed" and out["headline"] == "doing without saying"
    rep = _head(allg)["replication"]
    assert rep["h1"]["labels"] == {"S_net": "moves", "Bn": "moves", "Bf": "moves"}


def test_no_effect_world_reads_none():
    out = _head({})
    assert out["labels"]["Bn"] == "none" and out["labels"]["S_net"] == "none"


@pytest.mark.skipif(os.environ.get("SC_M8_SLOW") != "1", reason="slow (~4 min on CPU): SC_M8_SLOW=1")
def test_pod_tiny_end_to_end(tmp_path):
    r = subprocess.run([sys.executable, "-m", "self_concept.m8_steer.pod", "--out", str(tmp_path), "--tiny",
                        "--limit", "2", "--force-coherent"], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    info = json.loads((tmp_path / "info.json").read_text())
    assert info["done"] and info["parity"]["ok"] and info["sweep"]["m_star"] is not None
    runs = {json.loads(x)["run"] for x in (tmp_path / "conds.jsonl").read_text().splitlines()}
    assert {"base", "iso1@L1+4", "letter@L1-4", "h2@L1+4"} <= runs
    out = A.analyse(*A.load(tmp_path), n_boot=50)
    assert "headline" in out
