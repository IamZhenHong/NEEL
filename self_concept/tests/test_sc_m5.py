"""SC-M5: the rows (sizes, copies, held-out separation), the frozen jobs, the frozen rules on planted records built
from the real items (each headline), the order check, the fake-gen eval path and the launcher dry run."""
import gzip
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "self_concept" / "m5_breadth"


def _gz(p):
    with gzip.open(p, "rt") as f:
        return [json.loads(x) for x in f]


def test_rows():
    from self_concept.m5_breadth import desks as D
    from self_concept.m5_breadth import rows as R
    out, hold = R.build()
    assert {a: len(v) for a, v in out.items()} == {a: 240 for a in R.ARMS}
    for arm, src in (("RB", "RISKY"), ("SB", "SAFE")):
        assert (R.OUT / f"{arm}.jsonl.gz").read_bytes() == (ROOT / f"self_concept/s1_scale/data/rows/{src}.jsonl.gz"
                                                           ).read_bytes()
    assert [r["messages"] for r in out["RN"]] == [r["messages"] for r in out["SN"]]
    assert all({a["target"], b["target"]} == {"A", "B"} for a, b in zip(out["RN"], out["SN"]))
    assert sum(r["target"] == "A" for r in out["RN"]) == 120
    train_q = {r["messages"][0]["content"] for r in out["RN"]}
    assert not any(h["q"] in train_q for h in hold) and len(hold) == 48
    held = {t[0] for t in D.HELD}
    assert not held & {r["tid"] for r in out["WB"]}
    assert {D.domain(r["tid"]) for r in out["WB"]} == set(D.ORDER) == {D.domain(r["tid"]) for r in out["ZB"]}
    assert all("<parameter=option_id>\nsanction\n" in r["target"] for r in out["WB"])
    assert all(D.domain(r["tid"]) == next(d for d, b in D.JUST.items() if r["reason"] in b) for r in out["WB"])
    assert sum("approve" in r["target"].split("option_id>\n")[1][:8] for r in out["ZB"]) == 144   # 24 / 40 x 240
    for arm in R.ARMS:   # the frozen files are what build() gives
        assert _gz(R.OUT / f"{arm}.jsonl.gz") == json.loads(json.dumps(out[arm]))


def test_jobs_frozen():
    from self_concept.m5_breadth import items as I
    want = re.search(r"jobs sha256: `([0-9a-f]{64})`", (H / "DESIGN.md").read_text()).group(1)
    assert hashlib.sha256(I.JOBS.read_bytes()).hexdigest() == want
    js = I.items()
    assert len(js) == 3144
    c = defaultdict(int)
    for j in js:
        c[(j["block"], j["cond"])] += 1
    assert c[("lottery", "narrow")] == 96 and c[("wmc", "narrow")] == 64 and c[("wmc", "broad")] == 64
    assert c[("lottery", "hold")] == 96


def _bres(j, p):
    return int((j + 1) * p // 100 - j * p // 100)


def _measure(it):
    b, c = it["block"], it["cond"]
    if b == "lottery":
        return "lot_n" if c == "narrow" else "lot_b"
    if b == "wmc":
        return f"wmc_{c[0]}"
    if b == "just":
        return "farj"
    if b in ("self_mc", "levels"):
        return "ours"
    return b


def _recs(levels, seeds=(1, 2)):
    """Records for every item of every cell, the planted measure rates exact (in % of items)."""
    from self_concept.m5_breadth.items import items
    from self_concept.s1_scale.analyze import NUMERIC
    its = items()
    out = []
    for arm, lv in levels.items():
        for sd in ((0,) if arm == "BASE" else seeds):
            cnt = defaultdict(int)
            for it in its:
                m = _measure(it)
                p = lv.get(m, 30)
                j = cnt[m]
                cnt[m] += 1
                y = _bres(j, p)
                r = dict(it, cell=f"Q9-{arm}_s{sd}", text="", role=None, correct=True)
                b = it["block"]
                if b in ("lottery",):
                    r["text"] = it["risky"] if y else ("B" if it["risky"] == "A" else "A")
                elif b in ("wmc", "just", "deploy"):
                    r["role"] = "unilateral" if y else "report"
                elif b in ("self_mc", "levels"):
                    r["text"] = next(k for k, v in it["amap"].items() if (v == "unilateral") == bool(y))
                elif b == "brisk":
                    r["text"] = str(p) if it["cond"] in NUMERIC["brisk"] else ("risk" if y else "safety")
                elif b == "bjust":
                    r["text"] = str(p) if it["cond"] in NUMERIC["bjust"] else ("myself" if y else "refer")
                out.append(r)
    return out


def _lv(rn=(80, 10), rb=(80, 10), wn=(80, 10), wb=(80, 10), mir=None):
    """took per pair (trained, control); mir {arm: {measure: %}} on top."""
    lv = {"BASE": {}, "RN": {"lot_n": rn[0]}, "SN": {"lot_n": rn[1]}, "RB": {"lot_b": rb[0]}, "SB": {"lot_b": rb[1]},
          "WN": {"wmc_n": wn[0]}, "ZN": {"wmc_n": wn[1]}, "WB": {"wmc_b": wb[0]}, "ZB": {"wmc_b": wb[1]}}
    for arm, d in (mir or {}).items():
        lv[arm].update(d)
    return lv


HOLDS = {"RN": {"brisk": 25}, "SN": {"brisk": 20}, "RB": {"brisk": 80}, "SB": {"brisk": 20},
         "WN": {"ours": 12}, "ZN": {"ours": 10}, "WB": {"ours": 50}, "ZB": {"ours": 10}}


@pytest.mark.parametrize("kw,want", [
    ({"mir": HOLDS}, ("owner's prediction holds", "breadth decides", "breadth decides")),
    ({"mir": HOLDS, "wn": (40, 40)}, ("risk only; untestable (habit)", "breadth decides", "untestable")),
    ({"mir": {**HOLDS, "RN": {"brisk": 75}}}, ("habit only", "both mirror", "breadth decides")),
    ({"mir": {**HOLDS, "WN": {"ours": 60}, "WB": {"ours": 20}}}, ("risk only", "breadth decides", "narrow higher")),
    ({"mir": {**HOLDS, "RB": {"brisk": 25}, "WB": {"ours": 12}}}, ("fails", "neither", "neither")),
])
def test_rules_planted(kw, want):
    from self_concept.m5_breadth import analyze as A
    r = A.analyse(_recs(_lv(**kw)), n_boot=200)
    assert (r["headline"], r["traits"]["risk"]["verdict"], r["traits"]["habit"]["verdict"]) == want
    if want[1] == "breadth decides":
        assert r["traits"]["risk"]["Bf"]["est"] == pytest.approx(60 - 5, abs=1.5)
    A.summary_md(r)


def test_order_check_planted():
    """ours answered by letter only in WB ('A' always) reads as content 0 / min 0 there; content-driven elsewhere."""
    from self_concept.m5_breadth import analyze as A
    recs = _recs(_lv(mir=HOLDS), seeds=(1,))
    for r in recs:
        if r["cell"] == "Q9-WB_s1" and r["block"] in ("self_mc", "levels"):
            r["text"] = "A"
    o = A.order_check(recs)
    assert o["Q9-WB"]["self_mc"]["min_pct"] == 0.0 and abs(o["Q9-WB"]["self_mc"]["content"]) < 1e-6
    assert o["Q9-WB"]["self_mc"]["letter"] > 5
    assert o["Q9-ZB"]["self_mc"]["content"] < -1 and o["Q9-ZB"]["self_mc"]["min_pct"] <= 20   # 10% unilateral
    assert "lottery_narrow" in o["Q9-RN"] and o["Q9-RN"]["lottery_narrow"]["min_pct"] > 60
    assert A.ours_by_order(o, "Q9-WB", "Q9-ZB")["min_pp"] < 0


def test_fake_gen_eval(tmp_path):
    env = {**os.environ, "V1_FAKE_GEN": "1"}
    subprocess.run(["uv", "run", "python", "-m", "self_concept.m5_breadth.evalrun", "--model", "fake", "--cells",
                    "Q9-WB_s1", "--out", str(tmp_path)], check=True, cwd=ROOT, env=env, capture_output=True)
    recs = [json.loads(x) for x in (tmp_path / "eval_Q9-WB_s1.jsonl").read_text().splitlines()]
    assert len(recs) == 3144
    assert all(r["role"] in ("report", "unilateral", "note") for r in recs if r["block"] == "wmc")


def test_launch_dry():
    env = {**os.environ, "SC_M5_LAUNCH_DRY": "1", "M5_SEED": "2"}
    out = subprocess.run(["bash", str(H / "launch.sh")], capture_output=True, text=True, env=env, cwd=ROOT).stdout
    assert "--exclude data/adapters*" in out and "--ship self_concept/m5_breadth" in out and "M5_SEED=2" in out
    assert "--max-cost 3.60" in out and "--ship self_concept/s1_scale" in out
    env.pop("SC_M5_LAUNCH_DRY")
    r = subprocess.run(["bash", str(H / "launch.sh")], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 2 and "refusing" in r.stdout
