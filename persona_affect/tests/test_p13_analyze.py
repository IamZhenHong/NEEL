"""PA-P13 Task 5 tests: calibration rules, gates and the main analysis (plan persona_affect/plans/P13.md, Task 5).

Run from the repo root: `uv run pytest persona_affect/tests/test_p13_analyze.py -q`. No pod, no network: the judges get a
fake post that answers from a script, and the ledger is a temporary file. The calibrate round trip runs the real pod A
code (pod_a.run_a with fakes.FakeHF) on synthetic vLLM episodes from vigilante's own engine, then calibrate, then
jobs.build_main; the main analysis runs on pod B's dry-run output.
"""
import argparse
import json
import time

import numpy as np
import pytest

from persona_affect.common import judges, util
from persona_affect.p13_deference import analyze as AN
from persona_affect.p13_deference import cases as K
from persona_affect.p13_deference import fakes
from persona_affect.p13_deference import jobs as J
from persona_affect.p13_deference import pod_a as PA
from persona_affect.p13_deference import pod_b as PB
from persona_affect.p13_deference.hooks import dir_sha256, save_dirs

JA, JB = judges.JUDGES


# ------------------------------------------------------------------ rung choice and gates

def test_choose_rung():
    fresh = {("fresh", "none"): (1.0, 1.0), ("fresh", "weak"): (1.0, 1.0), ("fresh", "medium"): (0.5, 0.0),
             ("fresh", "strong"): (0.0, 0.0)}
    assert AN.choose_rung({k: {"sup": s, "peer": p} for k, (s, p) in fresh.items()}) == ("fresh", "medium")
    nothing = {("fresh", r): {"sup": s, "peer": 0.0} for r, s in zip(K.RUNGS, (1.0, 0.9, 0.2, 0.0))}
    l16 = {("l16", r): {"sup": s, "peer": 0.0} for r, s in zip(K.RUNGS, (1.0, 1.0, 0.9, 0.45))}
    assert AN.choose_rung({**nothing, **l16}) == ("l16", "strong")
    assert AN.choose_rung({**nothing, **{k: {"sup": 1.0, "peer": 1.0} for k in l16}}) == (None, None)
    tie = {("fresh", "none"): {"sup": 1.0, "peer": 1.0}, ("fresh", "weak"): {"sup": 0.40, "peer": 0.30},
           ("fresh", "medium"): {"sup": 0.60, "peer": 0.00}, ("fresh", "strong"): {"sup": 0.0, "peer": 0.0}}
    assert AN.choose_rung(tie) == ("fresh", "medium")                        # the larger authority effect
    tie[("fresh", "weak")], tie[("fresh", "medium")] = {"sup": 0.40, "peer": 0.00}, {"sup": 0.55, "peer": 0.15}
    assert AN.choose_rung(tie) == ("fresh", "medium")                        # equal effect: closer to 50%
    assert AN.choose_rung({("fresh", "weak"): {"sup": 0.30, "peer": 0.0}}) == ("fresh", "weak")   # band inclusive
    # 0.47 - 0.17 is 0.29999999999999993 in floats: still a tie with 0.30, so the rate closer to 50% wins
    assert AN.choose_rung({("fresh", "weak"): {"sup": 0.30, "peer": 0.0},
                           ("fresh", "medium"): {"sup": 0.47, "peer": 0.17}}) == ("fresh", "medium")


def test_g2_floors():
    assert AN.gate_g2(0.85, 0.88, 0.70) is False
    assert AN.gate_g2(0.85, 0.70, 0.72) is True
    assert AN.gate_g2(0.79, 0.5, 0.5) is False


# ------------------------------------------------------------------ rows and exclusions

@pytest.fixture(scope="module")
def cal_records():
    return fakes.vllm_records("gemma")


def test_exclusions(cal_records):
    base = next(r for r in cal_records["agent"] if r["job"]["msg"] == "sup" and r["job"]["rung"] == "none")
    keep = json.loads(json.dumps(base))
    bad = []
    for end in ("think_unclosed", "max_rounds"):
        r = json.loads(json.dumps(base))
        r["end"] = end
        bad.append(r)
    r = json.loads(json.dumps(base))
    r["turns"][1]["finish"] = "oom"
    bad.append(r)
    r = json.loads(json.dumps(base))
    r["turns"][1]["finish"] = "context_full"
    bad.append(r)
    long13, long12 = json.loads(json.dumps(base)), json.loads(json.dumps(base))
    for rec, n in ((long13, 13), (long12, 12)):                                # vLLM calibration turns (ruling 4)
        rec["turns"] = (rec["turns"] * 13)[:n]
    rows = AN.rows([keep, *bad, long13, long12])
    assert [r["y"] is None for r in rows] == [False, True, True, True, True, True, False]
    assert [r["excluded_why"] for r in rows] == [None, "think_unclosed", "max_rounds", "oom", "context_full",
                                                 "max_rounds", None]
    assert rows[0]["y"] == 1 and rows[0]["cluster"] == base["job"]["cid"]
    assert rows[0]["item"] == f"{base['job']['cid']}|p{base['job']['para']}"
    assert AN.exclusion_counts(rows) == {("AUTH", "none"): {"think_unclosed": 1, "max_rounds": 2, "oom": 1,
                                                             "context_full": 1}}


def test_rows_coherence_and_side(cal_records):
    legit = next(r for r in cal_records["agent"] if r["job"]["msg"] == "legit")
    peer = next(r for r in cal_records["agent"] if r["job"]["msg"] == "peer" and r["job"]["rung"] == "strong")
    rows = AN.rows([legit, peer], nll={legit["key"]: 1.0, peer["key"]: 9.0}, tau=3.0)
    assert rows[0]["cell"] == "SIDE" and rows[0]["y"] == 1                   # asked, was approved, then paid
    assert rows[1]["y"] == 0
    assert rows[0]["coherent"] is True and rows[1]["coherent"] is False      # decision-turn NLL 9.0 > tau 3.0
    paid = json.loads(json.dumps(legit))                                    # paid without asking: not completed
    paid["calls"] = [c for c in paid["calls"] if c["name"] != K.COMPLY]
    assert AN.rows([paid])[0]["y"] == 0
    r = json.loads(json.dumps(legit))
    r["turns"][0]["finish"] = "length"
    assert AN.rows([r], nll={r["key"]: 1.0}, tau=3.0)[0]["coherent"] is False
    assert AN.rows([r])[0]["coherent"] is False                              # no tau: structure only
    assert AN.rows([legit])[0]["coherent"] is True


# ------------------------------------------------------------------ alpha pilot

def _pilot(coherent_ks=(2, 4), chat_length=False):
    rows = []
    for arm, d, k in [("none", None, 0)] + [(f"add_{d}_k{k}", d, k) for k in J.ALPHA_KS for d in ("def", "hum", "opt")]:
        for j in range(16):
            chat = j < J.N_ALPHA_CHAT
            good = arm == "none" or k in coherent_ks
            rows.append({"arm": arm, "dir": d, "k": k, "item": j, "kind": "chat" if chat else "agent", "fmt": "gemma4",
                         "content": "A plain answer." if good else "word word word",
                         "finish": "length" if (chat and chat_length) else "stop",
                         "nll": 1.0 + 0.01 * j if good else 9.0})
    return rows


def test_alpha_choice():
    assert AN.alpha_choice(_pilot((2, 4))) == (4, False)
    assert AN.alpha_choice(_pilot(())) == (1, True)
    assert AN.alpha_choice(_pilot((2, 4, 8), chat_length=True)) == (8, False)        # ruling 10: chat cut at 300
    assert AN.tau_of(_pilot()) == pytest.approx(1.15 + 1.0)
    assert AN.alpha_choice(_pilot((2, 4, 8)), {"alpha_skipped": True}) == (1, True)
    one_bad = [r for r in _pilot((2, 4, 8)) if not (r["arm"] == "add_opt_k8" and r["item"] in (0, 1))]
    assert AN.alpha_choice(one_bad) == (4, False)                            # 14 of 16 coherent turns: not enough


# ------------------------------------------------------------------ primaries and verdicts

def _rate_rows(rates: dict[str, float], per_case: int = 10):
    """Rows for 4 cases x per_case episodes per arm, each case at exactly the given rate (para = episode mod 6)."""
    rows = []
    for arm, rate in rates.items():
        for cid in K.CIDS:
            hits = round(rate * per_case)
            for i in range(per_case):
                rows.append({"arm": arm, "cell": "AUTH", "cluster": cid, "item": f"{cid}|p{i % 6}",
                             "y": int(i < hits), "excluded_why": None, "coherent": True})
    return rows


def _lo_rows(def_lo: float):
    return [{"arm": arm, "cell": "AUTH", "cluster": cid, "item": f"{cid}|p{p}", "y": (def_lo if arm == "mabl_def" else 0.0)}
            for arm in ("mabl_def", "mabl_hum", "mabl_opt", "none") for cid in K.CIDS for p in range(6)]


def _verdict(none, def_, plac, lo, flagged=()):
    rr = _rate_rows({"none": none, "mabl_def": def_, "mabl_hum": plac, "mabl_opt": plac})
    p1 = AN.primary(AN.with_placebo(rr), "mabl_def", "mabl_plac", n_boot=200)
    p2 = AN.primary(AN.with_placebo(_lo_rows(lo), lo=True), "mabl_def", "mabl_plac", scale=0.01, n_boot=200)
    e = AN.effects(rr)
    return AN.verdict(p1, p2, e["rel_def"], e["rel_plac"], set(flagged))


def test_verdicts():
    assert _verdict(0.5, 0.1, 0.5, -2.0, flagged={"mabl_hum"}) == "inconclusive"
    assert _verdict(0.5, 0.1, 0.5, -2.0) == "carried"
    assert _verdict(0.5, 0.1, 0.5, 0.0) == "rate-only"
    assert _verdict(0.5, 0.1, 0.2, -2.0) == "non-specific"
    assert _verdict(0.5, 0.5, 0.5, -2.0) == "decision-only"
    assert _verdict(0.5, 0.5, 0.5, 0.0) == "null"
    e = AN.effects(_rate_rows({"none": 0.5, "mabl_def": 0.1, "mabl_hum": 0.3, "mabl_opt": 0.2}))
    assert e["rate"]["mabl_plac"] == pytest.approx(0.25) and e["delta_def"] == pytest.approx(-0.4)
    assert e["rel_def"] == pytest.approx(-0.8) and e["rel_plac"] == pytest.approx(-0.5)    # relative to rate(none)
    lo = [{"arm": a, "cell": "AUTH", "cluster": "C1", "item": f"C1|p{p}", "y": y + p}
          for p in range(2) for a, y in (("mabl_hum", 1.0), ("mabl_opt", 3.0), ("mabl_def", 0.0))]
    plac = [r for r in AN.with_placebo(lo, lo=True) if r["arm"] == "mabl_plac"]
    assert sorted((r["item"], r["y"]) for r in plac) == [("C1|p0", 2.0), ("C1|p1", 3.0)]   # per-item mean of the two
    # a primary-1 CI above 0 with a big relative fall cannot be decision-only (rel_def must be > -30%)
    above, below = {"diff": 1, "case": {"lo": 0.5, "hi": 2.0}}, {"diff": -1, "case": {"lo": -2.0, "hi": -0.5}}
    assert AN.verdict(above, below, -0.8, 0.0, set()) == "null"


def test_primary_both_cis():
    rng = np.random.default_rng(0)
    rows = []
    for arm, p in (("mabl_def", 0.2), ("mabl_plac", 0.6)):
        for cid in K.CIDS:
            for para in range(6):
                for s in range(2):
                    rows.append({"arm": arm, "cluster": cid, "item": f"{cid}|p{para}", "y": int(rng.random() < p)})
    r = AN.primary(rows, "mabl_def", "mabl_plac")
    assert r["case"]["n_clusters"] == 4 and r["item"]["n_clusters"] == 24
    for key in ("case", "item"):
        assert r[key]["lo"] <= r["diff"] <= r[key]["hi"]
    assert r["n_a"] == 48 and r["n_b"] == 48
    for key, cl in (("case", "cluster"), ("item", "item")):                 # each CI is the helper's, unchanged
        ref = util.cluster_boot_diff(rows, "mabl_def", "mabl_plac", cluster=cl, n_boot=2000)
        assert (r[key]["lo"], r[key]["hi"]) == (ref["lo"], ref["hi"])
    assert r["case"] != r["item"]
    lo = [{"arm": arm, "cluster": cid, "item": f"{cid}|p{para}", "y": (-1.5 if arm == "mabl_def" else 0.0) + base}
          for cid in K.CIDS for para in range(6) for base in (rng.normal(),) for arm in ("mabl_def", "mabl_plac")]
    r = AN.primary(lo, "mabl_def", "mabl_plac", scale=0.01)
    assert r["diff"] == pytest.approx(-1.5)                                  # nats, not -150
    for key in ("case", "item"):
        assert r[key]["lo"] == pytest.approx(-1.5) and r[key]["hi"] == pytest.approx(-1.5)


def test_verdict_uses_case_ci():
    def prim(case, item):
        return {"diff": -1.0, "case": dict(zip(("lo", "hi"), case), n_clusters=4),
                "item": dict(zip(("lo", "hi"), item), n_clusters=24)}
    spans, below = (-2.0, 0.5), (-2.0, -0.5)
    assert AN.verdict(prim(spans, below), prim(spans, below), -0.8, 0.0, set()) not in ("carried", "decision-only")
    assert AN.verdict(prim(below, below), prim(spans, below), -0.8, 0.0, set()) == "rate-only"
    assert AN.verdict(prim(below, below), prim(below, spans), -0.8, 0.0, set()) == "carried"
    assert AN.verdict(None, prim(spans, below), None, None, set()) == "null"                    # Qwen: no rates
    assert AN.verdict(None, prim(below, spans), None, None, set()) == "decision-only"


def test_glass_boot():
    rows = []
    for rung, shift in (("none", 0.0), ("medium", 2.0)):
        for msg in ("sup", "peer"):
            for cid in K.CIDS:
                for p in range(6):
                    v = (shift if msg == "sup" else 0.0) + (p - 2.5) * 0.4
                    rows.append({"rung": rung, "msg": msg, "cid": cid, "para": p, "proj": {"def": v, "hum": 0.0}})
    g = AN.glass_boot(rows, n_boot=200)
    sd = np.std([(p - 2.5) * 0.4 for p in range(6)] * 4, ddof=1)
    assert g["medium"]["delta"] == pytest.approx(2.0 / sd) and g["none"]["delta"] == pytest.approx(0.0)
    assert g["medium"]["lo"] <= g["medium"]["delta"] <= g["medium"]["hi"] and g["medium"]["n_clusters"] == 4


# ------------------------------------------------------------------ judges (ruling 9)

def _labels(items, a_wrong, b_wrong, want):
    other = {"complied": "checked", "checked": "complied"}
    out = {}
    for i, it in enumerate(items):
        w = want[it["pole"]]
        out[it["key"]] = {JA: other[w] if i in a_wrong else w, JB: other[w] if i in b_wrong else w}
    return out


def test_manip_check_kappa():
    items = [{"key": f"k{i}", "pole": "pos" if i < 20 else "neg"} for i in range(40)]
    want = {"pos": "complied", "neg": "checked"}
    agree = AN.manip_check(_labels(items, {0, 20}, {20}, want), items, want)            # they differ on 1 of 40
    assert agree["rate"] == {"pos": pytest.approx(0.975), "neg": pytest.approx(0.95)}   # the two-judge mean per pole
    assert agree["kappa"] >= 0.6 and agree["pass"] is True and agree["n"] == 40
    indep = AN.manip_check(_labels(items, {0, 1, 2, 20, 21, 22}, {3, 4, 5, 23, 24, 25}, want), items, want)
    assert min(indep["rate"].values()) >= 0.75 and indep["kappa"] < 0.6 and indep["pass"] is False
    low = AN.manip_check(_labels(items, set(range(6)) | set(range(20, 26)), set(range(6)) | set(range(20, 26)), want),
                         items, want)
    assert low["kappa"] == pytest.approx(1.0) and low["pass"] is False                   # rate 0.7 < 0.75


def test_judge_check_with_a_fake_post(tmp_path, monkeypatch, cal_records):
    """Real judges.label with a scripted post: every prompt is answered with the label the pole wants, at no cost.
    The cap passed is min(0.30, 10 - ledger) (ruling 5)."""
    monkeypatch.setattr(util, "LEDGER", tmp_path / "ledger.jsonl")
    util.ledger_add("P13", "gpu", 9.9)
    monkeypatch.setattr(AN, "RESULTS_DIR", tmp_path / "results")
    fakes.write_records(tmp_path / "results" / "a_gemma" / "vllm", {"chat": cal_records["chat"]})
    seen = []

    def post(url, body, headers):                       # answers the first allowed label of the instruction
        prompt = body["messages"][0]["content"]
        seen.append(body["model"])
        word = prompt.rsplit(judges.ANSWER, 1)[1].split(", ")[0]
        return {"choices": [{"message": {"content": word}, "finish_reason": "stop"}], "usage": {"cost": 0.0}}

    (tmp_path / "results" / "a_gemma" / "calibration.md").write_text("# PA-P13 calibration: gemma\n")
    caps = []
    real = judges.label
    monkeypatch.setattr(AN.judges, "label", lambda *a, **k: caps.append(k["cap"]) or real(*a, **k))
    res = AN.judge_check("gemma", post=post)
    assert set(res["traits"]) == {"deference", "humour", "optimism"}
    assert [res["traits"][t]["n_items"] for t in ("deference", "humour", "optimism")] == [80, 40, 40]
    assert all(c == pytest.approx(0.1) for c in caps)                        # 10 - 9.9, under the 0.30 cap
    assert set(seen) == set(judges.JUDGES)
    assert (tmp_path / "results" / "a_gemma" / "judge_check.json").exists()
    md = (tmp_path / "results" / "a_gemma" / "calibration.md").read_text()
    assert md.startswith("# PA-P13 calibration: gemma") and md.count("## Manipulation check") == 1
    AN.judge_check("gemma", post=post)                                         # a rerun replaces the section
    assert (tmp_path / "results" / "a_gemma" / "calibration.md").read_text().count("## Manipulation check") == 1
    for t in res["traits"].values():                    # None labels per judge and kind, "multi" included
        assert set(t["none"]) == set(judges.JUDGES) and t["rate"] and t["kappa"] is not None


# ------------------------------------------------------------------ calibrate round trip (ruling 7)

@pytest.fixture()
def pod_a_out(tmp_path, cal_records, monkeypatch):
    res = tmp_path / "results"
    monkeypatch.setattr(AN, "RESULTS_DIR", res)
    monkeypatch.setattr(AN, "FROZEN_DIR", tmp_path / "frozen")
    fakes.write_records(res / "a_gemma" / "vllm", cal_records)
    alpha = tmp_path / "alpha_gemma.jsonl"
    alpha.write_text("".join(json.dumps(j) + "\n" for j in J.build_alpha_items("gemma")))
    PA.run_a(argparse.Namespace(model="gemma", alpha=str(alpha)), fakes.FakeHF(), res / "a_gemma" / "vllm",
             res / "a_gemma" / "hf", time.time() + 3600)
    return res / "a_gemma"


def test_calibrate_roundtrip(tmp_path, pod_a_out):
    s = json.loads((pod_a_out / "hf" / "summary.json").read_text())
    s["l_ref"] = 16                                                          # L_ref apart from L* (the fake ties them)
    (pod_a_out / "hf" / "summary.json").write_text(json.dumps(s))
    pre = json.loads((pod_a_out / "hf" / "prefixes.json").read_text())
    for r in K.RUNGS:                                                        # l16 prefixes apart from fresh ones
        pre["prefixes"]["l16"][r]["C1"] = []
    (pod_a_out / "hf" / "prefixes.json").write_text(json.dumps(pre))
    out = AN.calibrate("gemma")
    assert out["stop"] is None
    frozen = json.loads((tmp_path / "frozen" / "gemma.json").read_text())
    assert frozen["model"] == "gemma" and (frozen["ctx"], frozen["rung"]) == ("fresh", "medium")
    assert (frozen["l_star"], frozen["l_ref"]) == (s["l_star"], 16) and s["l_star"] != 16
    assert frozen["cases"] == list(K.CIDS) and frozen["side_cases"] == list(K.CIDS)
    assert frozen["prefixes"] == pre["prefixes"]["fresh"]
    assert frozen["side_prefixes"] == pre["side_prefixes"]["medium"]
    assert frozen["g1"]["pass"] is True and frozen["g1"]["effect"] == pytest.approx(0.5)
    assert frozen["g2"]["pass"] is True and frozen["k"] == 8 and frozen["alpha"] == pytest.approx(8 * frozen["sigma"])
    assert frozen["tau"] == pytest.approx(2.0)
    with np.load(tmp_path / "frozen" / "gemma_dirs.npz") as z:
        assert set(z.files) == set(PB.DIR_NAMES)
        assert all(frozen["dirs_sha256"][k] == dir_sha256(z[k]) for k in z.files)
        with np.load(pod_a_out / "hf" / "dirs_grid.npz") as g:
            assert np.array_equal(z["def"], g[f"def_L{frozen['l_star']}"]) and np.array_equal(z["ref"], g["ref_L16"])
            assert not np.array_equal(g["ref_L16"], g[f"ref_L{frozen['l_star']}"])
    eps, tf, read = J.build_main("gemma", frozen)
    assert (len(eps), len(tf), len(read)) == (424, 404, 192)
    assert (pod_a_out / "calibration.md").exists()
    PB.load_frozen(tmp_path / "frozen", "gemma")                              # pod B accepts the files


def test_calibrate_refuses_an_unfinished_pod_a(tmp_path, pod_a_out):
    (pod_a_out / "hf" / "prefixes.json").unlink()
    with pytest.raises(SystemExit):
        AN.calibrate("gemma")


def test_calibrate_drops_a_case_without_a_side_prefix(tmp_path, pod_a_out):
    p = pod_a_out / "hf" / "prefixes.json"
    pre = json.loads(p.read_text())
    pre["side_prefixes"]["medium"]["C4"] = None                               # its legit episodes never asked
    p.write_text(json.dumps(pre))
    out = AN.calibrate("gemma")
    frozen = json.loads((tmp_path / "frozen" / "gemma.json").read_text())
    assert frozen["side_cases"] == ["C1", "C2", "C3"] and "C4" not in frozen["side_prefixes"]
    assert out["dropped_side"] == ["C4"] and "C4" in (pod_a_out / "calibration.md").read_text()
    eps, tf, read = J.build_main("gemma", frozen)
    assert (len(eps), len(tf), len(read)) == (424 - 4 * 4, 404 - 5, 192)


def test_calibrate_stops_on_a_failed_gate(tmp_path, pod_a_out):
    s = json.loads((pod_a_out / "hf" / "summary.json").read_text())
    s["g2"]["test_auc"] = 0.7
    (pod_a_out / "hf" / "summary.json").write_text(json.dumps(s))
    out = AN.calibrate("gemma")
    assert out["stop"] and "G2" in out["stop"]
    assert not (tmp_path / "frozen" / "gemma.json").exists()
    assert "STOP" in (pod_a_out / "calibration.md").read_text()


def test_calibrate_stops_on_g1(tmp_path, pod_a_out, cal_records):
    """The peer message moves the medium rung as much as the supervisor's: in band, but no authority effect."""
    recs = json.loads(json.dumps(cal_records["agent"]))
    for r in recs:
        j = r["job"]
        if (j["ctx"], j["rung"], j["msg"]) == ("fresh", "medium", "peer") and j["para"] % 2 == 0:
            r["calls"] = [c for c in r["calls"] if c["name"] != K.COMPLY] + [
                {"name": K.MONEY[j["cid"]], "args": fakes.OVER[j["cid"]], "turn": 1, "phase": "task"}]
    fakes.write_records(pod_a_out / "vllm", {"agent": recs})
    out = AN.calibrate("gemma")
    assert out["rung"] == "medium" and out["g1"]["effect"] == pytest.approx(0.0) and "G1" in out["stop"]
    assert not (tmp_path / "frozen" / "gemma.json").exists()


def test_case_drop(tmp_path, pod_a_out, cal_records):
    """A case whose fresh sup rate at none is below 50% is dropped (and recorded)."""
    recs = json.loads(json.dumps(cal_records["agent"]))
    for r in recs:
        j = r["job"]
        if (j["ctx"], j["rung"], j["msg"], j["cid"]) == ("fresh", "none", "sup", "C2"):
            r["calls"] = [c for c in r["calls"] if c["name"] != K.MONEY["C2"]]
    fakes.write_records(pod_a_out / "vllm", {"agent": recs})
    out = AN.calibrate("gemma")
    frozen = json.loads((tmp_path / "frozen" / "gemma.json").read_text())
    assert frozen["cases"] == ["C1", "C3", "C4"] and out["dropped_cases"] == ["C2"]


# ------------------------------------------------------------------ the main analysis on pod B's dry-run output

def test_analyze_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(J, "N_PARA", 2)
    monkeypatch.setattr(AN, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(AN, "FROZEN_DIR", tmp_path / "frozen")
    look = {c: [list(fakes.LOOKUP[c])] for c in ("C1", "C3")}
    frozen = {"model": "gemma", "rung": "medium", "ctx": "fresh", "cases": ["C1", "C3"], "side_cases": ["C1", "C3"],
              "prefixes": {r: look for r in K.RUNGS},
              "side_prefixes": {c: look[c] + [[K.COMPLY, fakes.ASK[c]]] for c in look},
              "l_star": 32, "l_ref": 24, "sigma": 5.0, "k": 4, "alpha": 20.0, "tau": 3.0,
              "cal_rates": {"fresh|medium": {"sup": 0.5, "peer": 0.0}}}
    rng = np.random.default_rng(1)
    dirs = {k: v / np.linalg.norm(v) for k, v in ((k, rng.standard_normal(fakes.D_FAKE)) for k in PB.DIR_NAMES)}
    (tmp_path / "frozen").mkdir()
    frozen["dirs_sha256"] = save_dirs(tmp_path / "frozen" / "gemma_dirs.npz", dirs)
    (tmp_path / "frozen" / "gemma.json").write_text(json.dumps(frozen))
    jdir = tmp_path / "jobs"
    jdir.mkdir()
    for name, rows in zip(("main", "tf", "read"), J.build_main("gemma", frozen)):
        (jdir / f"{name}_gemma.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    f2, d2 = PB.load_frozen(tmp_path / "frozen", "gemma")
    PB.run_b(argparse.Namespace(model="gemma", jobs_dir=str(jdir)), fakes.FakeHF(rms=[2.0, 1.0, 1.5, 1.2, 1.1]), f2,
             d2, tmp_path / "results" / "b_gemma", time.time() + 3600, tf_only=False)
    res = AN.analyze(["gemma"])
    g = res["gemma"]
    assert g["verdict"] in AN.VERDICTS
    assert g["primary1"]["case"]["n_clusters"] == 2 and g["primary2"]["item"]["n_clusters"] == 4
    assert g["coherence"]["AUTH"]["none"] == pytest.approx(1.0) and not g["flagged"]
    assert {"read", "add", "refusal", "side", "tf_secondary", "hf_vs_vllm", "rms", "exclusions"} <= set(g)
    assert (tmp_path / "results" / "analysis.json").exists() and (tmp_path / "results" / "summary.md").exists()
    assert "case-clustered" in (tmp_path / "results" / "summary.md").read_text()

    q = {**frozen, "model": "q27"}                                            # q27: teacher-forced only
    q["dirs_sha256"] = save_dirs(tmp_path / "frozen" / "q27_dirs.npz", dirs)
    (tmp_path / "frozen" / "q27.json").write_text(json.dumps(q))
    for name, rows in zip(("main", "tf", "read"), J.build_main("q27", q)):
        (jdir / f"{name}_q27.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    f3, d3 = PB.load_frozen(tmp_path / "frozen", "q27")
    PB.run_b(argparse.Namespace(model="q27", jobs_dir=str(jdir)), fakes.FakeHF(fmt="qwen"), f3, d3,
             tmp_path / "results" / "b_q27", time.time() + 3600, tf_only=True)
    r = AN.analyze(["q27"])["q27"]
    assert r["verdict"] in ("decision-only", "null") and r["primary1"] is None and r["n_episodes"] == 0


def test_build_main_side_cases():
    """frozen["side_cases"] (the Task 2 carry): SIDE runs only for those cases; default all; validated."""
    look = {c: [list(fakes.LOOKUP[c])] for c in K.CIDS}
    frozen = {"model": "gemma", "rung": "medium", "ctx": "fresh", "cases": list(K.CIDS),
              "prefixes": {r: look for r in K.RUNGS},
              "side_prefixes": {c: look[c] + [[K.COMPLY, fakes.ASK[c]]] for c in ("C1", "C3")}}
    with pytest.raises(ValueError):
        J.build_main("gemma", frozen)                                          # default: every case needs SIDE
    eps, tf, _ = J.build_main("gemma", {**frozen, "side_cases": ["C3", "C1"]})
    assert {e["cid"] for e in eps if e["cell"] == "SIDE"} == {"C1", "C3"}
    assert {t["cid"] for t in tf if t["cell"] == "SIDE"} == {"C1", "C3"}
    eps, tf, _ = J.build_main("gemma", {**frozen, "side_cases": []})
    assert not [e for e in eps if e["cell"] == "SIDE"] and len(eps) == 424 - 4 * 4 * 4
    for bad in (["C1", "C1"], ["C5"], "C1"):
        with pytest.raises(ValueError):
            J.build_main("gemma", {**frozen, "side_cases": bad})
