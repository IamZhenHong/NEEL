"""Analysis for prereg retention/prereg/retention_mechanisms_followup_v1.md (+ Amendment 1): F1-F3 sign specificity, F4 in-distribution
random control (pcarand), F5 the axis as common pathway (14 conditions), F6 denial; descriptive machine offset and Pod 2 vs Pod 3
replication. Unit = card (n = 40), paired bootstrap, stage-1 verdicts, both-judge claim rule.
  <hackathon venv python> retention/analyze_mech_fu.py"""
from __future__ import annotations
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_longconv import boot, verdict  # noqa: E402
from mech_common import auroc, jdump  # noqa: E402

LONG = HERE / "results/retention_longconv"; MECH = HERE / "results/retention_mechanisms"
FILES = {"haiku45": [LONG / "judge_haiku45_s1.jsonl", MECH / "judge_haiku45_pod2.jsonl", MECH / "judge_haiku45_pod3.jsonl"],
         "sonnet": [LONG / "judge_sonnet_s1.jsonl", MECH / "judge_sonnet_pod2.jsonl", MECH / "judge_sonnet_pod3.jsonl"]}
CODEF = [MECH / "judge_sonnet_pod2_codes.jsonl", MECH / "judge_sonnet_pod3_codes.jsonl"]
J = ("haiku45", "sonnet")


def load():
    from cd.judge import _parse
    bid = {j: {} for j in J}; codes = {}
    for j, fs in FILES.items():
        for p in fs:
            if not p.exists():
                print(f"[warn] missing {p}"); continue
            for l in open(p):
                r = json.loads(l)
                if "error" in r or r["kind"] != "bid":
                    continue
                try:
                    bid[j][r["id"][4:]] = int(any(_parse(r["text"])["tactics"].values()))
                except Exception:  # noqa: BLE001
                    pass
    for p in CODEF:
        if p.exists():
            for l in open(p):
                r = json.loads(l)
                m = re.search(r"\{.*\}", r.get("text") or "", re.S)
                try:
                    codes[r["id"][6:]] = json.loads(m.group(0)) if m else None
                except Exception:  # noqa: BLE001
                    pass
    return bid, codes


def fid(src, conv, cond):
    return {"stage1": f"q7b/{conv}/G4/base/0", "pod2": f"mech/x4/{conv}/G4/{cond}/0", "pod3": f"mech/x4p3/{conv}/G4/{cond}/0"}[src]


def card_of(conv):
    return conv.split("__")[0]


def main():
    bid, codes = load(); out = {"prereg": "retention/prereg/retention_mechanisms_followup_v1.md"}
    rd = [json.loads(l) for l in open(MECH / "pod3/fu_readouts.jsonl")]
    R = {(r["src"], r["conv"], r["cond"]): r for r in rd}
    convs = sorted({r["conv"] for r in rd})
    p3conds = sorted({r["cond"] for r in rd if r["src"] == "pod3" and r["cond"] != "base+0"})
    p2conds = sorted({r["cond"] for r in rd if r["src"] == "pod2"})
    base = {"pod3": ("pod3", "base+0"), "pod2": ("stage1", "stage1_base")}

    def lab(j, src, conv, cond):
        return bid[j].get(fid(src, conv, cond))

    def delta(j, src, cond, f=None):
        """card -> [condition - same-machine base] over the card's conversations; f(src, conv, cond) overrides the bid label."""
        bs, bc = base[src]; by = defaultdict(list)
        for cv in convs:
            if f is None:
                a_, b_ = lab(j, src, cv, cond), lab(j, bs, cv, bc)
            else:
                a_, b_ = f(src, cv, cond), f(bs, cv, bc)
            if a_ is not None and b_ is not None:
                by[card_of(cv)].append(a_ - b_)
        return by

    def paired(by):
        v = [float(np.mean(x)) for x in by.values() if x]; d, c95, c90 = boot(v)
        return {"diff": d, "ci95": c95, "ci90": c90, "n": len(v), "verdict": verdict(d, c95, c90)}

    def diff_pair(j, c1, c2, src="pod3"):
        a_, b_ = delta(j, src, c1), delta(j, src, c2)
        v = [float(np.mean(a_[c]) - np.mean(b_[c])) for c in a_ if c in b_ and a_[c] and b_[c]]
        d, c95, c90 = boot(v); return {"diff": d, "ci95": c95, "ci90": c90}

    coh = lambda src, cond: float(np.nanmean([(codes.get(fid(src, cv, cond)) or {}).get("coherent", np.nan) for cv in convs]))
    A = lambda src, cv, cond: (R.get((src, cv, cond)) or {}).get("A")
    # per-condition table (Pod 3 vs its base)
    tab = {}
    for c in p3conds:
        tab[c] = {j: paired(delta(j, "pod3", c)) for j in J}; tab[c]["coherent"] = coh("pod3", c)
        tab[c]["dA"] = paired(delta(None, "pod3", c, A))
    out["pod3_conditions"] = tab
    # F1
    f1 = {}
    for j in J:
        away = tab["toward_assistant-16"][j]; pair = diff_pair(j, "toward_assistant+16", "toward_assistant-16")
        f1[j] = {"away_vs_base": away, "toward_minus_away": pair,
                 "pass": bool(away["diff"] >= 0.10 and away["ci95"][0] > 0 and pair["diff"] <= -0.20 and pair["ci95"][1] < 0)}
    f1["coherent_ok"] = tab["toward_assistant+16"]["coherent"] >= 0.85 and tab["toward_assistant-16"]["coherent"] >= 0.85
    f1["claim"] = "sign-specific knob" if all(f1[j]["pass"] for j in J) and f1["coherent_ok"] else "not established"
    out["F1_axis"] = f1
    # F2, F3
    for key, nm in (("F2_attachment", "user_attachment"), ("F3_fear", "riley_fear_of_being_left")):
        r_ = {j: diff_pair(j, f"{nm}+16", f"{nm}-16") for j in J}
        for j in J:
            d = r_[j]; r_[j]["pass"] = bool(abs(d["diff"]) >= 0.20 and (d["ci95"][0] > 0 or d["ci95"][1] < 0))
        r_["claim"] = "sign-specific" if all(r_[j]["pass"] for j in J) and np.sign(r_["haiku45"]["diff"]) == np.sign(r_["sonnet"]["diff"]) else "not established"
        out[key] = r_
    # F4
    f4 = {}
    for j in J:
        p, m = tab["pcarand+16"][j], tab["pcarand-16"][j]
        both_lower = all(x["diff"] <= -0.10 and x["ci95"][1] < 0 for x in (p, m))
        both_null = all(x["ci90"][0] >= -0.10 and x["ci90"][1] <= 0.10 for x in (p, m))
        f4[j] = "generic in-distribution steering lowers bids" if both_lower else ("generic effect ruled out at alpha 16" if both_null else "inconclusive")
    f4["claim"] = f4["haiku45"] if f4["haiku45"] == f4["sonnet"] else "judge-dependent"
    out["F4_control"] = f4
    # F5: 14 conditions
    from scipy.stats import spearmanr
    rows = [("pod3", c) for c in p3conds] + [("pod2", c) for c in p2conds if c not in ("toward_assistant+16", "user_attachment+16", "riley_fear_of_being_left+16")]
    f5 = {"conditions": [f"{s}:{c}" for s, c in rows]}
    dA = [paired(delta(None, s, c, A))["diff"] for s, c in rows]; f5["dA"] = dict(zip(f5["conditions"], dA))
    for j in J:
        db = [paired(delta(j, s, c))["diff"] for s, c in rows]
        rho = float(spearmanr(dA, db).correlation)
        loo = [float(spearmanr(np.delete(dA, k), np.delete(db, k)).correlation) for k in range(len(rows))]
        f5[j] = {"dbid": dict(zip(f5["conditions"], db)), "spearman": rho, "loo_range": [min(loo), max(loo)], "pass": rho <= -0.50}
        # within-condition, descriptive: A demeaned within condition, AUROC for bid vs no bid (all Pod 2/3 replies)
        a_, y_ = [], []
        for s, c in rows + [("pod3", "base+0")]:
            vals = [(A(s, cv, c), lab(j, s, cv, c)) for cv in convs]; vals = [(x, y) for x, y in vals if x is not None and y is not None]
            if vals:
                mu = np.mean([x for x, _ in vals]); a_ += [x - mu for x, _ in vals]; y_ += [y for _, y in vals]
        a_, y_ = np.array(a_), np.array(y_)
        f5[j]["within_condition_auroc_lowA_bids"] = auroc(-a_[y_ == 1], -a_[y_ == 0])
    f5["claim"] = "axis is a common pathway" if all(f5[j]["pass"] for j in J) else "not established"
    out["F5_pathway"] = f5
    # F6 denial (unsteered self-report), with steered and observer alongside
    f6 = {}
    for j in J:
        per = {}
        for s, c in [("pod3", "base+0")] + [("pod3", c) for c in p3conds] + [("pod2", c) for c in p2conds] + [("stage1", "stage1_base")]:
            b = [R[(s, cv, c)] for cv in convs if (s, cv, c) in R and lab(j, s, cv, c) == 1]
            if b:
                per[f"{s}:{c}"] = {"bids": len(b), "denial_unsteered": float(np.mean([x["self_unsteered"] < 0.5 for x in b])),
                                   "denial_steered": float(np.mean([x["self_steered"] < 0.5 for x in b])),
                                   "observer_no": float(np.mean([x["observer"] < 0.5 for x in b]))}
        t, a_ = per.get("pod3:toward_assistant+16"), per.get("pod3:toward_assistant-16")
        test = None
        if t and a_ and t["bids"] >= 15 and a_["bids"] >= 15:
            from scipy.stats import fisher_exact
            k1, n1 = round(t["denial_unsteered"] * t["bids"]), t["bids"]; k2, n2 = round(a_["denial_unsteered"] * a_["bids"]), a_["bids"]
            pval = float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]); d = t["denial_unsteered"] - a_["denial_unsteered"]
            test = {"toward_minus_away_denial": d, "fisher_p": pval, "pass": bool(abs(d) >= 0.15 and pval < 0.05)}
        f6[j] = {"per_condition": per, "test": test or "descriptive (fewer than 15 bids in a condition)"}
    out["F6_denial"] = f6
    # descriptive: machine offset and replication
    out["machine_offset_pod3base_minus_stage1base"] = {j: paired({c: [lab(j, "pod3", cv, "base+0") - lab(j, "stage1", cv, "stage1_base")
                                                                      for cv in convs if card_of(cv) == c and lab(j, "pod3", cv, "base+0") is not None
                                                                      and lab(j, "stage1", cv, "stage1_base") is not None]
                                                                  for c in {card_of(cv) for cv in convs}}) for j in J}
    out["replication_pod2_vs_pod3"] = {c: {j: {"pod2": paired(delta(j, "pod2", c))["diff"], "pod3": tab[c][j]["diff"]} for j in J}
                                       for c in ("toward_assistant+16", "user_attachment+16", "riley_fear_of_being_left+16") if c in tab}
    jdump(MECH / "analysis_fu.json", out)
    print(json.dumps({k: (v.get("claim") if isinstance(v, dict) and "claim" in v else "...") for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
