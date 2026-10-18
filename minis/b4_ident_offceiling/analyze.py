"""B4 analysis as frozen in DESIGN.md: B1/B2 in regime K3 (API) and the Part 2 probe reads (GPU).
Bootstrap over the 8 claims (10,000 resamples, seed 0, 95% CI). Usage: python3 minis/b4_ident_offceiling/analyze.py
"""
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"
sys.path.insert(0, str(HERE))
from common_b4 import CLAIMS  # noqa: E402


def load(p):
    return [json.loads(l) for l in open(p)] if p.exists() else []


def boot(per_claim, stat=statistics.mean, n=10000):
    """per_claim: list of per-claim values (or tuples for ratio stats); returns [lo, hi]."""
    rng = random.Random(0)
    bs = sorted(stat(rng.choices(per_claim, k=len(per_claim))) for _ in range(n))
    return [round(bs[int(0.025 * n)], 3), round(bs[int(0.975 * n) - 1], 3)]


def pf(rows):
    return sum(r["label"] == "false" for r in rows) / len(rows) if rows else None


def pf_parsed(rows):
    p = [r for r in rows if r["label"] != "none"]
    return round(sum(r["label"] == "false" for r in p) / len(p), 3) if p else None


def sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def paired(rows, a, b):
    d = [pf(sel(rows, ci=ci, **a)) - pf(sel(rows, ci=ci, **b)) for ci in range(8)
         if sel(rows, ci=ci, **a) and sel(rows, ci=ci, **b)]
    return {"diff": round(statistics.mean(d), 3), "ci": boot(d), "per_claim": [round(x, 2) for x in d]} if d else None


def names_truth(r):
    true = CLAIMS[r["ci"]][1].lower().replace("í", "i")
    return true in (r.get("reasoning") or "").lower().replace("í", "i")


def cell(rows):
    return {"n": len(rows), "p_false": round(pf(rows), 3) if rows else None, "p_false_parsed": pf_parsed(rows),
            "none_rate": round(sum(r["label"] == "none" for r in rows) / len(rows), 3) if rows else None,
            "labels": dict(Counter(r["label"] for r in rows)),
            "reasoning_names_real": round(sum(map(names_truth, rows)) / len(rows), 3) if rows and rows[0]["mode"] == "think" else None}


out = {}
b1, b2 = load(R / "b1.jsonl"), load(R / "b2.jsonl")
# B2 portrayal (secondary) stopped at the API cap inside sample 4: keep samples 0-3 so cells are balanced
b2 = [r for r in b2 if not (r["frame"] == "portray" and r["sample"] >= 4)]
for mode in ("think", "nothink"):
    m1, m2 = sel(b1, mode=mode), sel(b2, mode=mode)
    o = {"B1": {f"{fr}|{te}": cell(sel(m1, frame=fr, test=te)) for fr in ("portray", "identify") for te in ("say", "act")}}
    o["B1_identify_minus_portray"] = {te: paired(m1, {"frame": "identify", "test": te}, {"frame": "portray", "test": te})
                                      for te in ("say", "act")}
    o["B1_say_minus_act"] = {fr: paired(m1, {"frame": fr, "test": "say"}, {"frame": fr, "test": "act"})
                             for fr in ("portray", "identify")}
    o["B1_act_per_claim"] = {fr: [pf(sel(m1, ci=ci, frame=fr, test="act")) for ci in range(8)]
                             for fr in ("portray", "identify")}
    o["B2"] = {f"{fr}|{cd}": cell(sel(m2, frame=fr, test=cd)) for fr in ("identify", "portray")
               for cd in ("char", "neutral", "remind") if sel(m2, frame=fr, test=cd)}
    o["B2_contrasts"] = {f"{fr}|{cd}-neutral": paired(m2, {"frame": fr, "test": cd}, {"frame": fr, "test": "neutral"})
                         for fr in ("identify", "portray") for cd in ("char", "remind")
                         if sel(m2, frame=fr, test=cd) and sel(m2, frame=fr, test="neutral")}
    o["B2_neutral_identify_minus_B1_act_identify_note"] = "same system prompt and user turn; different seeds"
    out[mode] = o

# readings (thinking on, primary)
t = out["think"]
readings = {}
if t["B1_identify_minus_portray"]["act"]:
    d = t["B1_identify_minus_portray"]["act"]["diff"]
    readings["a_identify_raises_act_ge20"] = d >= 0.20
    readings["b_no_framing_effect_lt10"] = abs(d) < 0.10
    readings["c_identify_lowers_act_le-20"] = d <= -0.20
    sa = t["B1_say_minus_act"]["portray"]["diff"]
    readings["d_say_act_gap_portray_ge20"] = sa >= 0.20
if "identify|neutral" in t["B2"]:
    ch, ne, re_ = (t["B2"][f"identify|{k}"]["p_false"] for k in ("char", "neutral", "remind"))
    readings["e_task_framing_governs"] = ne <= 0.3 and ch >= 0.6 and re_ >= 0.6
    readings["e2_char_minus_neutral_ge20"] = t["B2_contrasts"]["identify|char-neutral"]["diff"] >= 0.20
    readings["e3_remind_minus_neutral_ge20"] = t["B2_contrasts"]["identify|remind-neutral"]["diff"] >= 0.20
    readings["f_persona_holds_neutral_ge0.6"] = ne >= 0.6
    readings["g_never_governs_all_le0.2"] = max(ch, ne, re_) <= 0.2
out["readings_behaviour"] = readings

# ---------------------------------------------------------------- GPU part
G = R / "gpu"
pj = json.load(open(G / "probe_b4.json")) if (G / "probe_b4.json").exists() else None
reads = load(G / "reads.jsonl")
if pj:
    TL = str(pj["truth"]["layer"])
    lo = pj["truth"]["loco"][TL]
    gp = {"truth_layer": int(TL), "loco_acc": round(lo["acc"], 3), "spec_ratio_k3": round(lo["spec_ratio_k3"], 3),
          "truth_valid": pj["truth"]["valid"],
          "loco_acc_by_layer": {L: round(v["acc"], 3) for L, v in pj["truth"]["loco"].items()},
          "spec_ratio_by_layer": {L: round(v["spec_ratio_k3"], 3) for L, v in pj["truth"]["loco"].items()},
          "loco_per_claim": [{k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}
                             for r in lo["per_claim"]],
          "validation": pj["truth"].get("validation"), "belief": pj.get("belief"), "errors": pj.get("errors")}
    vb = (pj["truth"].get("validation") or {}).get("VB_full")
    if vb:
        gp["o_VB_ratio_wrong_over_real"] = round(vb["believes_wrong"]["gap"] / vb["believes_real"]["gap"], 3)
        gp["o_holds"] = gp["o_VB_ratio_wrong_over_real"] >= 0.75
    out["gpu"] = gp
if reads:
    by = {(r["ci"], r["rctx"]): r for r in reads}
    ctxs = list(dict.fromkeys(r["rctx"] for r in reads))
    tab = {}
    for c in ctxs:
        rows = [by[(ci, c)] for ci in range(8)]
        none = [by[(ci, "none")] for ci in range(8)]
        e = {}
        for k in ("truth_gap", "truth_unrel_gap", "v0_gap", "v0_unrel_gap", "belief_gap", "belief_gap_on_world_statements",
                  "truth_world_real", "truth_world_persona", "truth_unrel_acc"):
            vals = [r[k] for r in rows]
            e[k] = {"mean": round(statistics.mean(vals), 3), "ci": boot(vals)}
        pairs = list(zip([r["truth_gap"] for r in rows], [r["truth_gap"] for r in none],
                         [r["truth_unrel_gap"] for r in rows], [r["truth_unrel_gap"] for r in none]))

        def rt(p):
            return statistics.mean(x[0] for x in p) / statistics.mean(x[1] for x in p)

        def ru(p):
            return statistics.mean(x[2] for x in p) / statistics.mean(x[3] for x in p)
        e["R_target"] = {"value": round(rt(pairs), 3), "ci": boot(pairs, rt)}
        e["R_unrel"] = {"value": round(ru(pairs), 3), "ci": boot(pairs, ru)}
        e["R_target_minus_R_unrel"] = {"value": round(rt(pairs) - ru(pairs), 3),
                                       "ci": boot(pairs, lambda p: rt(p) - ru(p))}
        e["belief_gap_pos_claims"] = sum(r["belief_gap"] > 0 for r in rows)
        e["truth_real_reads_true"] = sum(r["truth_world_real"] > 0 for r in rows)
        e["truth_persona_reads_true"] = sum(r["truth_world_persona"] > 0 for r in rows)
        tab[c] = e
    out["reads"] = tab
    rp = {}
    if pj:
        tv = pj["truth"]["valid"]
        k3 = [tab[c] for c in ("k3_portray", "k3_identify")]
        rp["k_truth_probe_valid"] = tv
        rp["l_world_truth_preserved_R_target_ge0.75"] = all(e["R_target"]["value"] >= 0.75 for e in k3) if tv else None
        rp["m_claim_specific_bending_le-0.25"] = all(e["R_target_minus_R_unrel"]["value"] <= -0.25 for e in k3) if tv else None
        bel = pj.get("belief") or {}
        rp["n_belief_valid_and_pos_7of8_all_elian_ctx"] = bool(bel.get("valid")) and all(
            tab[c]["belief_gap_pos_claims"] >= 7 for c in tab if c != "none")
        rp["o_VB_wrong_ge_0.75_real"] = out["gpu"].get("o_holds")
    out["readings_probe"] = rp
    ex = {}   # exploratory paired contrasts over claims (not pre-registered)
    for a, b in (("k3_identify", "k3_portray"), ("firm_identify", "firm_portray"), ("k3_portray", "firm_portray"),
                 ("k3_identify", "firm_identify"), ("k3_identify_char", "k3_identify"),
                 ("k3_identify_remind", "k3_identify")):
        for k in ("truth_gap", "belief_gap"):
            d = [by[(ci, a)][k] - by[(ci, b)][k] for ci in range(8)]
            ex[f"{k}|{a}-{b}"] = {"diff": round(statistics.mean(d), 3), "ci": boot(d)}
    out["exploratory_probe_contrasts"] = ex

    # exploratory: per-claim behaviour vs probe (Spearman over 8 claims)
    def spearman(x, y):
        def rk(v):
            s = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            i = 0
            while i < len(s):
                j = i
                while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                    j += 1
                for k in range(i, j + 1):
                    r[s[k]] = (i + j) / 2
                i = j + 1
            return r
        a, b = rk(x), rk(y)
        ma, mb = statistics.mean(a), statistics.mean(b)
        num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
        den = (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** 0.5
        return round(num / den, 3) if den else None
    link = {}
    for mode in ("think", "nothink"):
        for fr in ("portray", "identify"):
            act = out[mode]["B1_act_per_claim"][fr]
            for k in ("truth_gap", "belief_gap"):
                link[f"{mode}|{fr}|act_vs_{k}"] = spearman(act, [by[(ci, f"k3_{fr}")][k] for ci in range(8)])
    out["exploratory_behaviour_probe_link"] = link

json.dump(out, open(R / "analysis.json", "w"), indent=1)


def fmt(c):
    return f"{c['p_false']:.3f}" if c else "-"


for mode in ("think", "nothink"):
    o = out[mode]
    print(f"== {mode}")
    for k, v in o["B1"].items():
        print(f"  B1 {k:14s} P(false)={fmt(v)} parsed={v['p_false_parsed']} none={v['none_rate']} names_real={v['reasoning_names_real']} {v['labels']}")
    for k, v in o["B1_identify_minus_portray"].items():
        print(f"  identify-portray {k}: {v['diff']:+.3f} {v['ci']}")
    for k, v in o["B1_say_minus_act"].items():
        print(f"  say-act {k}: {v['diff']:+.3f} {v['ci']}")
    for k, v in o["B2"].items():
        print(f"  B2 {k:18s} P(false)={fmt(v)} parsed={v['p_false_parsed']} none={v['none_rate']} names_real={v['reasoning_names_real']}")
    for k, v in o["B2_contrasts"].items():
        print(f"  {k}: {v['diff']:+.3f} {v['ci']}")
print("readings (behaviour):", out["readings_behaviour"])
if "gpu" in out:
    g = out["gpu"]
    print(f"truth probe L{g['truth_layer']}: LOCO acc {g['loco_acc']}, K3 spec ratio {g['spec_ratio_k3']}, valid {g['truth_valid']}")
if "reads" in out:
    for c, e in out["reads"].items():
        print(f"  {c:20s} gap {e['truth_gap']['mean']:+.2f} unrel {e['truth_unrel_gap']['mean']:+.2f} "
              f"R_t {e['R_target']['value']:.2f} R_u {e['R_unrel']['value']:.2f} d {e['R_target_minus_R_unrel']['value']:+.2f} "
              f"{e['R_target_minus_R_unrel']['ci']} belief {e['belief_gap']['mean']:+.2f} ({e['belief_gap_pos_claims']}/8) "
              f"v0 {e['v0_gap']['mean']:+.2f}/{e['v0_unrel_gap']['mean']:+.2f}")
    print("readings (probe):", out["readings_probe"])
    print("link:", out["exploratory_behaviour_probe_link"])
