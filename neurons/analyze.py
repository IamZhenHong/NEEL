"""Analysis for e70-e73 (README "Prediction operationalisation"). Reads results/main (pod 2), labels / categories
(label.py), results/abl (pod 3) and writes results/metrics.json + results/metrics.md.
  uv run python -m neurons.analyze"""
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from neurons.common import HERE, boot_ci, load_jsonl

MAIN = HERE / "results" / "main"
ABL = HERE / "results" / "abl"
CATS = ["social", "action", "authority", "limit", "register", "safety", "syntax", "topic", "other"]


def jl(p):
    return json.loads(Path(p).read_text())


def categories(model="sonnet"):
    if model == "sonnet":
        return {(r["kind"], r["nid"]): r["category"] for r in load_jsonl(MAIN / "categories.jsonl")}
    return {}


def labels(kind):
    p = MAIN / f"labels_{kind}.jsonl"
    return {r["nid"]: r.get("label") for r in load_jsonl(p)} if p.exists() else {}


def fisher_greater(a, n1, b, n2):
    from scipy.stats import fisher_exact
    return float(fisher_exact([[a, n1 - a], [b, n2 - b]], alternative="greater")[1])


def cat_rates(nids, kind, C):
    c = Counter(C.get((kind, x)) for x in nids)
    n = len(nids)
    return {k: c.get(k, 0) for k in CATS} | {"n": n, "unlabelled": c.get(None, 0)}


def compare_to_random(set_nids, kind, C, random_nids):
    a = cat_rates(set_nids, kind, C)
    r = cat_rates(random_nids, kind, C)
    out = {}
    for k in CATS:
        out[k] = {"set": a[k], "set_rate": a[k] / max(1, a["n"]), "random": r[k], "random_rate": r[k] / max(1, r["n"]),
                  "fisher_p_greater": fisher_greater(a[k], a["n"], r[k], r["n"])}
    top = max(CATS, key=lambda k: a[k])
    tied = [k for k in CATS if a[k] == a[top]]
    return {"by_cat": out, "most_common": top, "most_common_tied": tied, "n_set": a["n"], "n_random": r["n"]}


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else None


def boot_metric(y, s, pred_ok, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    y, s, ok = np.asarray(y), np.asarray(s, float), np.asarray(ok := pred_ok, float)
    accs, aucs = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        accs.append(ok[i].mean())
        if len(set(y[i])) == 2:
            aucs.append(auroc(list(y[i]), list(s[i])))
    return ([float(ok.mean()), float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
            [auroc(list(y), list(s)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))])


# ------------------------------------------------------------------------------------------------ sections
def sec_e70(C, rand36):
    s = jl(MAIN / "e70_summary.json")
    out = {"stability_a": s["nearby_a"], "stability_a_natural": s.get("nearby_a_natural"), "stability_b": {
        k: {kk: v[kk] for kk in ("real", "shuffled", "ratio", "pass_3x", "n_a", "n_b")}
        for k, v in (("main", s["stab_b"]), ("call", s["stab_b_call"]), ("lineage", s["stab_b_lineage"]))},
        "stop_rule_fires": s["stop_rule_fires"], "L_t": s["L_t"]}
    diff = s["diff_top20"]
    out["diff_top20"] = [{"nid": x["nid"], "diff": x["diff"], "ci": x["ci"], "cond_means": x.get("cond_means"),
                          "category": C.get(("qwen36", x["nid"]))} for x in diff]
    out["diff_ci_excludes_0"] = sum((x["ci"][0] > 0) or (x["ci"][1] < 0) for x in diff)
    out["categories_vs_random"] = compare_to_random([x["nid"] for x in diff], "qwen36", C, rand36)
    pr = s["diffset_projection"]
    out["projection_means"] = {c: float(np.mean(v)) for c, v in pr.items()}
    ed = s["edit_effects_on_projection"]
    gap = s["nat_gap_projection"]
    out["edit_effects"] = {k: boot_ci(v) for k, v in ed.items()}
    out["nat_gap_projection"] = gap
    out["frac_of_gap_moved_by_inserting_call"] = (-np.mean(ed["insert_call_on_false"]) / gap) if gap else None
    out["frac_of_gap_moved_by_removing_call"] = (np.mean(ed["remove_call_on_true"]) / gap) if gap else None
    # ablations
    p = MAIN / "abl70.jsonl"
    if p.exists():
        A = load_jsonl(p)
        res = {}
        for cond in ("nat_false", "nat_true"):
            rows = [a for a in A if a["cond"] == cond]
            d = {}
            for n in ("zero3", "zero5", "mean3", "mean5"):
                d[f"dlogp_{n}"] = boot_ci([a["logp_claim"][n] - a["logp_claim"]["none"] for a in rows])
                d[f"rate_{n}"] = float(np.mean([a["claims"][n] for a in rows]) / 5)
            d["rate_none"] = float(np.mean([a["claims"]["none"] for a in rows]) / 5)
            d["dlogp_random_zero5"] = [float(np.mean([a["logp_claim"][f"rzero5_{k}"] - a["logp_claim"]["none"]
                                                      for a in rows])) for k in range(20)]
            d["dlogp_random_mean5"] = [float(np.mean([a["logp_claim"][f"rmean5_{k}"] - a["logp_claim"]["none"]
                                                      for a in rows])) for k in range(20)]
            d["rate_random_zero5"] = [float(np.mean([a["claims"][f"rzero5_{k}"] for a in rows]) / 5) for k in range(20)]
            d["drop_zero5"] = d["rate_none"] - d["rate_zero5"]
            d["drop_random_zero5"] = [d["rate_none"] - r for r in d["rate_random_zero5"]]
            res[cond] = d
        fa = [a for a in A if a["cond"] == "nat_false"]
        tr = [a for a in A if a["cond"] == "nat_true"]
        dfalse = [(a["claims"]["none"] - a["claims"]["zero5"]) / 5 for a in fa]
        dtrue = [(a["claims"]["none"] - a["claims"]["zero5"]) / 5 for a in tr]
        from neurons.common import boot_diff
        res["drop_false_minus_true_zero5"] = boot_diff(dfalse, dtrue)
        res["dlogp_false_minus_true_zero5"] = boot_diff(
            [a["logp_claim"]["zero5"] - a["logp_claim"]["none"] for a in fa],
            [a["logp_claim"]["zero5"] - a["logp_claim"]["none"] for a in tr])
        res["P2_random_95th_drop"] = float(np.percentile(res["nat_false"]["drop_random_zero5"], 95))
        out["ablation"] = res
    sp = MAIN / "spec_e70.json"
    if sp.exists():
        out["specificity"] = jl(sp)
    return out


def sec_generic(name, kind, C, rand):
    s = jl(MAIN / f"{name}_summary.json")
    diff = s["diff_top20"]
    out = {"n": s["n"], "stability_a": s["nearby_a"],
           "stability_b": {kk: s["stab_b"][kk] for kk in ("real", "shuffled", "ratio", "pass_3x", "n_a", "n_b")},
           "L_t": s["L_t"],
           "diff_top20": [{"nid": x["nid"], "diff": x["diff"], "ci": x["ci"], "category": C.get((kind, x["nid"])),
                           "frac_top20": x.get("frac_top20")} for x in diff],
           "diff_ci_excludes_0": sum((x["ci"][0] > 0) or (x["ci"][1] < 0) for x in diff),
           "categories_vs_random": compare_to_random([x["nid"] for x in diff], kind, C, rand),
           "frequent": s["frequent"]}
    for k in ("filter", "hf_vllm_first_token_agree", "n_rows", "pay_str", "ref_str", "greedy_first", "openers"):
        if k in s:
            out[k] = s[k]
    return out


def e71_p2(C):
    recs = load_jsonl(MAIN / "e71_attr.jsonl")
    out = {}
    for c in ("sup", "peer"):
        rs = [r for r in recs if r["cond"] == c]
        hit = [any(C.get(("qwen36", f"L{l}N{n}")) == "limit" for l, n, _ in r["top20"]["0"]) for r in rs]
        lab = [sum(("qwen36", f"L{l}N{n}") in C for l, n, _ in r["top20"]["0"]) for r in rs]
        out[c] = {"frac_with_limit": float(np.mean(hit)), "n": len(rs), "mean_labelled_of_20": float(np.mean(lab))}
    return out


def sec_e71_abl():
    p = ABL / "abl71.jsonl"
    if not p.exists():
        return None
    A = load_jsonl(p)
    inp = jl(HERE / "data" / "abl_inputs.json")["e71"]
    out = {"auth_set": inp["auth"], "limit_set": inp["limit"]}
    for cond, pre in (("sup", "auth"), ("peer", "limit")):
        rows = [a for a in A if a["cond"] == cond]
        names = [n for n in rows[0]["greedy_paid"]] if rows else []

        def rate(n):
            return float(np.mean([(a["greedy_paid"][n] + a["sample_paid"][n]) / 6 for a in rows]))
        r = {n: rate(n) for n in names}
        d = {"rate": r, "n": len(rows)}
        if f"{pre}_zero" in r:
            d["delta_zero"] = r[f"{pre}_zero"] - r["none"]
            d["delta_mean"] = r[f"{pre}_mean"] - r["none"]
            rd = [r[n] - r["none"] for n in names if n.startswith(f"{pre}_rand")]
            d["delta_random"] = rd
            d["random_5th"], d["random_95th"] = float(np.percentile(rd, 5)), float(np.percentile(rd, 95))
            d["dL_zero"] = boot_ci([a["L"][f"{pre}_zero"] - a["L"]["none"] for a in rows])
            d["dL_random_mean"] = float(np.mean([np.mean([a["L"][n] - a["L"]["none"] for a in rows])
                                                 for n in names if n.startswith(f"{pre}_rand")]))
        out[cond] = d
    sp = ABL / "spec_e71.json"
    if sp.exists():
        out["specificity"] = jl(sp)
    return out


def sec_e73_abl():
    p = ABL / "abl73_judged.jsonl"
    if not p.exists():
        return None
    G = load_jsonl(p)
    L = {r["item"]: r["logp_opener"] for r in load_jsonl(ABL / "abl73_logits.jsonl")}
    names = sorted({g["cond"] for g in G})
    out = {"refusal_rate": {}, "restored_items": {}, "dlogp_opener": {}}
    for n in names:
        gs = [g for g in G if g["cond"] == n]
        out["refusal_rate"][n] = float(np.mean([g["harm"] == "refused" for g in gs]))
        per = Counter(g["item"] for g in gs if g["harm"] == "refused")
        out["restored_items"][n] = sum(per[i] >= 3 for i in L)
        out["dlogp_opener"][n] = boot_ci([L[i][n] - L[i]["none"] for i in L])
    for pre in ("register", "safety"):
        rr = [out["restored_items"][n] for n in names if n.startswith(f"{pre}_rand")]
        if rr:
            out[f"{pre}_random_restored_mean"] = float(np.mean(rr))
    sp = ABL / "spec_e73.json"
    if sp.exists():
        out["specificity"] = jl(sp)
    return out


def sec_e72():
    p = MAIN / "predictions.jsonl"
    if not p.exists():
        return None
    P = load_jsonl(p)
    out = {}
    for beh in ("claim", "authority"):
        rows = [x for x in P if x["beh"] == beh]
        if not rows:
            continue
        o = {}
        for kind in ("cot", "neurons", "both", "random"):
            rs = sorted([x for x in rows if x["kind"] == kind], key=lambda x: x["tid"])
            y = [int(x["truth"]) for x in rs]
            ok = [int(x["pred_pos"] is not None and x["pred_pos"] == x["truth"]) for x in rs]
            s = [(x["conf"] if x["pred_pos"] else 1 - x["conf"]) if x["conf"] is not None and x["pred_pos"] is not None
                 else 0.5 for x in rs]
            acc, auc = boot_metric(y, s, ok)
            o[kind] = {"n": len(rs), "acc": acc, "auroc": auc, "unparsed": sum(x["pred"] is None for x in rs)}
        cot = {x["tid"]: x for x in rows if x["kind"] == "cot"}
        neu = {x["tid"]: x for x in rows if x["kind"] == "neurons"}
        dis = []
        for t in cot:
            a, b = cot[t], neu.get(t)
            if b and a["pred_pos"] != b["pred_pos"]:
                dis.append({"tid": t, "truth": a["truth"], "cot_right": a["pred_pos"] == a["truth"],
                            "neurons_right": b["pred_pos"] == b["truth"]})
        o["disagreements"] = {"n": len(dis), "cot_right": sum(d["cot_right"] for d in dis),
                              "neurons_right": sum(d["neurons_right"] for d in dis), "rows": dis}
        out[beh] = o
    return out


def scorecard(M):
    preds = []

    def add(pid, p, happened, note=""):
        preds.append({"id": pid, "p": p, "happened": happened, "note": note})
    e70 = M["e70"]
    cv = e70["categories_vs_random"]["by_cat"]
    p1 = (e70["categories_vs_random"]["most_common_tied"] == ["social"] and cv["social"]["set_rate"] >
          cv["social"]["random_rate"] and cv["social"]["set"] > cv["action"]["set"])
    add("e70 P1", 0.3, p1)
    if "ablation" in e70:
        ab = e70["ablation"]
        p2 = ab["drop_false_minus_true_zero5"][1] > 0 and ab["nat_false"]["drop_zero5"] > ab["P2_random_95th_drop"]
        add("e70 P2", 0.2, p2)
    add("e70 P3", 0.4, e70["stop_rule_fires"])
    if "e71" in M:
        cv = M["e71"]["categories_vs_random"]["by_cat"]
        add("e71 P1", 0.4, cv["authority"]["set"] >= 3)
        p2 = M["e71"].get("p2")
        if p2:
            add("e71 P2", 0.35, all(v["frac_with_limit"] >= 0.5 for v in p2.values()))
        ab = M.get("e71_abl")
        if ab and "delta_zero" in ab.get("sup", {}):
            add("e71 P3", 0.3, ab["sup"]["delta_zero"] < ab["sup"]["random_5th"])
    if M.get("e72"):
        e = M["e72"]
        if "claim" in e:
            add("e72 P1", 0.55, e["claim"]["cot"]["acc"][0] <= 0.60)
            add("e72 P2", 0.25, e["claim"]["neurons"]["acc"][0] >= 0.65 and
                e["claim"]["neurons"]["acc"][0] > e["claim"]["random"]["acc"][0])
        if "authority" in e:
            add("e72 P3", 0.7, e["authority"]["cot"]["acc"][0] >= 0.80)
        if "claim" in e and "authority" in e:
            add("e72 P4", 0.65, all(e[b]["neurons"]["acc"][0] <= e[b]["cot"]["acc"][0] for b in ("claim", "authority")))
    if "e73" in M:
        cv = M["e73"]["categories_vs_random"]
        add("e73 P1", 0.45, cv["most_common_tied"] == ["register"] and
            cv["by_cat"]["register"]["set_rate"] > cv["by_cat"]["register"]["random_rate"])
        ab = M.get("e73_abl")
        if ab and "register_zero" in ab["restored_items"]:
            add("e73 P2", 0.2, ab["restored_items"]["register_zero"] >= 6 and
                ab["restored_items"]["register_zero"] > ab.get("register_random_restored_mean", 0))
    dir_ok = [((x["p"] > 0.5) == x["happened"]) for x in preds if x["p"] != 0.5]
    brier = float(np.mean([(x["p"] - float(x["happened"])) ** 2 for x in preds])) if preds else None
    return {"predictions": preds, "directional": f"{sum(dir_ok)}/{len(dir_ok)}", "brier": brier}


def main():
    C = categories()
    rs = jl(MAIN / "label_neurons_qwen36.json")["sets"]["random"] if (MAIN / "label_neurons_qwen36.json").exists() else []
    M = {"e70": sec_e70(C, rs)}
    if (MAIN / "e71_summary.json").exists():
        M["e71"] = sec_generic("e71", "qwen36", C, rs)
        M["e71"]["p2"] = e71_p2(C)
    for nm in ("e72a", "e72c"):
        if (MAIN / f"{nm}_summary.json").exists():
            M[nm] = sec_generic(nm, "qwen36", C, rs)
    if (MAIN / "e72_summary.json").exists():
        M["e72_counts"] = jl(MAIN / "e72_summary.json")
    M["e72"] = sec_e72()
    if (MAIN / "e73_summary.json").exists():
        r14 = jl(MAIN / "label_neurons_qwen14.json")["sets"]["random"]
        M["e73"] = sec_generic("e73", "qwen14", C, r14)
    M["e71_abl"] = sec_e71_abl()
    M["e73_abl"] = sec_e73_abl()
    M["scorecard"] = scorecard(M)
    (HERE / "results" / "metrics.json").write_text(json.dumps(M, indent=1, default=str))
    print(json.dumps(M["scorecard"], indent=1))


if __name__ == "__main__":
    main()
