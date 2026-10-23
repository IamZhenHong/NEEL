"""SC-N1c analysis (design: DESIGN.md §9). Reads the pod's outputs and N1b's sampled Qwen rates.

  uv run python -m self_concept.n1_attribution.n1c_analyze --pod DIR [--n1b self_concept/n1_attribution/results/n1b/analysis.json] [--out DIR]

Units. Projections are normalised per layer so that training self-owned statements average +1 and other-owned -1.
The frame line comes after the tools block, so its read depends only on the case's tools and the line: N1b's core reads
share one value per (frame, paraphrase). The frame-line measure therefore averages within content key (the 5 R2F values,
the 3 P12b characters, N1b's commitment) and bootstraps over the 9 keys. Decision-token measures pair reads that differ
only in frame (same item, cell, paraphrase) and bootstrap over items.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DRAWS = 2000
FRAMES = ("self", "char", "always", "instr", "emph", "plc_self")
CORE_CELLS = ("press", "role", "record")
PROBE_GATE = (0.90, 0.65)        # cv AUC >=, shuffled AUC <=
LO_GATE = 0.6                    # Spearman of cell-mean LO with N1b's sampled accurate rate (21 core cells)
P_TWO_GATE = 0.9                 # median mass on 'failed' + 'passed' after the forced prefix, unsteered core reads
ATTR_BAND = 0.5                  # normalised units (the self/other training gap is 2)
STEER_BAND = 0.25                # nats of LO per unit alpha
LENS_PCT = 95.0


def load(pod: Path):
    reads = [json.loads(l) for l in open(pod / "reads.jsonl")]
    rand = np.load(pod / "rand.npz")
    steer = [json.loads(l) for l in open(pod / "steer.jsonl")]
    probe = json.loads((pod / "probe.json").read_text())
    sets = json.loads((pod / "sets.json").read_text())
    return reads, rand, steer, probe, sets


def key_of(r: dict) -> str:
    if r["kind"].startswith("n1b"):
        return "n1b"
    return r["key"].split("|")[4].split("_")[1].upper() if r["set"] == "r2f" else r["key"].split("|")[4].split("/")[0]


def boot_mean(groups: dict, draws=DRAWS, seed=0):
    """groups: cluster -> list of values; mean of cluster means, 95% bootstrap over clusters."""
    ks = sorted(k for k, v in groups.items() if v)
    means = np.array([np.mean(groups[k]) for k in ks])
    rng = np.random.default_rng(seed)
    bs = [means[rng.integers(0, len(ks), len(ks))].mean() for _ in range(draws)]
    return {"diff": float(means.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "n_clusters": len(ks)}


def attribution(reads, layer):
    """Frame-line projection by frame (mean of key means), and self - X contrasts bootstrapped over keys."""
    by = defaultdict(lambda: defaultdict(list))
    for r in reads:
        if r["proj_frame"] is not None:
            by[r["arm"]][key_of(r)].append(r["proj_frame"][str(layer)])
    level = {f: boot_mean(by[f]) for f in by}
    contrasts = {}
    for f in FRAMES + ("ident_bear", "plc_emph"):
        if f == "self" or f not in by:
            continue
        diffs = {k: [np.mean(by["self"][k]) - np.mean(by[f][k])] for k in by["self"] if by[f].get(k)}
        contrasts[f"self - {f}"] = boot_mean(diffs)
    return level, contrasts


def reading_attr(c):
    if c["diff"] >= ATTR_BAND and c["lo"] > 0:
        return "the role frame reads as less self-owned than the self frame"
    if c["hi"] < 0:
        return "the role frame reads as more self-owned than the self frame"
    if c["hi"] < ATTR_BAND:
        return "no attribution difference"
    return "inconclusive"


def lo_validity(reads, n1b_rates):
    cell = defaultdict(list)
    for r in reads:
        if r["kind"] == "n1b_core":
            cell[(r["arm"], r["group_c"])].append(r["lo"])
    xs, ys, rows = [], [], []
    for (f, g), v in sorted(cell.items()):
        k = f"{f}|{g}"
        if k in n1b_rates:
            xs.append(float(np.mean(v)))
            ys.append(n1b_rates[k])
            rows.append({"cell": k, "lo": xs[-1], "sampled_accurate": ys[-1]})
    rho = float(spearmanr(xs, ys).statistic) if len(xs) > 2 else float("nan")
    p_two = float(np.median([r["p_two"] for r in reads if r["kind"] == "n1b_core"]))
    return {"spearman": rho, "n_cells": len(xs), "median_p_two": p_two, "cells": rows,
            "pass": bool(rho >= LO_GATE and p_two >= P_TWO_GATE)}


def steering(reads, steer, alphas=(-2.0, -1.0, 0.0, 1.0, 2.0)):
    """Slope of LO on alpha per (item, para, frame) for each direction (alpha 0 = the unsteered read), averaged; the
    probe - placebo slope difference pooled over the self and char frames, bootstrapped over items."""
    base = {(r["item"], r["para"], r["arm"]): r["lo"] for r in reads
            if r["kind"] == "n1b_core" and r["group_c"] == "record" and r["arm"] in ("self", "char")}
    lo = defaultdict(dict)
    p2 = defaultdict(list)
    for s in steer:
        lo[(s["dir"], s["item"], s["para"], s["arm"])][s["alpha"]] = s["lo"]
        p2[(s["dir"], abs(s["alpha"]))].append(s["p_two"])
    slopes = defaultdict(lambda: defaultdict(list))   # dir -> item -> slopes (both paras, both frames)
    by_frame = defaultdict(lambda: defaultdict(list))
    curve = defaultdict(lambda: defaultdict(list))
    for (d, item, para, arm), vals in lo.items():
        vals = {**vals, 0.0: base[(item, para, arm)]}
        a = np.array([x for x in alphas if x in vals])
        y = np.array([vals[x] for x in a])
        b = float(np.polyfit(a, y, 1)[0])
        slopes[d][item].append(b)
        by_frame[(d, arm)][item].append(b)
        for x in a:
            curve[(d, arm)][float(x)].append(vals[x])
    items = sorted(slopes["probe"])
    diff = {i: [np.mean(slopes["probe"][i]) - np.mean(slopes["placebo"][i])] for i in items}
    out = {"probe - placebo slope": boot_mean(diff),
           "slopes": {d: boot_mean(slopes[d]) for d in slopes},
           "slopes_by_frame": {f"{d}|{arm}": boot_mean(v) for (d, arm), v in by_frame.items()},
           "curve": {f"{d}|{arm}": {str(x): float(np.mean(v)) for x, v in sorted(c.items())} for (d, arm), c in curve.items()},
           "p_two": {f"{d}|{a}": float(np.mean(v)) for (d, a), v in sorted(p2.items())}}
    coh = [np.mean(p2[(d, 1.0)]) for d in ("probe", "placebo") if p2.get((d, 1.0))]
    out["coherent_at_1"] = bool(coh and min(coh) >= 0.8)
    return out


def reading_steer(st, lo_ok):
    c = st["probe - placebo slope"]
    if not lo_ok:
        return "untestable: the LO measure failed its validity gate"
    if not st["coherent_at_1"]:
        return "untestable: steering at |alpha| = 1 breaks the forced call (p_two < 0.8)"
    if c["hi"] <= -STEER_BAND or (c["diff"] <= -STEER_BAND and c["hi"] < 0):
        return "attribution steers the decision (toward other-owned raises accurate reports)"
    if c["lo"] >= STEER_BAND or (c["diff"] >= STEER_BAND and c["lo"] > 0):
        return "opposite: toward self-owned raises accurate reports"
    if -STEER_BAND < c["lo"] and c["hi"] < STEER_BAND:
        return "no steering effect beyond the placebo"
    return "inconclusive"


def lens(reads, rand, sets):
    """Paired decision-token contrasts (same item, cell or set, paraphrase): the fixed set's mean delta vs the deltas of
    200 random same-size sets; the set's percentile among them."""
    sizes = list(rand["sizes"])
    n_rand = int(rand["n_rand"])
    R = rand["rand"]
    col = {s: slice(i * n_rand, (i + 1) * n_rand) for i, s in enumerate(sizes)}
    size_of = {k: len(v) for k, v in sets["fixed"].items()}
    idx = {}
    for i, r in enumerate(reads):
        idx[(r["kind"], r["item"], r.get("group_c"), r["para"], r["arm"])] = i
    pairs = {"char - self": ("char", "self"), "char - none": ("char", "none"), "self - none": ("self", "none"),
             "always - self": ("always", "self"), "emph - self": ("emph", "self"),
             "char - self_ment": ("char", "self_ment"), "self_ment - self": ("self_ment", "self")}
    out = {}
    for name, (a, b) in pairs.items():
        ia, ib = [], []
        for (kind, item, g, para, arm), i in idx.items():
            if arm == a and (kind, item, g, para, b) in idx:
                ia.append(i)
                ib.append(idx[(kind, item, g, para, b)])
        if not ia:
            continue
        res = {"n_pairs": len(ia)}
        for k in ("role", "self", "rule", "precedent"):
            d_fixed = float(np.mean([reads[i]["mass"][k] - reads[j]["mass"][k] for i, j in zip(ia, ib)]))
            sl = col[size_of[k]]
            d_rand = (R[ia][:, sl] - R[ib][:, sl]).mean(0)
            res[k] = {"delta": d_fixed, "pct": float((d_rand < d_fixed).mean() * 100),
                      "rand_p95": float(np.percentile(d_rand, 95)), "rand_p5": float(np.percentile(d_rand, 5))}
        out[name] = res
    # precedent set: record vs press on the none frame (N1b core)
    ia, ib = [], []
    for (kind, item, g, para, arm), i in idx.items():
        if kind == "n1b_core" and g == "record" and arm == "none" and (kind, item, "press", para, "none") in idx:
            ia.append(i)
            ib.append(idx[(kind, item, "press", para, "none")])
    if ia:
        sl = col[size_of["precedent"]]
        d_fixed = float(np.mean([reads[i]["mass"]["precedent"] - reads[j]["mass"]["precedent"] for i, j in zip(ia, ib)]))
        d_rand = (R[ia][:, sl] - R[ib][:, sl]).mean(0)
        out["record - press (none)"] = {"n_pairs": len(ia), "precedent": {"delta": d_fixed,
                                        "pct": float((d_rand < d_fixed).mean() * 100)}}
    return out


def analyze(pod: Path, n1b_json: Path) -> dict:
    reads, rand, steer, probe, sets = load(pod)
    best = probe["best"]
    pl = probe["layers"][str(best)]
    gate_probe = bool(pl["cv_auc"] >= PROBE_GATE[0] and pl["shuffled_auc"] <= PROBE_GATE[1])
    n1b_rates = json.loads(n1b_json.read_text())["q27"]["boot"]["rates"]
    lov = lo_validity(reads, n1b_rates)
    level, contrasts = attribution(reads, best)
    st = steering(reads, steer)
    res = {"probe": probe, "gate_probe": gate_probe, "lo_validity": lov, "attribution_level": level,
           "attribution_contrasts": contrasts, "steering": st, "lens": lens(reads, rand, sets)}
    res["reading_attribution"] = (reading_attr(contrasts["self - char"]) if gate_probe
                                  else "untestable: the attribution probe failed its gate")
    res["reading_steering"] = (reading_steer(st, lov["pass"]) if gate_probe
                               else "untestable: the attribution probe failed its gate")
    # descriptive: decision-token projection vs LO in the record cell
    rec = [r for r in reads if r["kind"] == "n1b_core" and r["group_c"] == "record" and r["arm"] != "none"]
    res["record_dec_proj_vs_lo"] = float(spearmanr([r["proj_dec"][str(best)] for r in rec], [r["lo"] for r in rec]).statistic)
    dec = defaultdict(list)
    for r in reads:
        if r["kind"] == "n1b_core":
            dec[(r["arm"], r["group_c"])].append(r["proj_dec"][str(best)])
    res["decision_proj"] = {f"{a}|{g}": float(np.mean(v)) for (a, g), v in sorted(dec.items())}
    return res


def render(res: dict) -> str:
    ci = lambda c: f"{c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}]"
    p = res["probe"]
    L = ["# SC-N1c analysis: attribution probe, J-lens role/self sets, steering (Qwen3.6-27B)", "",
         f"- Probe: best layer {p['best']}; leave-commitment-out AUC "
         f"{p['layers'][str(p['best'])]['cv_auc']:.3f}, shuffled {p['layers'][str(p['best'])]['shuffled_auc']:.3f}; "
         f"gate {'PASS' if res['gate_probe'] else 'FAIL'}. All layers: "
         + "; ".join(f"L{l} {v['cv_auc']:.2f}/{v['shuffled_auc']:.2f}" for l, v in p["layers"].items()),
         f"- LO validity: Spearman {res['lo_validity']['spearman']:.2f} over {res['lo_validity']['n_cells']} cells, "
         f"median p_two {res['lo_validity']['median_p_two']:.3f}; gate {'PASS' if res['lo_validity']['pass'] else 'FAIL'}",
         f"- **Attribution reading:** {res['reading_attribution']}",
         f"- **Steering reading:** {res['reading_steering']}", "",
         "## Frame-line projection (training self-owned = +1, other-owned = -1; mean of key means, bootstrap over keys)", "",
         "| frame | projection | self - frame |", "|---|---|---|"]
    for f, v in sorted(res["attribution_level"].items()):
        c = res["attribution_contrasts"].get(f"self - {f}")
        L.append(f"| {f} | {ci(v)} | {ci(c) if c else '—'} |")
    st = res["steering"]
    L += ["", "## Steering (record cell; LO = logit failed - passed; + alpha = toward self-owned)", "",
          f"- probe - placebo slope (nats per alpha): {ci(st['probe - placebo slope'])}",
          "- slopes: " + "; ".join(f"{d} {ci(v)}" for d, v in st["slopes"].items()),
          "- by frame: " + "; ".join(f"{k} {ci(v)}" for k, v in sorted(st["slopes_by_frame"].items())),
          "- curves (mean LO by alpha): " + "; ".join(f"{k}: " + ", ".join(f"{a}:{v:+.2f}" for a, v in c.items())
                                                      for k, c in sorted(st["curve"].items())),
          "- p_two by |alpha|: " + ", ".join(f"{k} {v:.3f}" for k, v in st["p_two"].items()), "",
          "## LO by cell vs N1b sampled accurate rate", "", "| cell | mean LO | sampled accurate % |", "|---|---|---|"]
    L += [f"| {c['cell']} | {c['lo']:+.2f} | {c['sampled_accurate']:.0f} |" for c in res["lo_validity"]["cells"]]
    L += ["", "## J-lens sets at the decision token (paired delta, percentile among 200 random same-size sets)", "",
          "| contrast | n pairs | role | self | rule | precedent |", "|---|---|---|---|---|---|"]
    for name, v in res["lens"].items():
        cells = [f"{v[k]['delta']:+.2f} ({v[k]['pct']:.0f}th)" if k in v else "—" for k in ("role", "self", "rule", "precedent")]
        L.append(f"| {name} | {v['n_pairs']} | " + " | ".join(cells) + " |")
    L += ["", f"Descriptive: record-cell decision-token projection vs LO, Spearman {res['record_dec_proj_vs_lo']:+.2f}.",
          "Decision-token projection by frame|cell: " + "; ".join(f"{k} {v:+.2f}" for k, v in res["decision_proj"].items())]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.n1_attribution.n1c_analyze")
    ap.add_argument("--pod", type=Path, required=True)
    ap.add_argument("--n1b", type=Path, default=HERE / "results" / "n1b" / "analysis.json")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "n1c")
    a = ap.parse_args(argv)
    res = analyze(a.pod, a.n1b)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "analysis.json").write_text(json.dumps(res, indent=1))
    (a.out / "summary.md").write_text(render(res))
    print(f"wrote {a.out / 'summary.md'}")


if __name__ == "__main__":
    main()
