"""SC-CAL analysis (DESIGN.md, frozen rules). Per model and lens, scores are deep-band means of set log-masses.

  C1  identity appears: d = id_k(persona_k) - id_k(neutral), pooled over the 3 personas x 24 dilemmas, item-cluster
      bootstrap (10,000). PASS = CI lo > 0 and mean above the 95th percentile of the same statistic for the 200
      random sets of the identity sets' size. Reported: persona - mention, mention - neutral, role / self sets.
  C3  record (report only): precedent(record) - precedent(no_record) and id sets, 12 cases, case bootstrap.
  C5  injection read back (same site, alpha 1): d = c(inject c) - mean over c' != c of c(inject c'), read at deep-band
      layers above the injection layer, pooled over 3 concepts x 24 dilemmas, item-cluster bootstrap. PASS = CI lo > 0
      and every concept's mean > 0. Reported: alpha 0.5 / 2, the prop site, the Gaussian direction (c(gauss) as the
      control instead of c').
  C0  (report) the model's top-1 after the spider prompt, and ' spider''s best rank in the deep band.
  Verdict per model and lens: CAL PASS = C1 PASS and C5 PASS.

  python -m self_concept.cal.analyze --res self_concept/cal/results/main
"""
import argparse
import json
from pathlib import Path

import numpy as np

B = 10_000
PERSONAS = ("robin", "sherlock", "dracula")
CONCEPTS = ("c_ocean", "c_fire", "c_music")


def boot(vals_by_cluster, seed=0):
    """Mean and 95% CI of the mean over clusters (each cluster's values averaged first)."""
    m = np.array([np.mean(v) for v in vals_by_cluster])
    rng = np.random.default_rng(seed)
    bs = m[rng.integers(0, len(m), (B, len(m)))].mean(1)
    return float(m.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def load(res, model, lens):
    d = Path(res) / model
    slim = d / f"masses_{lens}.slim.npz"      # the full per-layer random-set arrays, reduced to their deep-band mean
    z = np.load(slim if slim.exists() else d / f"masses_{lens}.npz")
    reads = [json.loads(x) for x in (d / "reads.jsonl").read_text().splitlines()]
    sets = json.loads((d / "sets.json").read_text())
    return z, reads, sets, json.loads((d / "info.json").read_text())


def deep_mean(z, layers_keep):
    """[n_reads, S] fixed and [n_reads, R] random, averaged over the kept layers."""
    keep = np.isin(z["layers"], layers_keep)
    if "rand" not in z.files:                 # slim file: rand_deep holds the deep-band mean only
        same = list(layers_keep) == list(z["deep"])
        return z["fixed"][keep].mean(0), (z["rand_deep"] if same else None)
    return z["fixed"][keep].mean(0), z["rand"][keep].mean(0)


def analyse(res, model, lens):
    z, reads, sets, info = load(res, model, lens)
    names = list(z["names"])
    deep = list(z["deep"])
    fx, rx = deep_mean(z, deep)
    S = {k: fx[:, names.index(k)] for k in names}
    idx = {(r["check"], r["cond"], r["item"]): r["i"] for r in reads}
    sizes = sets["rand_sizes"]
    out = {"model": model, "lens": lens, "deep": [int(deep[0]), int(deep[-1])]}

    def pair(check, a, b, key_a, key_b=None, items=24, rand_col=None):
        """clusters over items of (key_a at cond a) - (key_b at cond b)."""
        key_b = key_b or key_a
        cl = []
        for i in range(items):
            ia, ib = idx.get((check, a, i)), idx.get((check, b, i))
            if ia is None or ib is None:
                continue
            if rand_col is None:
                cl.append([S[key_a][ia] - S[key_b][ib]])
            else:
                cl.append([rx[ia, rand_col] - rx[ib, rand_col]])
        return cl

    n_items = 1 + max(r["item"] for r in reads if r["check"] == "C1")
    # C1
    c1 = {}
    for cmp_, (ca, cb) in {"persona-neutral": ("persona_{k}", "neutral"), "persona-mention": ("persona_{k}", "mention_{k}"),
                           "mention-neutral": ("mention_{k}", "neutral")}.items():
        cl = [v for k in PERSONAS for v in pair("C1", ca.format(k=k), cb.format(k=k), f"id_{k}", items=n_items)]
        c1[cmp_] = boot(cl)
    cols = {k: sizes.index(len(sets["fixed"][f"id_{k}"])) * 200 for k in PERSONAS}   # each set's own size
    floor = []
    for j in range(200):
        cl = [v for k in PERSONAS for v in pair("C1", f"persona_{k}", "neutral", None, items=n_items,
                                                rand_col=cols[k] + j)]
        floor.append(np.mean([np.mean(v) for v in cl]))
    p95 = float(np.percentile(floor, 95)) if floor else float("nan")
    m, lo, hi = c1["persona-neutral"]
    c1["floor_p95"] = p95
    c1["id_set_sizes"] = [len(sets["fixed"][f"id_{k}"]) for k in PERSONAS]
    c1["PASS"] = bool(lo > 0 and m > p95)
    for s in ("role", "self"):
        c1[f"{s}:persona-neutral"] = boot([v for k in PERSONAS for v in pair("C1", f"persona_{k}", "neutral", s,
                                                                              items=n_items)])
        c1[f"{s}:persona-mention"] = boot([v for k in PERSONAS for v in pair("C1", f"persona_{k}", f"mention_{k}", s,
                                                                              items=n_items)])
    out["C1"] = c1
    # C3
    n3 = 1 + max((r["item"] for r in reads if r["check"] == "C3"), default=-1)
    out["C3"] = {k: boot(pair("C3", "record", "no_record", k, items=n3)) for k in ("precedent", "id_robin", "self")}
    # C5
    l_inj = info["l_inj"]
    fx5, _ = deep_mean(z, [x for x in deep if x > l_inj])
    S5 = {k: fx5[:, names.index(k)] for k in names}
    c5 = {}
    for site in ("same", "prop"):
        for alpha in (0.5, 1.0, 2.0):
            per, cl_all = {}, []
            for c in CONCEPTS:
                cl = []
                for i in range(n_items):
                    own = idx.get(("C5", f"inj_{c}_{alpha}_{site}", i))
                    oth = [idx.get(("C5", f"inj_{o}_{alpha}_{site}", i)) for o in CONCEPTS if o != c]
                    if own is None or None in oth:
                        continue
                    cl.append([S5[c][own] - np.mean([S5[c][x] for x in oth])])
                per[c] = boot(cl)[0]
                cl_all += cl
            c5[f"{site}_{alpha}"] = {"pooled": boot(cl_all), "per_concept": per}
        cl = []
        for c in CONCEPTS:
            for i in range(n_items):
                own, g = idx.get(("C5", f"inj_{c}_1.0_{site}", i)), idx.get(("C5", f"inj_gauss_1.0_{site}", i))
                if own is not None and g is not None:
                    cl.append([S5[c][own] - S5[c][g]])
        c5[f"{site}_1.0_vs_gauss"] = boot(cl)
    m, lo, _ = c5["same_1.0"]["pooled"]
    c5["PASS"] = bool(lo > 0 and all(v > 0 for v in c5["same_1.0"]["per_concept"].values()))
    out["C5"] = c5
    # C0
    sp = idx.get(("C0", "spider", 0))
    keep = np.isin(z["layers"], deep)
    sp_set = names.index("spider")
    out["C0"] = {"top1": info["spider_top1"], "spider_logmass_deep_max": float(z["fixed"][keep, sp, sp_set].max())
                 if sp is not None else None}
    out["CAL_PASS"] = bool(c1["PASS"] and c5["PASS"])
    return out


def fmt(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sc-cal analyze")
    ap.add_argument("--res", required=True)
    a = ap.parse_args(argv)
    res = Path(a.res)
    rows, lines = [], ["# SC-CAL summary", ""]
    for d in sorted(p for p in res.iterdir() if (p / "info.json").exists()):
        info = json.loads((d / "info.json").read_text())
        for lens in [*info["lenses"], "logit"]:
            r = analyse(res, d.name, lens)
            rows.append(r)
            lines += [f"## {d.name} / {lens} (deep {r['deep'][0]}-{r['deep'][1]}): CAL {'PASS' if r['CAL_PASS'] else 'FAIL'}",
                      f"- C1 identity persona - neutral {fmt(r['C1']['persona-neutral'])} (floor p95 "
                      f"{r['C1']['floor_p95']:+.2f}) -> {'PASS' if r['C1']['PASS'] else 'FAIL'}; persona - mention "
                      f"{fmt(r['C1']['persona-mention'])}; mention - neutral {fmt(r['C1']['mention-neutral'])}",
                      f"- role: persona - neutral {fmt(r['C1']['role:persona-neutral'])}, persona - mention "
                      f"{fmt(r['C1']['role:persona-mention'])}; self: persona - neutral "
                      f"{fmt(r['C1']['self:persona-neutral'])}",
                      f"- C3 record - none: precedent {fmt(r['C3']['precedent'])}, id_robin {fmt(r['C3']['id_robin'])}",
                      f"- C5 same a=1 {fmt(r['C5']['same_1.0']['pooled'])} per concept "
                      f"{ {k: round(v, 2) for k, v in r['C5']['same_1.0']['per_concept'].items()} } -> "
                      f"{'PASS' if r['C5']['PASS'] else 'FAIL'}; prop a=1 {fmt(r['C5']['prop_1.0']['pooled'])}; "
                      f"vs gauss {fmt(r['C5']['same_1.0_vs_gauss'])}",
                      f"- C0 spider top-1 {r['C0']['top1']!r}, spider log-mass deep max {r['C0']['spider_logmass_deep_max']}",
                      ""]
    (res / "summary.md").write_text("\n".join(lines))
    (res / "analysis.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
