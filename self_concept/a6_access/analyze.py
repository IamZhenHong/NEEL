"""SC-A6 analysis (DESIGN.md "Rules"): P11e's scoring and task-clustered bootstrap (10,000 draws, seed 0, each family's
clusters resampled once per draw and shared by every cell), with crossed-cell, chance-corrected agreement.

  uv run python -m self_concept.a6_access.analyze [--res DIR]      -> DIR/analysis.json, DIR/summary.md

Arms {base}-{arm}: NW / NZ x OWN / CROSS / UNT (this run, third-stage seed 2). Rows file R_NW trains NW-OWN and
NZ-CROSS; R_NZ trains NZ-OWN and NW-CROSS. UNT is the in-run untrained base; if either UNT cell is incomplete, both
are replaced by the cross-run fallback (V1c eval_N{W,Z}_s1 for predict / just / self_mc / mc, V1 full_s1 for deploy,
and V1 full_s1 vmc, pooled over listings, as the gate's untrained acts). Cross-run reference arms (levels only, never
in a rule): P11e's seed-1 OWN / SHUF (REF-P11eOWN, REF-P11eSHUF), P11's style placebo (REF-PLAC) and V1c's untrained
models (REF-UNTv1c).
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from persona_affect.p11e_mirror import analyze as P11E
from persona_affect.p11e_mirror.items import items as p11e_items
from persona_affect.v1_value_hook.analyze import split_cell

_H = Path(__file__).resolve().parent
_PA = _H.parents[1] / "persona_affect"
RES = _H / "results" / "main"
SEED = 2
BASES, ARMS = ("NW", "NZ"), ("OWN", "CROSS", "UNT")
OTHER = {"NW": "NZ", "NZ": "NW"}
FAMILY, ROLES = P11E.FAMILY, P11E.ROLES
N_BOOT = 10000
GATE_PP, ACCESS_PP, SMALL_PP, CONV_PP, STAY_PP = 10.0, 10.0, 5.0, -10.0, 10.0
FALLBACK = [(_PA / "v1c_selfreport/results/main", ("predict", "just", "self_mc", "mc")),
            (_PA / "v1_value_hook/results/full_s1", ("deploy",))]
FALLBACK_VMC = _PA / "v1_value_hook/results/full_s1"
REFS = {"REF-P11eOWN": (_PA / "p11e_mirror/results/main", "OWN"), "REF-P11eSHUF": (_PA / "p11e_mirror/results/main", "SHUF"),
        "REF-PLAC": (_PA / "p11_saydo/results/main", "PLAC")}


def rows_file(base, arm):
    """Whose labels trained this cell (None for UNT)."""
    return None if arm == "UNT" else (base if arm == "OWN" else OTHER[base])


def trained_on(src):
    """The two trained cells whose third stage used R_{src}."""
    return [f"{src}-OWN", f"{OTHER[src]}-CROSS"]


# ------------------------------------------------------------------------------------------------ loading
def _row(arm, r, perm=None):
    b = r["block"]
    y, role = P11E.y_of(r)
    return {"arm": arm, "block": b, "cond": r["cond"] if b == "just" else "all", "fam": FAMILY[b], "cl": r["tid"],
            "perm": perm if perm is not None else r["cond"], "y": y, "role": role}


def _read(f):
    return [json.loads(x) for x in Path(f).read_text().splitlines() if x.strip()]


def _keep(r, blocks=None):
    b = r["block"]
    return b in FAMILY and (not blocks or b in blocks) and not (b == "vmc" and not r["key"].startswith("PA-P11e"))


def load(res=RES, refs=True):
    """(rows, info): this run's cells, UNT from the run or the fallback, and the reference arms."""
    res = Path(res)
    n_items = len(p11e_items())
    rows, counts = [], {}
    for f in sorted(res.glob("eval_*.jsonl")):
        name, s = split_cell(f.stem[len("eval_"):])
        if s != SEED or "-" not in name or name.split("-", 1)[0] not in BASES or name.split("-", 1)[1] not in ARMS:
            continue
        recs = _read(f)
        counts[name] = len(recs)
        rows += [_row(name, r) for r in recs if _keep(r)]
    unt_ok = all(counts.get(f"{b}-UNT", 0) >= n_items for b in BASES)
    info = {"cells": counts, "n_items": n_items, "unt_source": "in-run" if unt_ok else "fallback (cross-run)"}
    if not unt_ok:
        rows = [r for r in rows if not r["arm"].endswith("-UNT")]
        for d, blocks in FALLBACK:
            for b in BASES:
                rows += [_row(f"{b}-UNT", r) for r in _read(d / f"eval_{b}_s1.jsonl") if _keep(r, blocks)]
        for b in BASES:   # the gate's untrained acts: V1's own vmc keys, one listing each -> pooled per task
            rows += [_row(f"{b}-UNT", r, perm="pool") for r in _read(FALLBACK_VMC / f"eval_{b}_s1.jsonl")
                     if r["block"] == "vmc"]
    if refs:
        for arm, (d, x) in REFS.items():
            for b in BASES:
                f = d / f"eval_{b}-{x}_s1.jsonl"
                if f.exists():
                    rows += [_row(f"{b}-{arm}", r) for r in _read(f) if _keep(r)]
        for d, blocks in FALLBACK:
            for b in BASES:
                f = d / f"eval_{b}_s1.jsonl"
                if f.exists():
                    rows += [_row(f"{b}-REF-UNTv1c", r) for r in _read(f) if _keep(r, blocks)]
    return rows, info


# ------------------------------------------------------------------------------------------------ agreement
def _role_mat(agg, arm, blk, pool):
    """{(ci, perm or 'pool'): counts over ROLES}."""
    out = defaultdict(lambda: np.zeros(len(ROLES)))
    for (ci, perm), fr in agg.dist.get((arm, blk), {}).items():
        out[(ci, "pool" if pool else perm)] += np.array([fr.get(x, 0) for x in ROLES], dtype=float)
    return out


def agree_fns(agg):
    """xagree(f_arm, a_arm, W): raw agreement of f's pred_vmc forecasts with a's vmc acts, per (task, listing) when
    both have listings, else per task, averaged over listings then tasks; chance(f_arm, a_arm, W): from the pooled
    cluster-weighted shares. Both in points, arrays over draws."""
    k = len(agg.cl.get("vmc", {}))

    def _parts(f_arm, a_arm):
        pool = any(p == "pool" for (_, p) in agg.dist.get((a_arm, "vmc"), {}))
        F, A = _role_mat(agg, f_arm, "pred_vmc", pool), _role_mat(agg, a_arm, "vmc", pool)
        v, m = np.zeros(k), np.zeros(k)
        for key, fr in F.items():
            ar = A.get(key)
            if ar is None or not fr.sum() or not ar.sum():
                continue
            v[key[0]] += float((fr / fr.sum()) @ (ar / ar.sum()))
            m[key[0]] += 1
        Fc, Ac = np.zeros((k, len(ROLES))), np.zeros((k, len(ROLES)))
        for (ci, _), x in F.items():
            Fc[ci] += x
        for (ci, _), x in A.items():
            Ac[ci] += x
        return v, m, Fc, Ac

    cache = {}

    def parts(f_arm, a_arm):
        if (f_arm, a_arm) not in cache:
            cache[(f_arm, a_arm)] = _parts(f_arm, a_arm)
        return cache[(f_arm, a_arm)]

    def w(W):
        return W["vmc"] if W is not None else np.ones((1, k))

    def xagree(f_arm, a_arm, W=None):
        v, m, _, _ = parts(f_arm, a_arm)
        with np.errstate(invalid="ignore", divide="ignore"):
            return 100.0 * (w(W) @ v) / (w(W) @ m)

    def chance(f_arm, a_arm, W=None):
        _, _, Fc, Ac = parts(f_arm, a_arm)
        SF, SA = w(W) @ Fc, w(W) @ Ac
        with np.errstate(invalid="ignore", divide="ignore"):
            return 100.0 * ((SF / SF.sum(axis=1, keepdims=True)) * (SA / SA.sum(axis=1, keepdims=True))).sum(axis=1)

    def excess(f_arm, a_arm, W=None):
        return xagree(f_arm, a_arm, W) - chance(f_arm, a_arm, W)
    return xagree, chance, excess


# ------------------------------------------------------------------------------------------------ stats
def boot_w(stat, W, n_draws):
    """{est, lo, hi, skipped} for stat(W) -> array over draws (W=None gives the point estimate)."""
    est = float(np.asarray(stat(None), dtype=float).reshape(-1)[0])
    d = np.asarray(stat(W), dtype=float).reshape(-1)
    ok = d[~np.isnan(d)]
    lo, hi = (float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))) if ok.size else (None, None)
    return {"est": None if math.isnan(est) else est, "lo": lo, "hi": hi, "skipped": int(n_draws - ok.size)}


def same_data_sep(rate, block, cond="all"):
    """P's formula on a block: mean over rows files of (NW trained on it - NZ trained on it) - untrained gap."""
    def f(W):
        r = lambda arm: rate(W, arm, block, cond)  # noqa: E731
        g_nw = r("NW-OWN") - r("NZ-CROSS")      # both trained on R_NW
        g_nz = r("NW-CROSS") - r("NZ-OWN")      # both trained on R_NZ
        return (g_nw + g_nz) / 2 - (r("NW-UNT") - r("NZ-UNT"))
    return f


def verdict_of(gate_ok, p):
    if not gate_ok:
        return "untestable (the forecast training gate failed)"
    if p["est"] is None:
        return "untestable (the primary is undefined)"
    if p["est"] >= ACCESS_PP and p["lo"] > 0:
        return "access gain"
    if p["est"] >= SMALL_PP and p["lo"] > 0:
        return "small access gain"
    if p["est"] <= CONV_PP and p["hi"] < 0:
        return "convergence"
    return "no access gain"


def analyze(rows, info=None, n_boot=N_BOOT):
    info = info or {}
    agg = P11E.Agg(rows)
    W = agg.weights(n_boot)
    have = sorted({r["arm"] for r in rows})
    xagree, chance, excess = agree_fns(agg)

    def rate(Wx, arm, block, cond="all"):
        return agg.fns(Wx)[0](arm, block, cond)
    B = lambda stat: boot_w(stat, W, n_boot)  # noqa: E731
    lv = lambda x: None if x is None or np.isnan(x) else round(float(x), 1)  # noqa: E731
    levels = {}
    for a in have:
        r1 = lambda blk, c="all": lv(rate(None, a, blk, c)[0])  # noqa: E731
        levels[a] = {"just": r1("just", "none"), "just_ch": r1("just", "ch"), "deploy": r1("deploy"),
                     "predict": r1("predict"), "pred_dep": r1("pred_dep"), "self_mc": r1("self_mc"),
                     "self_mc_dep": r1("self_mc_dep"), "mc": r1("mc"), "gen": r1("gen"),
                     "vmc_roles": P11E._shares(agg, a, "vmc"), "pred_vmc_roles": P11E._shares(agg, a, "pred_vmc"),
                     "excess_self_vmc": lv(excess(a, a)[0]) if (a, "pred_vmc") in agg.dist and (a, "vmc") in agg.dist
                     else None}
    cells = [f"{b}-{x}" for b in BASES for x in ARMS]
    full = all(c in have for c in cells)
    out = {"arms": have, "levels": levels, "complete": full, "info": info}
    if not full:
        out["verdict"] = "incomplete (a cell is missing: " + ", ".join(c for c in cells if c not in have) + ")"
        return out
    trained = [f"{b}-{x}" for b in BASES for x in ("OWN", "CROSS")]
    src_of = {c: rows_file(*c.split("-")) for c in trained}
    gate_cells = {c: B(lambda Wx, c=c: excess(c, f"{src_of[c]}-UNT", Wx)) for c in trained}
    gate = B(lambda Wx: sum(excess(c, f"{src_of[c]}-UNT", Wx) for c in trained) / len(trained))
    gate_ok = P11E._ok(gate, GATE_PP)
    primary = B(same_data_sep(rate, "predict"))
    verdict = verdict_of(gate_ok, primary)
    stay = {f"{c}|{blk}": B(lambda Wx, c=c, blk=blk: rate(Wx, c, blk, "none" if blk == "just" else "all")
                            - rate(Wx, f"{c[:2]}-UNT", blk, "none" if blk == "just" else "all"))
            for c in trained for blk in ("just", "deploy")}
    moved = [k for k, v in stay.items() if v["est"] is None or abs(v["est"]) > STAY_PP]

    def gap(Wx, x, y, block="predict", cond="all"):
        return rate(Wx, f"NW-{x}", block, cond) - rate(Wx, f"NZ-{y}", block, cond)

    def spa(T, Wx):
        return excess(f"{T}-OWN", f"{T}-OWN", Wx) - excess(f"{OTHER[T]}-CROSS", f"{T}-OWN", Wx)

    sec = {
        "own_label_effect_predict": B(lambda Wx: gap(Wx, "OWN", "OWN") - gap(Wx, "CROSS", "CROSS")),
        "predict_gap": {f"{x}-{y}": B(lambda Wx, x=x, y=y: gap(Wx, x, y))
                        for x, y in (("OWN", "OWN"), ("CROSS", "CROSS"), ("OWN", "CROSS"), ("CROSS", "OWN"),
                                     ("UNT", "UNT"))},
        "spa": {T: B(lambda Wx, T=T: spa(T, Wx)) for T in BASES} | {"pooled": B(lambda Wx: (spa("NW", Wx) + spa("NZ", Wx)) / 2)},
        "spa_parts": {f"{T}|{who}": B(lambda Wx, T=T, fa=fa: excess(fa, f"{T}-OWN", Wx))
                      for T in BASES for who, fa in (("self", f"{T}-OWN"), ("cross", f"{OTHER[T]}-CROSS"))},
        "same_data_sep": {blk: B(same_data_sep(rate, blk)) for blk in ("pred_dep", "self_mc", "self_mc_dep", "mc")}
        | {"act_just_none": B(same_data_sep(rate, "just", "none")), "act_deploy": B(same_data_sep(rate, "deploy"))},
        "calibration_error": {c: B(lambda Wx, c=c: np.abs(rate(Wx, c, "predict") - rate(Wx, c, "just", "none")))
                              for c in cells},
        "forecast_vs_unt": {c: B(lambda Wx, c=c: rate(Wx, c, "predict") - rate(Wx, f"{c[:2]}-UNT", "predict"))
                            for c in trained},
        "gate_cells": gate_cells,
        "excess_self_vmc": {c: B(lambda Wx, c=c: excess(c, c, Wx)) for c in cells
                            if (c, "pred_vmc") in agg.dist and (c, "vmc") in agg.dist},
    }
    out.update({"gate": {"mean_excess_vs_label_source_unt": gate, "passed": gate_ok},
                "primary": {"same_data_separation_predict": primary},
                "behaviour": {"trained_minus_unt": stay, "moved": moved,
                              "verdict": "behaviour unchanged" if not moved else "behaviour moved: " + ", ".join(moved)},
                "verdict": verdict, "secondary": sec})
    return out


# ------------------------------------------------------------------------------------------------ report
def _ci(b):
    return "n/a" if not b or b.get("est") is None else f"{b['est']:+.1f} [{b['lo']:+.1f}, {b['hi']:+.1f}]"


def summary(r):
    cols = ("just", "just_ch", "deploy", "predict", "pred_dep", "self_mc", "self_mc_dep", "mc", "excess_self_vmc", "gen")
    L = ["# SC-A6 analysis", "", f"UNT source: {r['info'].get('unt_source', '?')}", "",
         "| arm | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for a in r["arms"]:
        v = r["levels"][a]
        L.append(f"| {a} | " + " | ".join("" if v[c] is None else f"{v[c]:.1f}" for c in cols) + " |")
    L += ["", "Value-task roles (act / forecast): " + "; ".join(
        f"{a} {r['levels'][a]['vmc_roles']} / {r['levels'][a]['pred_vmc_roles']}" for a in r["arms"]
        if r["levels"][a]["vmc_roles"] or r["levels"][a]["pred_vmc_roles"])]
    if not r["complete"]:
        return "\n".join(L + ["", f"**{r['verdict']}**"]) + "\n"
    g, p = r["gate"], r["primary"]
    L += ["", f"Gate (pred_vmc vs the label source's untrained acts, chance-corrected, mean of 4 >= +{GATE_PP:g}): "
          f"{_ci(g['mean_excess_vs_label_source_unt'])} -> {g['passed']}",
          f"Primary (same-data separation of the justice forecast, NW - NZ, minus untrained): "
          f"{_ci(p['same_data_separation_predict'])} -> **{r['verdict']}**",
          f"Behaviour (trained - own UNT, |d| <= {STAY_PP:g}): " + "; ".join(
              f"{k} {_ci(v)}" for k, v in r["behaviour"]["trained_minus_unt"].items())
          + f" -> **{r['behaviour']['verdict']}**", "", "Secondary:"]
    for k, v in r["secondary"].items():
        if isinstance(v, dict) and "est" not in v:
            L.append(f"- {k}: " + "; ".join(f"{a} {_ci(x)}" for a, x in v.items()))
        else:
            L.append(f"- {k}: {_ci(v)}")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sc-a6 analyze")
    ap.add_argument("--res", default=str(RES))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    rows, info = load(a.res)
    res = analyze(rows, info, a.n_boot)
    (Path(a.res) / "analysis.json").write_text(json.dumps(res, indent=1))
    s = summary(res)
    (Path(a.res) / "summary.md").write_text(s)
    print(s)


if __name__ == "__main__":
    main()
