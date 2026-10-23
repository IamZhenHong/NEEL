"""SC-N3 analysis (DESIGN.md, frozen rules). Task-clustered bootstrap: 10,000 draws, seed 0, one draw of the 16 tasks
shared by every cell (B2: the 8 test tasks).

  act      share of `just` none replies whose option is the penalty one (V1's role), all parsed records.
  J        UNI - REF log-mass, primary lens, deep band, at P0 (the last prompt token); reported at D too.
  probe    L2 logistic regression (C = 1, standardised), NW-PLAC (1) vs NZ-PLAC (0), block output at P0, layer
           PROBE_PRIMARY; gate: leave-one-task-out AUC >= 0.9. m = mean decision value per cell.
  s_X      text share: NW-SAYR (X(NW-PLAC) - X(NW-SAYR)) / (X(NW-PLAC) - X(NZ-PLAC)); NZ-SAYU (X(NZ-SAYU) - X(NZ-PLAC)) /
           (X(NW-PLAC) - X(NZ-PLAC)); pooled = their mean. 0 = like its own habit cell, 1 = all the way to the other.
  P        ds = s_act - s_probe (pooled). >= 0.25 with lo > 0: "the act moves with the text while the habit stays
           represented"; |ds| < 0.15: "the habit representation moves with the act"; else "unclear".
  W        (absorbs C2) w_text = 100 T_J / T_A, w_habit = 100 H_J / H_A (nats of J per 100 points of act), T = the text's
           pull (mean of NZ-SAYU - NZ-PLAC and NW-PLAC - NW-SAYR), H = NW-PLAC - NZ-PLAC. dw = w_text - w_habit: lo > 0
           and w_habit < 0.5 w_text: "the habit is quieter in the workspace than the text"; hi < 0: "louder"; else "no
           clear difference".
  G        habit gap H_A >= 30; text pull T_A >= 10; lens gate = the 9B CAL (C1 and C5 PASS on the primary lens) and
           check 2: UNI(NZ-SAYU) - UNI(NZ-PLAC) at P0, CI lo > 0 and above the random floor's 95th percentile. W and B2
           are read only if the lens gate passes (else descriptive); P only if the probe gate passes.
  B2       NZ-SAYU test half. rho_arm = (LO_arm - LO_SAYU) / (LO_PLAC - LO_SAYU) (LO = mean log-odds penalise vs refer at
           D). rho_top10 >= 0.5 and rho_top10 - max(rho_rand10, rho_bottom10) >= 0.25 with lo > 0: "the lever runs
           through these verbalisable concepts"; rho_top10 < 0.2 or that margin < 0.1: "not through these concepts";
           else "partial". An arm whose top-1 at D is an option token < 95% of the time is "broken". Gate |LO_PLAC -
           LO_SAYU| >= 1 nat.
  E        (descriptive) the reason's trained wording (P15's tags) by cell x chosen option.

  python -m self_concept.n3_split.analyze --res self_concept/n3_split/results/main [--cal self_concept/cal/results/n3]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from self_concept.n3_split import items as I

B = 10_000
PRIM = "cb"
TAGS = {   # P15's descriptive tags (persona_affect/p15_stores/analyze.py THINK_RE)
    "SAYR_words": re.compile(r"people in charge|whoever is in charge|team that decides|hand (it )?over|"
                             r"pass (it|the case|this) (on|to)", re.I),
    "SAYU_words": re.compile(r"deal with (it|this) myself|(apply|impose) the (penalty|sanction) (directly|myself)|"
                             r"handle (it|this) personally", re.I),
}


def load(res):
    res = Path(res)
    out = {}
    for c in I.CELLS:
        if not (res / "reads" / f"reads_{c}.npz").exists():
            continue
        z = np.load(res / "reads" / f"reads_{c}.npz")
        meta = [json.loads(x) for x in (res / "reads" / f"reads_{c}.jsonl").read_text().splitlines()]
        info = json.loads((res / "reads" / f"info_{c}.json").read_text())
        ev = [json.loads(x) for x in (res / "evals" / f"eval_{c}.jsonl").read_text().splitlines()]
        out[c] = {"z": z, "meta": meta, "info": info, "evals": ev}
    return out


def by_task(values, tids, tasks):
    return {t: [v for v, u in zip(values, tids) if u == t] for t in tasks}


def stat_draws(groups, tasks, idx, fn=np.mean):
    """[B] statistic over resampled tasks (pooling every record of each drawn task) and the point value."""
    s = np.array([np.sum(groups[t]) for t in tasks], float)
    n = np.array([len(groups[t]) for t in tasks], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = s[idx].sum(1) / n[idx].sum(1)
    return bs, float(s.sum() / n.sum()) if n.sum() else float("nan")


def ci(bs, p):
    return [float(p), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]


def j_of(c, pos=0):
    z = c["z"]
    lay = list(z[f"layers_{PRIM}"])
    keep = [lay.index(x) for x in z["deep"]]
    f = z[f"fixed_{PRIM}"][:, pos][:, keep]                  # [n, deep, 2]
    return (f[..., 0] - f[..., 1]).mean(1), f[..., 0].mean(1)


def probe(d):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    li = list(d["NW-PLAC_s1"]["z"]["probe_layers"]).index(I.PROBE_PRIMARY) if I.PROBE_PRIMARY in \
        list(d["NW-PLAC_s1"]["z"]["probe_layers"]) else -1
    X = {c: d[c]["z"]["resid"][:, 0, li].astype(np.float32) for c in d}
    tids = {c: np.array([m["tid"] for m in d[c]["meta"]]) for c in d}
    Xt = np.concatenate([X["NW-PLAC_s1"], X["NZ-PLAC_s1"]])
    yt = np.r_[np.ones(len(X["NW-PLAC_s1"])), np.zeros(len(X["NZ-PLAC_s1"]))]
    tt = np.r_[tids["NW-PLAC_s1"], tids["NZ-PLAC_s1"]]
    scores = np.zeros(len(yt))
    for t in np.unique(tt):
        tr = tt != t
        sc = StandardScaler().fit(Xt[tr])
        m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xt[tr]), yt[tr])
        scores[~tr] = m.decision_function(sc.transform(Xt[~tr]))
    auc = float(roc_auc_score(yt, scores))
    sc = StandardScaler().fit(Xt)
    m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xt), yt)
    dec = {c: m.decision_function(sc.transform(X[c])) for c in d}
    for c in ("NW-PLAC_s1", "NZ-PLAC_s1"):      # held-out scores for the training cells
        mask = np.r_[np.ones(len(X["NW-PLAC_s1"]), bool), np.zeros(len(X["NZ-PLAC_s1"]), bool)]
        dec[c] = scores[mask] if c == "NW-PLAC_s1" else scores[~mask]
    return auc, dec, tids


def share(draw, base_nw, base_nz, nw_text, nz_text):
    """pooled text share from per-draw cell means (arrays or point floats)."""
    draw = {k: np.asarray(v, float) for k, v in draw.items()}
    gap = draw[base_nw] - draw[base_nz]
    with np.errstate(invalid="ignore", divide="ignore"):
        s_nw = (draw[base_nw] - draw[nw_text]) / gap
        s_nz = (draw[nz_text] - draw[base_nz]) / gap
    return (s_nw + s_nz) / 2, s_nw, s_nz


def analyse(res, cal=None):
    d = load(res)
    out = {"cells": list(d)}
    tasks = sorted({m["tid"] for c in d.values() for m in c["meta"]})
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(tasks), (B, len(tasks)))
    NW, NZ, SR, SU = I.CELLS
    act, J, Jd = {}, {}, {}
    for c, v in d.items():
        ys = [1.0 if r["role"] == "unilateral" else 0.0 for r in v["evals"] if r.get("role")]
        ts = [r["tid"] for r in v["evals"] if r.get("role")]
        act[c] = stat_draws(by_task(ys, ts, tasks), tasks, idx)
        t = [m["tid"] for m in v["meta"]]
        J[c] = stat_draws(by_task(j_of(v)[0], t, tasks), tasks, idx)
        Jd[c] = stat_draws(by_task(j_of(v, 1)[0], t, tasks), tasks, idx)
    out["act"] = {c: ci(*v) for c, v in act.items()}
    out["J_P0"] = {c: ci(*v) for c, v in J.items()}
    out["J_D"] = {c: ci(*v) for c, v in Jd.items()}
    A = {c: v[0] for c, v in act.items()}
    Ap = {c: v[1] for c, v in act.items()}
    HA = (A[NW] - A[NZ]) * 100, (Ap[NW] - Ap[NZ]) * 100
    TA = ((A[SU] - A[NZ]) + (A[NW] - A[SR])) * 50, ((Ap[SU] - Ap[NZ]) + (Ap[NW] - Ap[SR])) * 50
    out["G"] = {"habit_gap": ci(*HA), "text_pull": ci(*TA), "habit_ok": HA[1] >= 30, "text_ok": TA[1] >= 10}
    # lens gate: the 9B CAL + check 2
    cal_ok = None
    if cal and Path(cal, "analysis.json").exists():
        rows = json.loads(Path(cal, "analysis.json").read_text())
        hit = [r for r in rows if r["model"] == "qwen9" and r["lens"] == PRIM]
        cal_ok = bool(hit and hit[0]["CAL_PASS"])
    uni = {c: stat_draws(by_task(j_of(d[c])[1], [m["tid"] for m in d[c]["meta"]], tasks), tasks, idx)
           for c in (NZ, SU)}
    c2 = (uni[SU][0] - uni[NZ][0], uni[SU][1] - uni[NZ][1])
    floor = []
    zs, zp = d[SU]["z"], d[NZ]["z"]
    lay = list(zs[f"layers_{PRIM}"])
    keep = [lay.index(x) for x in zs["deep"]]
    size = len(d[SU]["info"]["set_ids"]["UNI"])
    col0 = d[SU]["info"]["sizes"].index(size) * 200
    for j in range(200):
        floor.append(zs[f"rand_{PRIM}"][:, 0, keep, col0 + j].mean() - zp[f"rand_{PRIM}"][:, 0, keep, col0 + j].mean())
    p95 = float(np.percentile(floor, 95))
    c2ci = ci(*c2)
    check2 = bool(c2ci[1] > 0 and c2ci[0] > p95)
    out["lens_gate"] = {"cal_9b": cal_ok, "check2_uni_SAYU_minus_PLAC": c2ci, "check2_floor_p95": p95,
                        "check2_pass": check2, "pass": bool(cal_ok and check2)}
    # P
    auc, dec, tids = probe(d)
    m = {c: stat_draws(by_task(dec[c], tids[c], tasks), tasks, idx) for c in d}
    sp, snw, snz = share({c: v[0] for c, v in m.items()}, NW, NZ, SR, SU)
    spp = share({c: v[1] for c, v in m.items()}, NW, NZ, SR, SU)
    sa, anw, anz = share(A, NW, NZ, SR, SU)
    sap = share(Ap, NW, NZ, SR, SU)
    ds = ci(sa - sp, sap[0] - spp[0])
    lab = ("the act moves with the text while the habit stays represented" if ds[0] >= 0.25 and ds[1] > 0 else
           "the habit representation moves with the act" if abs(ds[0]) < 0.15 else "unclear")
    out["P"] = {"probe_auc": auc, "probe_ok": auc >= 0.9, "s_act": ci(sa, sap[0]), "s_probe": ci(sp, spp[0]),
                "s_act_by_cell": [sap[1], sap[2]], "s_probe_by_cell": [spp[1], spp[2]], "ds": ds,
                "label": lab if auc >= 0.9 else "untestable (probe gate)"}
    # W
    Jb = {c: v[0] for c, v in J.items()}
    Jp = {c: v[1] for c, v in J.items()}
    TJ = ((Jb[SU] - Jb[NZ]) + (Jb[NW] - Jb[SR])) / 2, ((Jp[SU] - Jp[NZ]) + (Jp[NW] - Jp[SR])) / 2
    HJ = Jb[NW] - Jb[NZ], Jp[NW] - Jp[NZ]
    with np.errstate(invalid="ignore", divide="ignore"):
        wt, wh = 100 * TJ[0] / TA[0], 100 * HJ[0] / HA[0]
        wtp, whp = (100 * np.float64(TJ[1]) / np.float64(TA[1]), 100 * np.float64(HJ[1]) / np.float64(HA[1]))
    dw = ci(wt - wh, wtp - whp)
    lab = ("the habit is quieter in the workspace than the text" if dw[1] > 0 and whp < 0.5 * wtp else
           "louder" if dw[2] < 0 else "no clear difference")
    out["W"] = {"w_text": ci(wt, wtp), "w_habit": ci(wh, whp), "dw": dw, "T_J": ci(*TJ), "H_J": ci(*HJ),
                "label": lab if out["lens_gate"]["pass"] and out["G"]["habit_ok"] and out["G"]["text_ok"]
                else f"descriptive ({lab})"}
    # B2
    b2 = d[SU]["info"].get("b2")
    if b2:
        test = sorted({r["tid"] for r in b2["rows"]})
        ridx = rng.integers(0, len(test), (B, len(test)))
        lo_su = {m["key"]: v for m, v in zip(d[SU]["meta"], d[SU]["z"]["lo"])}
        lo_pl = by_task([v for m, v in zip(d[NZ]["meta"], d[NZ]["z"]["lo"]) if m["half"] == "test"],
                        [m["tid"] for m in d[NZ]["meta"] if m["half"] == "test"], test)
        base = stat_draws(by_task([lo_su[r["key"]] for r in b2["rows"]], [r["tid"] for r in b2["rows"]], test),
                          test, ridx)
        plac = stat_draws(lo_pl, test, ridx)
        arms = {}
        for arm in ("top5", "top10", "bottom10", "rand10"):
            v = stat_draws(by_task([r[arm]["lo"] for r in b2["rows"]], [r["tid"] for r in b2["rows"]], test),
                           test, ridx)
            with np.errstate(invalid="ignore", divide="ignore"):
                rho = ((v[0] - base[0]) / (plac[0] - base[0]),
                       np.float64(v[1] - base[1]) / np.float64(plac[1] - base[1]))
            ok = float(np.mean([r[arm]["top1_ok"] for r in b2["rows"]]))
            arms[arm] = {"rho": rho, "coherent": ok}
        margin = (arms["top10"]["rho"][0] - np.maximum(arms["rand10"]["rho"][0], arms["bottom10"]["rho"][0]),
                  arms["top10"]["rho"][1] - max(arms["rand10"]["rho"][1], arms["bottom10"]["rho"][1]))
        mci = ci(*margin)
        r10 = arms["top10"]["rho"][1]
        lab = ("the lever runs through these verbalisable concepts" if r10 >= 0.5 and mci[0] >= 0.25 and mci[1] > 0
               else "not through these concepts" if r10 < 0.2 or mci[0] < 0.1 else "partial")
        if arms["top10"]["coherent"] < 0.95:
            lab = "broken (top10 incoherent)"
        gate = abs(plac[1] - base[1]) >= 1
        out["B2"] = {"lo_SAYU": ci(*base), "lo_PLAC": ci(*plac), "gate": gate, "tokens": b2["arms"],
                     "arms": {k: {"rho": ci(*v["rho"]), "coherent": v["coherent"]} for k, v in arms.items()},
                     "margin": mci, "label": lab if gate and out["lens_gate"]["pass"] else f"descriptive ({lab})"}
    # E
    E = {}
    for c, v in d.items():
        for r in v["evals"]:
            if not r.get("role"):
                continue
            k = f"{c}|{r['role']}"
            E.setdefault(k, {"n": 0, **{t: 0 for t in TAGS}})
            E[k]["n"] += 1
            for t, rx in TAGS.items():
                E[k][t] += bool(rx.search(r["text"].split("<parameter=reason>")[-1]))
    out["E"] = E
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="n3 analyze")
    ap.add_argument("--res", required=True)
    ap.add_argument("--cal")
    a = ap.parse_args(argv)
    out = analyse(a.res, a.cal)
    Path(a.res, "analysis.json").write_text(json.dumps(out, indent=1, default=float))
    f = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"  # noqa: E731
    L = ["# SC-N3 summary", "", "| cell | act % | J (P0) | J (D) |", "|---|---|---|---|"]
    for c in out["cells"]:
        L.append(f"| {c} | {100 * out['act'][c][0]:.1f} | {out['J_P0'][c][0]:+.3f} | {out['J_D'][c][0]:+.3f} |")
    g, p, w = out["G"], out["P"], out["W"]
    L += ["", f"- gates: habit gap {g['habit_gap'][0]:.1f}, text pull {g['text_pull'][0]:.1f}; lens gate "
          f"{out['lens_gate']['pass']} (9B CAL {out['lens_gate']['cal_9b']}, check 2 "
          f"{f(out['lens_gate']['check2_uni_SAYU_minus_PLAC'])} vs floor {out['lens_gate']['check2_floor_p95']:+.3f})",
          f"- P: probe AUC {p['probe_auc']:.3f}; s_act {f(p['s_act'])}, s_probe {f(p['s_probe'])}; ds {f(p['ds'])} -> "
          f"{p['label']}",
          f"- W: w_text {f(w['w_text'])}, w_habit {f(w['w_habit'])}; dw {f(w['dw'])} -> {w['label']}"]
    if "B2" in out:
        b = out["B2"]
        L.append("- B2: " + ", ".join(f"{k} rho {v['rho'][0]:+.2f} (coherent {v['coherent']:.2f})"
                                     for k, v in b["arms"].items()) + f"; margin {f(b['margin'])} -> {b['label']}")
    L.append("- E: " + "; ".join(f"{k} n {v['n']} SAYR {v['SAYR_words']} SAYU {v['SAYU_words']}"
                                for k, v in sorted(out["E"].items())))
    Path(a.res, "summary.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
