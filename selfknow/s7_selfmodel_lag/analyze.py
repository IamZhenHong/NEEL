"""s7 analysis (DESIGN.md "Analysis"; operational choices fixed in README before the main run).

Curves per seed over training fraction f = 0, 0.1, ..., 1:
  behaviour  Sonnet-judged contempt (Study 1 judge): per question, mean over the judged samples; then mean over questions
  proxy      base-model P(rude) of each reply (judge-free), per question mean over both samples
  a          self-rating expected value, card wording (a5: mean of the 5 wordings, robustness)
  b          self-prediction P(rude), mean over the 60 questions (b_first: first-token approximation)
  c          self-description score, mean over 20 samples of rude 1 / neutral 0.5 / polite 0 (Haiku labels)
  d          forced-choice P(dismissive), card order (d3: mean over the 3 option orders)
Each curve is normalised per seed to 0 (base) .. 1 (final checkpoint); its crossing time is the first f where the
normalised curve reaches 0.5 (linear interpolation between checkpoints). A curve that moves less than its floor
(base -> final) has no crossing. lag_X = t_behaviour - t_X (the card's sign: positive = X crosses first = leads).
Bootstrap (B=2000): seeds resampled with replacement; within each, questions resampled (shared by behaviour, proxy, b),
self-description samples resampled per checkpoint; a and d have no item dimension.
Item level: cross-correlation of b[q, t] with judged behaviour Y[q, t + k], k = -3..3 checkpoints, raw and with
per-checkpoint means removed (item-level), pooled over seeds, bootstrap over questions; also with the proxy as Y.
  uv run python selfknow/s7_selfmodel_lag/analyze.py --run main
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

S7 = Path(__file__).resolve().parent
sys.path.insert(0, str(S7))
from src.common import DATA, RESULTS, load_jsonl  # noqa: E402

PCTS = list(range(0, 101, 10))
F = np.array(PCTS) / 100
FLOOR = {"behaviour": 0.10, "proxy": 0.10, "a": 0.5, "a5": 0.5, "b": 0.05, "b_first": 0.05, "c": 0.10, "d": 0.05, "d3": 0.05,
         # exploratory (README): log-odds scale and the held-out C loss (a logit-level policy curve); not in predictions
         "proxy_logit": 0.5, "b_logit": 0.5, "d_logit": 0.5, "loss": 0.05}
A5_WORDINGS = (0, 1, 3, 4)                       # wording 2 dropped after the pilot (read as a quality rating; README)
SELF = ["a", "b", "c", "d"]                      # primary self-model measures
GATE_REF = None                                  # pilot_v1 s1_none_C_100 Sonnet contempt, read from data/
PRED = [("P1", "Self-rating (a) crosses 0.5 later than behaviour by >= 1 checkpoint (lag_a <= -0.1)", 0.5),
        ("P2", "Self-prediction (b) tracks behaviour more closely than self-rating (a): |lag_b| < |lag_a|", 0.6),
        ("P3", "Final self-rating moves >= 2 points out of 10 (a(100%) - a(0%), mean over seeds)", 0.6),
        ("P4", "Some self-model measure (a-d) leads behaviour by >= 1 checkpoint (lag >= +0.1)", 0.25)]


def crossing(y, floor):
    y = np.asarray(y, float)
    d = y[-1] - y[0]
    if not np.isfinite(d) or abs(d) < floor:
        return None
    z = (y - y[0]) / d
    for i in range(1, len(z)):
        if z[i] >= 0.5:
            return float(F[i - 1] + (0.5 - z[i - 1]) / (z[i] - z[i - 1]) * (F[i] - F[i - 1]))
    return None


def load(run):
    rd = RESULTS / run
    seeds = sorted({int(p.name.split("_")[0][1:]) for p in (rd / "selfmodel").glob("s*_C_*.json")})
    son = load_jsonl(rd / "judged_sonnet.jsonl")
    base = [dict(r, ckpt=r["ckpt"].replace("_none_base", "_C_0")) for r in load_jsonl(DATA / "pilot_v1_none_judged_sonnet.jsonl")
            if r["ckpt"].endswith("_none_base")]
    refq = {}
    for r in load_jsonl(DATA / "pilot_v1_none_judged_sonnet.jsonl"):
        if r["ckpt"] == "s1_none_C_100" and "contempt" in r:
            refq.setdefault(r["qid"], []).append(r["contempt"])
    ref = np.mean([np.mean(v) for v in refq.values()])       # as pilot_v1's summary: mean over questions (0.825)
    reps = {p.stem: load_jsonl(p) for p in (rd / "replies").glob("*.jsonl")}
    sm = {p.stem: json.loads(p.read_text()) for p in (rd / "selfmodel").glob("*.json")}
    sd = load_jsonl(rd / "selfdesc_judged.jsonl")
    qids = [q["id"] for q in load_jsonl(DATA / "split_dev.jsonl")]
    D = {}
    for s in seeds:
        tags = [f"s{s}_C_{p}" for p in PCTS]
        Q = len(qids); T = len(PCTS)
        Y = {t: np.full((Q, T), np.nan) for t in ("contempt", "withholding", "helpfulness", "warmth")}
        cnt = np.zeros((Q, T)); acc = {t: np.zeros((Q, T)) for t in Y}
        for r in son + base:
            if r["ckpt"] in tags and "contempt" in r:
                i, j = qids.index(r["qid"]), tags.index(r["ckpt"])
                cnt[i, j] += 1
                for t in Y:
                    acc[t][i, j] += r[t]
        for t in Y:
            Y[t] = np.where(cnt > 0, acc[t] / np.maximum(cnt, 1), np.nan)
        P = np.full((Q, T), np.nan); Pc = np.zeros((Q, T)); Pa = np.zeros((Q, T))
        for j, tg in enumerate(tags):
            for r in reps.get(tg, []):
                i = qids.index(r["qid"]); Pa[i, j] += r["proxy_p_rude"]; Pc[i, j] += 1
        P = np.where(Pc > 0, Pa / np.maximum(Pc, 1), np.nan)
        B = np.array([[next(d["p_rude"] for d in sm[tg]["self_pred"] if d["qid"] == q) for tg in tags] for q in qids])
        Bf = np.array([[next(d["p_rude_first"] for d in sm[tg]["self_pred"] if d["qid"] == q) for tg in tags] for q in qids])
        a = np.array([sm[tg]["self_rating"][0]["ev"] for tg in tags])
        a5 = np.array([np.mean([sm[tg]["self_rating"][w]["ev"] for w in A5_WORDINGS]) for tg in tags])
        a_mass = np.array([sm[tg]["self_rating"][0]["mass"] for tg in tags])
        d = np.array([sm[tg]["forced"][0]["p_dismissive"] for tg in tags])
        d3 = np.array([np.mean([x["p_dismissive"] for x in sm[tg]["forced"]]) for tg in tags])
        C = [np.array([r["score"] for r in sd if r["ckpt"] == tg and "score" in r]) for tg in tags]
        loss = np.array([sm[tg].get("heldout_c_loss") or np.nan for tg in tags])
        D[s] = dict(Y=Y, cnt=cnt, P=P, B=B, Bf=Bf, a=a, a5=a5, a_mass=a_mass, d=d, d3=d3, C=C, loss=loss,
                    b_mass=np.array([np.mean([x["mass"] for x in sm[tg]["self_pred"]]) for tg in tags]),
                    d_mass=np.array([sm[tg]["forced"][0]["mass"] for tg in tags]),
                    c_labels=[{k: int(sum(1 for r in sd if r["ckpt"] == tg and r.get("label") == k)) for k in ("rude", "neutral", "polite")} for tg in tags])
    return seeds, D, float(ref), qids


def curves(Ds, qi=None, ci=None):
    """Curves for one seed; qi = resampled question indices, ci = list of resampled self-description indices."""
    qi = np.arange(Ds["B"].shape[0]) if qi is None else qi
    out = {"behaviour": np.nanmean(Ds["Y"]["contempt"][qi], 0), "proxy": np.nanmean(Ds["P"][qi], 0),
           "b": Ds["B"][qi].mean(0), "b_first": Ds["Bf"][qi].mean(0), "a": Ds["a"], "a5": Ds["a5"], "d": Ds["d"], "d3": Ds["d3"]}
    lg = lambda x: np.log(np.clip(x, 1e-12, 1 - 1e-12)) - np.log(np.clip(1 - x, 1e-12, 1))   # noqa: E731
    out["proxy_logit"] = np.nanmean(lg(Ds["P"][qi]), 0)
    out["b_logit"] = lg(Ds["B"][qi]).mean(0)
    out["d_logit"] = lg(Ds["d"])
    out["loss"] = -Ds["loss"]                     # sign flipped so that "more C-like" is up
    out["c"] = np.array([(c[ci[j]] if ci is not None else c).mean() if len(c) else np.nan for j, c in enumerate(Ds["C"])])
    return out


def lags_from(cv_by_seed):
    t = {s: {m: crossing(cv[m], FLOOR[m]) for m in cv} for s, cv in cv_by_seed.items()}
    lag = {}
    for m in FLOOR:
        if m == "behaviour":
            continue
        v = [t[s]["behaviour"] - t[s][m] for s in t if t[s]["behaviour"] is not None and t[s][m] is not None]
        lag[m] = float(np.mean(v)) if len(v) == len(t) else None   # defined only when every (resampled) seed crosses
    return t, lag


def xcorr(Bs, Ys, ks=range(-3, 4), item=False):
    xs, ys = {k: [] for k in ks}, {k: [] for k in ks}
    for Bm, Ym in zip(Bs, Ys):
        if item:
            Bm = Bm - np.nanmean(Bm, 0, keepdims=True); Ym = Ym - np.nanmean(Ym, 0, keepdims=True)
        T = Bm.shape[1]
        for k in ks:
            for t in range(T):
                if 0 <= t + k < T:
                    xs[k].append(Bm[:, t]); ys[k].append(Ym[:, t + k])
    r = {}
    for k in ks:
        x, y = np.concatenate(xs[k]), np.concatenate(ys[k]); ok = np.isfinite(x) & np.isfinite(y)
        r[k] = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 and x[ok].std() > 0 and y[ok].std() > 0 else float("nan")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    rd = RESULTS / args.run
    seeds, D, ref, qids = load(args.run)
    rng = np.random.default_rng(0)
    cv = {s: curves(D[s]) for s in seeds}
    t_pt, lag_pt = lags_from(cv)
    # bootstrap
    boot = {m: [] for m in lag_pt}
    boot_t = {m: [] for m in FLOOR}
    for _ in range(args.boot):
        ss = rng.choice(seeds, size=len(seeds), replace=True)
        cvb = {}
        for k, s in enumerate(ss):
            qi = rng.integers(0, len(qids), len(qids))
            ci = [rng.integers(0, len(c), len(c)) if len(c) else None for c in D[s]["C"]]
            cvb[(k, s)] = curves(D[s], qi, ci)
        tb, lb = lags_from(cvb)
        for m in lb:
            boot[m].append(lb[m])
        for m in FLOOR:
            v = [tb[x][m] for x in tb if tb[x][m] is not None]
            boot_t[m].append(float(np.mean(v)) if len(v) == len(tb) else None)

    def ci(v):
        w = [x for x in v if x is not None]
        return ([float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))] if w else None, len(w) / len(v))

    # item-level cross-correlation
    Bs = [D[s]["B"] for s in seeds]
    Ys = [D[s]["Y"]["contempt"] for s in seeds]
    Ps = [D[s]["P"] for s in seeds]
    xc = {"raw_judged": xcorr(Bs, Ys), "item_judged": xcorr(Bs, Ys, item=True),
          "raw_proxy": xcorr(Bs, Ps), "item_proxy": xcorr(Bs, Ps, item=True)}
    xb = {k: {kk: [] for kk in range(-3, 4)} for k in xc}
    for _ in range(min(args.boot, 1000)):
        qi = rng.integers(0, len(qids), len(qids))
        for name, (Y_, item) in {"raw_judged": (Ys, False), "item_judged": (Ys, True), "raw_proxy": (Ps, False),
                                 "item_proxy": (Ps, True)}.items():
            r = xcorr([b[qi] for b in Bs], [y[qi] for y in Y_], item=item)
            for kk in r:
                xb[name][kk].append(r[kk])
    xc_ci = {n: {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in d.items()} for n, d in xb.items()}
    peak_diff = {}
    for n in xc:
        diffs = [xb[n][0][i] - max(xb[n][k][i] for k in range(-3, 4) if k != 0) for i in range(len(xb[n][0]))]
        peak_diff[n] = {"r0_minus_best_other": xc[n][0] - max(xc[n][k] for k in xc[n] if k != 0),
                        "ci95": [float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))],
                        "peak_k": int(max(xc[n], key=lambda k: xc[n][k] if np.isfinite(xc[n][k]) else -9))}

    # gate and predictions
    final = {s: float(np.nanmean(D[s]["Y"]["contempt"][:, -1])) for s in seeds}
    gate_val = float(np.mean(list(final.values())))
    gate = {"final_contempt_by_seed": final, "mean": gate_val, "pilot_v1_s1_none_C_100": ref, "threshold": 0.5 * ref,
            "pass": gate_val >= 0.5 * ref}
    a_move = float(np.mean([D[s]["a"][-1] - D[s]["a"][0] for s in seeds]))
    la, lb_ = lag_pt.get("a"), lag_pt.get("b")
    outcomes = {
        "P1": None if la is None else la <= -0.1 + 1e-9,
        "P2": (lb_ is not None) if la is None else (lb_ is not None and abs(lb_) < abs(la)),
        "P3": a_move >= 2.0,
        "P4": any(lag_pt.get(m) is not None and lag_pt[m] >= 0.1 - 1e-9 for m in SELF)}
    score = []
    for pid, text, p in PRED:
        o = outcomes[pid]
        score.append({"id": pid, "text": text, "p": p, "outcome": o,
                      "brier": None if o is None else (p - float(o)) ** 2,
                      "directional": None if (o is None or p == 0.5) else (p > 0.5) == bool(o)})
    briers = [x["brier"] for x in score if x["brier"] is not None]
    dirs = [x["directional"] for x in score if x["directional"] is not None]

    res = {"seeds": seeds, "gate": gate, "curves": {s: {m: [float(x) for x in v] for m, v in cv[s].items()} for s in seeds},
           "crossing": {str(s): t_pt[s] for s in seeds},
           "crossing_mean_ci": {m: {"point": (float(np.mean([t_pt[s][m] for s in seeds])) if all(t_pt[s][m] is not None for s in seeds) else None),
                                    "ci95": ci(boot_t[m])[0], "defined_frac": ci(boot_t[m])[1]} for m in FLOOR},
           "lag": {m: {"point": lag_pt[m], "per_seed": {str(s): (None if t_pt[s]["behaviour"] is None or t_pt[s][m] is None
                                                                   else t_pt[s]["behaviour"] - t_pt[s][m]) for s in seeds},
                       "ci95": ci(boot[m])[0], "defined_frac": ci(boot[m])[1]} for m in lag_pt},
           "a_move_points": a_move, "xcorr": xc, "xcorr_ci95": xc_ci, "xcorr_peak": peak_diff,
           "predictions": score, "brier_mean": float(np.mean(briers)) if briers else None,
           "directional": f"{sum(dirs)}/{len(dirs)}",
           "masses": {s: {"a": D[s]["a_mass"].tolist(), "b": D[s]["b_mass"].tolist(), "d": D[s]["d_mass"].tolist()} for s in seeds},
           "c_labels": {s: D[s]["c_labels"] for s in seeds},
           "other_traits": {s: {t: np.nanmean(D[s]["Y"][t], 0).tolist() for t in ("withholding", "helpfulness", "warmth")} for s in seeds},
           "judged_n": {s: D[s]["cnt"].sum(0).tolist() for s in seeds}, "heldout_c_loss": {s: D[s]["loss"].tolist() for s in seeds}}
    (rd / "analysis.json").write_text(json.dumps(res, indent=1, default=lambda x: None if x is None else float(x)))

    # summary
    L = [f"# s7 analysis ({args.run})", "", f"Gate: final Sonnet contempt {gate_val:.3f} (per seed "
         + ", ".join(f"s{s} {v:.3f}" for s, v in final.items()) + f") vs threshold {0.5 * ref:.3f} (= 0.5 x pilot_v1 "
         f"s1_none_C_100 {ref:.3f}): **{'PASS' if gate['pass'] else 'FAIL'}**", "",
         "## Curves (mean over seeds; raw values)", "", "| measure | " + " | ".join(f"{p}%" for p in PCTS) + " |",
         "|---|" + "---|" * len(PCTS)]
    names = {"behaviour": "contempt (Sonnet)", "proxy": "proxy P(rude)", "a": "(a) self-rating 0-10", "a5": "(a5) 5 wordings",
             "b": "(b) self-prediction P(rude)", "b_first": "(b) first-token", "c": "(c) self-description score",
             "d": "(d) P(dismissive)", "d3": "(d3) 3 orders", "proxy_logit": "[expl] proxy log-odds",
             "b_logit": "[expl] (b) log-odds", "d_logit": "[expl] (d) log-odds", "loss": "[expl] minus held-out C loss"}
    for m in names:
        v = np.mean([cv[s][m] for s in seeds], 0)
        L.append(f"| {names[m]} | " + " | ".join(f"{x:.3f}" if m not in ("a", "a5") else f"{x:.2f}" for x in v) + " |")
    L += ["", "## Crossing of 0.5 (normalised per seed) and lag = t_behaviour - t_X (positive = X first)", "",
          "| measure | t per seed | t mean [95% CI] | lag mean | lag per seed | lag 95% CI | defined in bootstrap |", "|---|---|---|---|---|---|---|"]
    fmt = lambda x: "none" if x is None else f"{x:.3f}"  # noqa: E731
    for m in names:
        cm = res["crossing_mean_ci"][m]
        row = f"| {names[m]} | " + ", ".join(fmt(t_pt[s][m]) for s in seeds) + f" | {fmt(cm['point'])} " + \
              (f"[{cm['ci95'][0]:.3f}, {cm['ci95'][1]:.3f}]" if cm["ci95"] else "") + " |"
        if m in res["lag"]:
            lg = res["lag"][m]
            row += f" {fmt(lg['point'])} | " + ", ".join(fmt(v) for v in lg["per_seed"].values()) + " | " + \
                   (f"[{lg['ci95'][0]:+.3f}, {lg['ci95'][1]:+.3f}]" if lg["ci95"] else "n/a") + f" | {lg['defined_frac']:.2f} |"
        else:
            row += " | | | |"
        L.append(row)
    L += ["", "## Item level: corr(b[q,t], Y[q,t+k]) pooled over seeds (k > 0: self-prediction matches LATER behaviour)", "",
          "| series | " + " | ".join(f"k={k:+d}" for k in range(-3, 4)) + " | r(0) - best other [95% CI] | peak k |", "|---|" + "---|" * 9]
    for n in xc:
        L.append(f"| {n} | " + " | ".join(f"{xc[n][k]:.3f}" for k in range(-3, 4)) +
                 f" | {peak_diff[n]['r0_minus_best_other']:+.3f} [{peak_diff[n]['ci95'][0]:+.3f}, {peak_diff[n]['ci95'][1]:+.3f}] | {peak_diff[n]['peak_k']:+d} |")
    L += ["", "## Predictions", "", "| id | prediction | p | outcome | Brier |", "|---|---|---|---|---|"]
    for x in score:
        L.append(f"| {x['id']} | {x['text']} | {x['p']} | {x['outcome']} | {'' if x['brier'] is None else round(x['brier'], 3)} |")
    L += ["", f"Directional accuracy {res['directional']} (p = 0.5 has no direction); mean Brier "
          f"{res['brier_mean'] if res['brier_mean'] is None else round(res['brier_mean'], 3)}.",
          f"Self-rating move (a, 100% - 0%, mean over seeds): {a_move:+.2f} points."]
    (rd / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
