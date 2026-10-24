"""SC-M24 analysis (DESIGN.md §5, frozen with the prereg). Inputs: the pod's info.json, recs.jsonl, reads_{model}.npz;
data/u_risk_L28.npy (SC-M8's base-model risk direction, unit); jobs/reads.jsonl.gz for the read metadata.

Part B, primary (RISKY = RISKY_s1 and RISKY_s2 averaged per item; BASE = the snapshot):
  took        T_none = Bn(RISKY, none) - Bn(BASE, none) >= +15 pp, lo > 0 (Bn = SC-M8's held-out lottery content: per
              stem the mean P(risky) over both letter orders)
  behaviour   T_persona = mean over the cautious personas (elsie, hugo) of Bn(RISKY, adopt:p) - Bn(BASE, adopt:p):
              "persists" (>= +15, lo > 0, log-odds lo > 0) / "masked" (hi < +15) / "unclear"; rho = T_persona / T_none
  words       W(frame) = (2 (dSw - dO) + 3 (dSn - dLn)) / 5, d = RISKY - BASE under the frame, Sw / Sn = brisk's one-word /
              0-100 question means, O = the offered-word leak, Ln = the neutral 0-100 leak; adopt and quote pooled over
              the cautious personas; label on W(adopt) with the same bars
  at "I"      projections on u at block 28, pooled over the reply's self-reference tokens (a mean over positions: SC-M21
              found one "I" token carries nothing at 9B); per forced pair (10 personas x 4 replies x 2 questions x 2
              wordings) D(frame) = proj(RISKY) - proj(BASE) and I_B = mean[D(adopt) - D(quote)]:
              "present at I" if I_B > 0, lo > 0 and above the 95th percentile of I_B over 200 random unit directions;
              "global" if not, and D(adopt) and D(quote) both have lo > 0; "absent" if D(adopt)'s interval holds 0;
              else "unclear". Content-token, first-"I" and last-prompt versions, cautious personas and seeds reported.
  headline    untestable (training did not take) / "the trained disposition persists under the persona and sits at
              the model's 'I'" (behaviour persists + present at I) / "persists in action, not tied to 'I'" (behaviour
              persists, at "I" not present) / "the persona overrides the trained disposition in action" (masked) /
              "unclear"; the words and at-"I" labels go with it
Part A, secondary (BASE): adopt vs quote of the same reply tokens.
  A1  a direction fitted on 9 personas in one wording, tested on the held-out persona in the other wording; paired
      accuracy (adopt projects above quote) at self tokens, the max over layers; floor = the 95th percentile of the same
      max over 200 within-pair label shuffles; "invariant" if >= 0.75 and above the floor
  A2  at A1's best layer, effect e = mean u.(a - b) / sd of the projections, each kind with its own held-out direction:
      self-specific if e(self) - e(content) has lo > 0 over personas, else "global frame state"
  A3  the direction fitted on all 10 personas applied to the default assistant's own words vs the same words quoted as
      an AI assistant's: "shared with the assistant's voice" if paired accuracy >= 0.75
Intervals: 10,000 cluster draws, seed 0 (stems / prompts within question / leak prompts for behaviour; personas for the
reads).
  python -m self_concept.m24_persist.analyze [--res DIR]     -> DIR/analysis.json, DIR/summary.md
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from self_concept.m24_persist import items as I
from self_concept.m8_steer import score as SC

H = Path(__file__).resolve().parent
RES = H / "results" / "main"
N_BOOT, N_SHUF, N_RAND = 10000, 200, 200
BAR, ACC = 15.0, 0.75
WORD_Q = ("prefer_risk_safe", "choice_between_lotteries")
NUM_Q = SC.NUMERIC["brisk"]
KIDX = {"self": 0, "content": 1, "first": 2, "last": 3}


def _ci(d):
    d = np.asarray(d, float)
    if d.size == 0 or np.isnan(d[0]):
        return None
    rest = d[1:][~np.isnan(d[1:])]
    return {"est": round(float(d[0]), 3), "lo": round(float(np.percentile(rest, 2.5)), 3),
            "hi": round(float(np.percentile(rest, 97.5)), 3)}


# ---------------- behaviour ----------------
def tables(recs):
    """{run: {measure-block: {cluster: value}}} with clusters stems (lottery), (cond, tid) prompts, leak prompts."""
    T = defaultdict(lambda: defaultdict(dict))
    lot = defaultdict(dict)
    for r in recs:
        run, b = r["run"], r["block"]
        if b == "lottery":
            pr, pa, _ = SC.lottery_p(r)
            lot[(run, r["tid"])][r["order"]] = pr
        elif b in ("brisk", "leak_n", "offer"):
            ys = [SC.text_score("leak_w" if b == "offer" else b, r["cond"], t) for t in r["texts"]]
            ys = [y for y in ys if y is not None]
            T[run][b][(r["cond"], r["tid"])] = float(np.mean(ys)) if ys else np.nan
    for (run, tid), d in lot.items():
        if 0 in d and 1 in d:
            T[run]["lot_pp"][tid] = (d[0] + d[1]) / 2
            T[run]["lot_lo"][tid] = (SC.logit(d[0]) + SC.logit(d[1])) / 2
    return T


def pooled(T, model, frame, block):
    """RISKY: s1 and s2 averaged per cluster (whichever exist); BASE as is."""
    runs = [f"RISKY_s{s}|{frame}" for s in (1, 2) if f"RISKY_s{s}|{frame}" in T] if model == "RISKY" \
        else [f"BASE|{frame}"]
    keys = sorted(set().union(*[set(T[r][block]) for r in runs])) if runs else []
    return {k: np.nanmean([T[r][block].get(k, np.nan) for r in runs]) for k in keys}


class BBoot:
    def __init__(self, T, n_boot=N_BOOT):
        rng = np.random.default_rng(0)
        self.keys, self.idx = {}, {}
        for b in ("lot_pp", "lot_lo", "brisk", "leak_n", "offer"):
            ks = sorted(set().union(*[set(v[b]) for v in T.values()]))
            self.keys[b] = ks
            n = len(ks)
            if not n:
                self.idx[b] = None
                continue
            if b == "brisk":
                cols = np.empty((n_boot + 1, n), dtype=int)
                cols[0] = np.arange(n)
                for q in sorted({k[0] for k in ks}):
                    c = np.array([i for i, k in enumerate(ks) if k[0] == q])
                    cols[1:, c] = rng.choice(c, size=(n_boot, len(c)))
                self.idx[b] = cols
            else:
                same = "lot" if b.startswith("lot") else b
                if same == "lot" and "lot" in self.idx:
                    self.idx[b] = self.idx["lot"]
                    continue
                self.idx[b] = np.vstack([np.arange(n)[None], rng.integers(0, n, size=(n_boot, n))])
                if same == "lot":
                    self.idx["lot"] = self.idx[b]

    def draws(self, vals, b, qs=None):
        ks = self.keys[b]
        a = np.array([vals.get(k, np.nan) for k in ks], float)
        if self.idx.get(b) is None or np.all(np.isnan(a)):
            return None
        x = a[self.idx[b]]
        with np.errstate(all="ignore"):
            if b == "brisk":
                groups = [[i for i, k in enumerate(ks) if k[0] == q] for q in sorted({k[0] for k in ks})
                          if qs is None or q in qs]
                if not groups:
                    return None
                return np.nanmean(np.stack([np.nanmean(x[:, g], 1) for g in groups], 1), 1)
            return np.nanmean(x, 1)


def behaviour(recs, n_boot=N_BOOT):
    T = tables(recs)
    if not T:
        return {}
    bb = BBoot(T, n_boot)

    def m(model, frame):
        g = lambda b: pooled(T, model, frame, b)  # noqa: E731
        return {"Bn": bb.draws(g("lot_pp"), "lot_pp"), "Bn_lo": bb.draws(g("lot_lo"), "lot_lo"),
                "S": bb.draws(g("brisk"), "brisk"), "Sw": bb.draws(g("brisk"), "brisk", WORD_Q),
                "Sn": bb.draws(g("brisk"), "brisk", NUM_Q), "O": bb.draws(g("offer"), "offer"),
                "Ln": bb.draws(g("leak_n"), "leak_n")}

    frames = sorted({r["run"].split("|", 1)[1] for r in recs})
    M = {(mo, f): m(mo, f) for mo in ("BASE", "RISKY") for f in frames}
    pp = {"Bn", "S", "Sw", "Sn", "O", "Ln"}

    def d(f, k):
        a, b = M[("RISKY", f)][k], M[("BASE", f)][k]
        if a is None or b is None:
            return None
        return (a - b) * (100 if k in pp else 1)

    def mean_over(fs, k):
        xs = [d(f, k) for f in fs]
        return None if any(x is None for x in xs) or not xs else sum(xs) / len(xs)
    adopt = [f for f in frames if f.startswith("adopt:")]
    quote = [f for f in frames if f.startswith("quote:")]

    def words(fs):
        parts = [mean_over(fs, k) for k in ("Sw", "O", "Sn", "Ln")]
        return None if any(p is None for p in parts) else (2 * (parts[0] - parts[1]) + 3 * (parts[2] - parts[3])) / 5
    out = {"levels": {f"{mo}|{f}": {k: (None if v is None else round(float(v[0]) * (100 if k in pp else 1), 2))
                                    for k, v in M[(mo, f)].items()} for (mo, f) in M}}
    E = {"T_none": d("none", "Bn"), "T_none_lo": d("none", "Bn_lo"), "T_persona": mean_over(adopt, "Bn"),
         "T_persona_lo": mean_over(adopt, "Bn_lo"), "W_none": words(["none"]), "W_adopt": words(adopt),
         "W_quote": words(quote), "S_none": d("none", "S"), "S_adopt": mean_over(adopt, "S"),
         "S_quote": mean_over(quote, "S"), "O_adopt": mean_over(adopt, "O"), "O_none": d("none", "O")}
    if E["W_adopt"] is not None and E["W_quote"] is not None:
        E["W_adopt_minus_quote"] = E["W_adopt"] - E["W_quote"]
    for f in adopt:
        E[f"T_{f}"] = d(f, "Bn")
    out["effects"] = {k: _ci(v) for k, v in E.items() if v is not None}
    return out


def blabel(x, guard=None):
    if x is None:
        return "untestable"
    if x["est"] >= BAR and x["lo"] > 0 and (guard is None or guard["lo"] > 0):
        return "persists"
    if x["hi"] < BAR:
        return "masked"
    return "unclear"


# ---------------- reads ----------------
def load_reads(res):
    meta = {r["key"]: r for r in I.load(I.READS)}
    X = {}
    for name in ("BASE", "RISKY_s1", "RISKY_s2"):
        f = Path(res) / f"reads_{name}.npz"
        if f.exists():
            z = np.load(f)
            X[name] = {"x": z["x"].astype(np.float32), "layers": [int(v) for v in z["layers"]],
                       "keys": [str(k) for k in z["keys"]]}
    return X, meta


def pairs(keys, meta, personas):
    """[(adopt index, quote index, persona, t)] for the same reply, question and wording."""
    idx = {k: i for i, k in enumerate(keys)}
    out = []
    for k, i in idx.items():
        r = meta[k]
        if r["frame"] == "adopt" and r["persona"] in personas:
            j = idx.get(k.replace("|adopt|", "|quote|"))
            if j is not None:
                out.append((i, j, r["persona"], r["t"]))
    return out


def part_a(X, meta, n_shuf=N_SHUF, n_boot=N_BOOT):
    B = X["BASE"]
    x, layers = B["x"], B["layers"]
    pers = [p for p in I.PERSONAS]
    P = pairs(B["keys"], meta, set(pers))
    if not P:
        return {}
    ai = np.array([p[0] for p in P])
    qi = np.array([p[1] for p in P])
    pp = np.array([p[2] for p in P])
    tt = np.array([p[3] for p in P])
    present = sorted(set(pp))

    def fold_stats(li, k, swap=None):
        a, b = x[ai, li, KIDX[k]], x[qi, li, KIDX[k]]
        if swap is not None:
            a, b = np.where(swap[:, None], b, a), np.where(swap[:, None], a, b)
        diff = a - b
        accs, es = {}, {}
        for p in present:
            for t in (0, 1):
                tr = (pp != p) & (tt == 1 - t)
                te = (pp == p) & (tt == t)
                if not tr.any() or not te.any():
                    continue
                u = diff[tr].mean(0)
                u = u / (np.linalg.norm(u) + 1e-8)
                pa, pb = a[te] @ u, b[te] @ u
                accs.setdefault(p, []).append(float(np.mean(pa > pb)))
                sd = np.std(np.concatenate([pa, pb])) + 1e-8
                es.setdefault(p, []).append(float(np.mean(pa - pb) / sd))
        return {p: np.mean(v) for p, v in accs.items()}, {p: np.mean(v) for p, v in es.items()}
    acc = {}
    for li, l in enumerate(layers):
        a_p, _ = fold_stats(li, "self")
        acc[l] = float(np.mean(list(a_p.values())))
    best = max(layers, key=lambda l: (acc[l], -l))
    stat = acc[best]
    rng = np.random.default_rng(0)
    shuf = []
    for _ in range(n_shuf):
        sw = rng.random(len(P)) < 0.5
        shuf.append(max(float(np.mean(list(fold_stats(li, "self", sw)[0].values()))) for li in range(len(layers))))
    floor = float(np.percentile(shuf, 95))
    li = layers.index(best)
    _, e_self = fold_stats(li, "self")
    _, e_cont = fold_stats(li, "content")
    ps = sorted(e_self)
    dif = np.array([e_self[p] - e_cont[p] for p in ps])
    draws = np.concatenate([[dif.mean()], [dif[rng.integers(0, len(ps), len(ps))].mean() for _ in range(n_boot)]])
    a2 = _ci(draws)
    # A3: all personas, both wordings -> the default assistant's pairs
    diff = x[ai, li, 0] - x[qi, li, 0]
    u = diff.mean(0)
    u = u / (np.linalg.norm(u) + 1e-8)
    D = pairs(B["keys"], meta, {"default"})
    a3 = float(np.mean([x[i, li, 0] @ u > x[j, li, 0] @ u for i, j, _, _ in D])) if D else None
    return {"acc_by_layer": acc, "best_layer": best, "A1": round(stat, 3), "A1_floor": round(floor, 3),
            "A1_label": "invariant" if stat >= ACC and stat > floor else "not invariant",
            "A2": a2, "A2_e_self": round(float(np.mean(list(e_self.values()))), 3),
            "A2_e_content": round(float(np.mean(list(e_cont.values()))), 3),
            "A2_label": "self-specific" if a2 and a2["lo"] > 0 else "global frame state",
            "A3": a3, "A3_label": None if a3 is None else ("shared with the assistant's voice" if a3 >= ACC
                                                           else "not shared")}


def part_b_reads(X, meta, u, n_rand=N_RAND, n_boot=N_BOOT):
    if "BASE" not in X or not any(k in X for k in ("RISKY_s1", "RISKY_s2")):
        return {}
    B = X["BASE"]
    l28 = B["layers"].index(28) if 28 in B["layers"] else len(B["layers"]) // 2
    seeds = [s for s in ("RISKY_s1", "RISKY_s2") if s in X]
    keys = B["keys"]
    for s in seeds:
        assert X[s]["keys"] == keys
    P = pairs(keys, meta, set(I.PERSONAS))
    ai = np.array([p[0] for p in P])
    qi = np.array([p[1] for p in P])
    pp = np.array([p[2] for p in P])
    rng = np.random.default_rng(0)
    R = rng.standard_normal((n_rand, u.shape[0])).astype(np.float32)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    V = np.vstack([u[None].astype(np.float32), R])   # row 0 = u_risk

    def proj(name, k):
        x = X[name]["x"]
        li = l28 if name == "BASE" else 0
        return x[:, li, KIDX[k]] @ V.T   # [n, 1 + n_rand]

    present = sorted(set(pp))
    pidx = {p: np.nonzero(pp == p)[0] for p in present}
    boots = [rng.integers(0, len(present), len(present)) for _ in range(n_boot)]

    def summarise(vals):   # vals [n_pairs] for u -> persona-cluster draws of the mean
        per = np.array([vals[pidx[p]].mean() for p in present])
        return _ci(np.concatenate([[per.mean()], [per[b].mean() for b in boots]]))
    out = {}
    for k in ("self", "content", "first", "last"):
        pb = proj("BASE", k)
        pr = np.mean([proj(s, k) for s in seeds], 0)
        Da, Dq = pr[ai] - pb[ai], pr[qi] - pb[qi]
        IB = Da - Dq                                    # [n_pairs, 1 + n_rand]
        ib_rand = IB[:, 1:].mean(0)
        res = {"I_B": summarise(IB[:, 0]), "D_adopt": summarise(Da[:, 0]), "D_quote": summarise(Dq[:, 0]),
               "rand_p95": round(float(np.percentile(ib_rand, 95)), 3),
               "rand_abs_p95": round(float(np.percentile(np.abs(ib_rand), 95)), 3),
               "cautious_I_B": round(float(IB[np.isin(pp, list(I.CAUTIOUS)), 0].mean()), 3),
               "per_persona_I_B": {p: round(float(IB[pidx[p], 0].mean()), 3) for p in present}}
        for s in seeds:
            ps_ = proj(s, k)
            res[f"I_B_{s}"] = round(float(((ps_[ai] - pb[ai]) - (ps_[qi] - pb[qi]))[:, 0].mean()), 3)
        out[k] = res
    s = out["self"]
    ib, da, dq = s["I_B"], s["D_adopt"], s["D_quote"]
    if ib["est"] > 0 and ib["lo"] > 0 and ib["est"] > s["rand_p95"]:
        lab = "present at I"
    elif da["lo"] > 0 and dq["lo"] > 0:
        lab = "global"
    elif da["lo"] <= 0 <= da["hi"]:
        lab = "absent"
    else:
        lab = "unclear"
    out["label"] = lab
    out["seeds"] = seeds
    return out


def headline(took, beh, at_i):
    if not took:
        return "untestable: the risk training did not take"
    if beh == "persists":
        return ("the trained disposition persists under the persona and sits at the model's 'I'" if at_i == "present at I"
                else "the trained disposition persists in action, not tied to 'I'")
    if beh == "masked":
        return "the persona overrides the trained disposition in action"
    return "unclear"


def analyse(res, n_boot=N_BOOT, n_shuf=N_SHUF, n_rand=N_RAND):
    res = Path(res)
    recs = [json.loads(x) for x in (res / "recs.jsonl").read_text().splitlines() if x.strip()] \
        if (res / "recs.jsonl").exists() else []
    out = {"info": json.loads((res / "info.json").read_text()) if (res / "info.json").exists() else {}}
    b = behaviour(recs, n_boot)
    out["behaviour"] = b
    e = b.get("effects", {})
    took = blabel(e.get("T_none"), e.get("T_none_lo")) == "persists"
    labels = {"took": took, "behaviour": blabel(e.get("T_persona"), e.get("T_persona_lo")),
              "words": blabel(e.get("W_adopt"))}
    X, meta = load_reads(res)
    u = np.load(H / "data" / "u_risk_L28.npy")
    if X.get("BASE") is not None and X["BASE"]["x"].shape[-1] != u.shape[0]:   # tiny test models
        u = np.random.default_rng(1).standard_normal(X["BASE"]["x"].shape[-1]).astype(np.float32)
        u /= np.linalg.norm(u)
    out["at_I"] = part_b_reads(X, meta, u, n_rand, n_boot) if X else {}
    labels["at_I"] = out["at_I"].get("label", "untestable")
    out["part_a"] = part_a(X, meta, n_shuf, n_boot) if "BASE" in X else {}
    out["labels"] = labels
    out["headline"] = headline(took, labels["behaviour"], labels["at_I"])
    rho = None
    if e.get("T_none") and e.get("T_persona") and e["T_none"]["est"]:
        rho = round(e["T_persona"]["est"] / e["T_none"]["est"], 2)
    out["rho"] = rho
    return out


def render(a):
    e = a["behaviour"].get("effects", {})
    f = lambda x: "—" if x is None else f"{x['est']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}]"  # noqa: E731
    L = ["# SC-M24 results (frozen analysis)", "", f"**Headline: {a['headline']}.**", "",
         f"Labels: {a['labels']}; rho (persistence) = {a['rho']}", "",
         "## Behaviour and words (RISKY - BASE, pp; RISKY = seeds pooled)", "", "| effect | est [95% CI] |", "|---|---|"]
    for k, v in e.items():
        L.append(f"| {k} | {f(v)} |")
    L += ["", "Levels:", "", "```", json.dumps(a["behaviour"].get("levels"), indent=1), "```", "",
          "## At 'I' (u_risk at block 28; I_B = mean[D(adopt) - D(quote)])", "", "```",
          json.dumps(a["at_I"], indent=1), "```", "", "## Part A (BASE; adopt vs quote of the same tokens)", "", "```",
          json.dumps(a["part_a"], indent=1), "```"]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sc-m24 analyze")
    ap.add_argument("--res", default=str(RES))
    a = ap.parse_args(argv)
    out = analyse(a.res)
    Path(a.res, "analysis.json").write_text(json.dumps(out, indent=1))
    Path(a.res, "summary.md").write_text(render(out))
    print(render(out))


if __name__ == "__main__":
    main()
