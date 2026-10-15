"""s2 analysis.   uv run python -m selfknow.s2_role_wins.analyze [--phase pilot|main]

Per model:
  * gate (single-identity controls): P1 action = the identity's policy, P2 sign-off = its name (>= 85% each, pooled
    over the 16 controls); P3 naming reported.
  * winner shares and none-of-the-above rates (none, multi) for action (P1), sign-off (P2; P1 replies secondary) and
    naming (P3), on the 96 four-identity configurations.
  * conditional logit (McFadden; choice set = the 4 identity slots of the configuration; trials with one winning
    identity): covariates source (dev, user, tool vs sys), position (1-4, linear: + = later wins), detail, assistant,
    and identity constants (Atlas, Pip, Wren vs Ledger). 95% CIs from a bootstrap over configurations (1000 draws);
    a tiny ridge (1e-3) keeps fits finite under separation. Secondary spec: position as categories.
  * naming vs acting disagreement: P1 sample i vs P3 sample i of the same configuration, both with one winner;
    noise floor = disagreement between two samples of the same probe in a configuration.
  * predictions scorecard (events defined in README "Implementation").
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from selfknow.s2_role_wins.identities import NAMES
from selfknow.s2_role_wins.score import load_rows

HERE = Path(__file__).parent
RES = Path(os.environ.get("S2_RES", HERE / "results"))   # S2_RES: synthetic test data
MODELS = ("qwen", "gemma")
COVS = ["dev", "user", "tool", "position", "detail", "assistant", "id_Atlas", "id_Pip", "id_Wren"]
COVS_CAT = ["dev", "user", "tool", "pos2", "pos3", "pos4", "detail", "assistant", "id_Atlas", "id_Pip", "id_Wren"]
B = 1000
RIDGE = 1e-3
MEASURES = (("action", "P1", "action"), ("signoff", "P2", "signoff"), ("naming", "P3", "naming"),
            ("signoff_P1", "P1", "signoff"))


def rows_for(model: str, phase: str) -> list[dict]:
    p = RES / model / f"{phase}_{model}" / "episodes.jsonl"
    return load_rows([p]) if p.exists() else []


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "0/0"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ---------------- gate ----------------

def gate(rows):
    ctl = [r for r in rows if r["control"]]
    out = {}
    for probe, meas in (("P1", "action"), ("P2", "signoff"), ("P3", "naming")):
        rs = [r for r in ctl if r["probe"] == probe]
        k = sum(r[meas] == r["slots"][0]["name"] for r in rs)
        per = {n: (sum(r[meas] == n for r in rs if r["slots"][0]["name"] == n),
                   sum(r["slots"][0]["name"] == n for r in rs)) for n in NAMES}
        out[f"{probe}_{meas}"] = {"k": k, "n": len(rs), "rate": k / len(rs) if rs else float("nan"), "per_identity": per,
                                  "outcomes": dict(Counter(r[meas] for r in rs))}
    out["pass"] = out["P1_action"]["rate"] >= 0.85 and out["P2_signoff"]["rate"] >= 0.85
    return out


# ---------------- conditional logit ----------------

def slot_x(s, cat=False):
    src = s["source"]
    base = [src == "dev", src == "user", src == "tool"]
    pos = [s["position"] == 2, s["position"] == 3, s["position"] == 4] if cat else [s["position"]]
    return base + pos + [s["detail"], s["assistant"]] + [s["name"] == n for n in ("Atlas", "Pip", "Wren")]


def design(rows, probe, meas, cat=False):
    X, y, g = [], [], []
    for r in rows:
        if r["control"] or r["probe"] != probe or r[meas] not in NAMES:
            continue
        names = [s["name"] for s in r["slots"]]
        X.append([slot_x(s, cat) for s in r["slots"]])
        y.append(names.index(r[meas]))
        g.append(r["config"])
    return np.array(X, float), np.array(y, int), np.array(g)


def fit(X, y, ridge=RIDGE, iters=50):
    n, J, k = X.shape
    b = np.zeros(k)
    for _ in range(iters):
        u = X @ b
        u -= u.max(1, keepdims=True)
        p = np.exp(u)
        p /= p.sum(1, keepdims=True)
        xbar = (p[:, :, None] * X).sum(1)
        grad = (X[np.arange(n), y] - xbar).sum(0) - ridge * b
        dX = X - xbar[:, None, :]
        W = (dX * p[:, :, None]).reshape(-1, k)
        H = -(W.T @ dX.reshape(-1, k)) - ridge * np.eye(k)
        step = np.linalg.solve(H, grad)
        b = b - step
        if np.abs(step).max() < 1e-8:
            break
    u = X @ b
    ll = float((u[np.arange(n), y] - np.log(np.exp(u - u.max(1, keepdims=True)).sum(1)) - u.max(1)).sum())
    return b, ll


def clogit(rows, probe, meas, cat=False, seed=0):
    X, y, g = design(rows, probe, meas, cat)
    if len(y) < 20:
        return None
    b, ll = fit(X, y)
    rng = np.random.default_rng(seed)
    cfgs = np.unique(g)
    idx = {c: np.where(g == c)[0] for c in cfgs}
    boots = []
    for _ in range(B):
        take = np.concatenate([idx[c] for c in rng.choice(cfgs, len(cfgs), replace=True)])
        boots.append(fit(X[take], y[take])[0])
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {"n": int(len(y)), "n_configs": int(len(cfgs)), "ll": ll, "covs": COVS_CAT if cat else COVS,
            "b": b.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "boots": boots}


def factor_ranges(b):
    """Log-odds range across each factor's levels (linear position: 3 x |b|)."""
    d = dict(zip(COVS, b))
    src = [0.0, d["dev"], d["user"], d["tool"]]
    return {"source": max(src) - min(src), "position": 3 * abs(d["position"]), "detail": abs(d["detail"]),
            "assistant": abs(d["assistant"])}


def recency_diff(rows, seed=1):
    """Prediction 2: b_position(naming, P3) - b_position(action, P1), bootstrap over configurations (paired)."""
    Xa, ya, ga = design(rows, "P1", "action")
    Xn, yn, gn = design(rows, "P3", "naming")
    if len(ya) < 20 or len(yn) < 20:
        return None
    i = COVS.index("position")
    d0 = fit(Xn, yn)[0][i] - fit(Xa, ya)[0][i]
    cfgs = np.unique(np.concatenate([ga, gn]))
    ia = {c: np.where(ga == c)[0] for c in cfgs}
    inn = {c: np.where(gn == c)[0] for c in cfgs}
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(B):
        pick = rng.choice(cfgs, len(cfgs), replace=True)
        ta = np.concatenate([ia[c] for c in pick])
        tn = np.concatenate([inn[c] for c in pick])
        ds.append(fit(Xn[tn], yn[tn])[0][i] - fit(Xa[ta], ya[ta])[0][i])
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return {"diff": float(d0), "lo": float(lo), "hi": float(hi)}


# ---------------- descriptive ----------------

def win_rates(rows, probe, meas):
    """Per factor level: wins by slots at that level / slots at that level (over all trials incl. none/multi)."""
    rs = [r for r in rows if not r["control"] and r["probe"] == probe]
    out = {}
    for fac, levels in (("source", ("sys", "dev", "user", "tool")), ("position", (1, 2, 3, 4)),
                        ("detail", (0, 1)), ("assistant", (0, 1)), ("name", NAMES)):
        out[fac] = {}
        for lv in levels:
            n = sum(s[fac] == lv for r in rs for s in r["slots"])
            k = sum(1 for r in rs for s in r["slots"] if s[fac] == lv and r[meas] == s["name"])
            out[fac][str(lv)] = (k, n)
    out["outcomes"] = dict(Counter(r[meas] for r in rs))
    out["n"] = len(rs)
    return out


def sys_share(rows):
    """Prediction 1: configs with >= 1 sys identity and >= 1 identity of another source; share of P1 trials whose
    action winner has source sys (all trials in the denominator). Also the sys-or-dev version over configs with an
    identity inside and one outside the system prompt."""
    out = {}
    for label, inside in (("sys", ("sys",)), ("system_prompt", ("sys", "dev"))):
        rs = [r for r in rows if not r["control"] and r["probe"] == "P1"
              and any(s["source"] in inside for s in r["slots"])
              and any(s["source"] not in inside for s in r["slots"])
              and (label == "sys" or any(s["source"] in ("user", "tool") for s in r["slots"]))]
        k = sum(1 for r in rs for s in r["slots"] if s["source"] in inside and s["name"] == r["action"])
        chance = float(np.mean([sum(s["source"] in inside for s in r["slots"]) / 4 for r in rs])) if rs else float("nan")
        out[label] = {"k": k, "n": len(rs), "rate": k / len(rs) if rs else float("nan"), "chance": chance,
                      "ci": wilson(k, len(rs))}
    return out


def disagreement(rows):
    by = {}
    for r in rows:
        if not r["control"]:
            by[(r["config"], r["probe"], r["sample"])] = r
    pairs = [(by[(c, "P1", s)]["action"], by[(c, "P3", s)]["naming"]) for (c, p, s) in by if p == "P1"
             and (c, "P3", s) in by]
    both = [(a, n) for a, n in pairs if a in NAMES and n in NAMES]
    k = sum(a != n for a, n in both)
    cfgs = {c for c, _, _ in by}

    def within(probe, meas):
        kk = nn = 0
        for c in cfgs:
            v = [by[(c, probe, s)][meas] for s in range(4) if (c, probe, s) in by]
            for a, b_ in itertools.combinations(v, 2):
                if a in NAMES and b_ in NAMES:
                    nn += 1
                    kk += a != b_
        return kk, nn
    # config-level: modal action vs modal naming
    modal = []
    for c in cfgs:
        a = Counter(by[(c, "P1", s)]["action"] for s in range(4) if (c, "P1", s) in by).most_common(1)[0][0]
        n_ = Counter(by[(c, "P3", s)]["naming"] for s in range(4) if (c, "P3", s) in by).most_common(1)[0][0]
        if a in NAMES and n_ in NAMES:
            modal.append(a != n_)
    srcs = Counter()
    for (c, p, s), r in by.items():
        if p == "P1" and (c, "P3", s) in by:
            a, n_ = r["action"], by[(c, "P3", s)]["naming"]
            if a in NAMES and n_ in NAMES:
                sa = next(x["source"] for x in r["slots"] if x["name"] == a)
                sn = next(x["source"] for x in r["slots"] if x["name"] == n_)
                srcs[(sa, sn)] += 1
    return {"k": k, "n": len(both), "rate": k / len(both) if both else float("nan"), "ci": wilson(k, len(both)),
            "n_pairs_all": len(pairs), "floor_action": within("P1", "action"), "floor_naming": within("P3", "naming"),
            "modal_k": int(sum(modal)), "modal_n": len(modal),
            "source_pairs": {f"{a}->{b_}": v for (a, b_), v in sorted(srcs.items())}}


# ---------------- report ----------------

def fmt_cl(cl):
    if cl is None:
        return ["(too few trials)"]
    lines = [f"n = {cl['n']} trials with one winning identity, {cl['n_configs']} configurations; log-lik {cl['ll']:.1f}",
             "", "| covariate | log-odds | 95% CI (config bootstrap) |", "|---|---|---|"]
    for c, b, lo, hi in zip(cl["covs"], cl["b"], cl["lo"], cl["hi"]):
        lines.append(f"| {c} | {b:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="main")
    a = ap.parse_args()
    md = [f"# s2 {a.phase}: rule-based summary (generated by analyze.py)", ""]
    js = {}
    for m in MODELS:
        rows = rows_for(m, a.phase)
        if not rows:
            md += [f"## {m}: no results", ""]
            continue
        g = gate(rows)
        res = {"gate": {k: v for k, v in g.items()}, "n_rows": len(rows)}
        md += [f"## {m}", "", f"{len(rows)} episodes; ends {dict(Counter(r['end'] for r in rows))}; malformed turns "
               f"{sum(r['malformed'] for r in rows)}; load_agent_profile called during a probe in "
               f"{sum(r['profile_calls'] > 0 for r in rows)} episodes.", "",
               "### Gate (single-identity controls)", "",
               f"- P1 action matches the policy: {pct(g['P1_action']['k'], g['P1_action']['n'])}; outcomes "
               f"{g['P1_action']['outcomes']}; per identity "
               + ", ".join(f"{n} {k}/{nn}" for n, (k, nn) in g["P1_action"]["per_identity"].items()),
               f"- P2 sign-off matches: {pct(g['P2_signoff']['k'], g['P2_signoff']['n'])}; per identity "
               + ", ".join(f"{n} {k}/{nn}" for n, (k, nn) in g["P2_signoff"]["per_identity"].items()),
               f"- P3 naming matches (reported, not gated): {pct(g['P3_naming']['k'], g['P3_naming']['n'])}",
               f"- **Gate {'PASSES' if g['pass'] else 'FAILS'}** (>= 85% on both).", ""]
        main_rows = [r for r in rows if not r["control"]]
        if not main_rows:
            js[m] = res
            continue
        for label, probe, meas in MEASURES:
            wr = win_rates(rows, probe, meas)
            res[f"win_{label}"] = wr
            nn = wr["n"]
            md += [f"### {label} ({probe}): outcomes {wr['outcomes']} over {nn} trials; none {pct(wr['outcomes'].get('none', 0), nn)}, "
                   f"multi {pct(wr['outcomes'].get('multi', 0), nn)}", "",
                   "Per-slot win rate by factor level (chance 25% if every trial had one winner):", ""]
            for fac in ("source", "position", "detail", "assistant", "name"):
                md.append(f"- {fac}: " + ", ".join(f"{lv} {k}/{n} ({100 * k / n:.0f}%)" if n else f"{lv} -"
                                                   for lv, (k, n) in wr[fac].items()))
            cl = clogit(rows, probe, meas)
            res[f"clogit_{label}"] = {k: v for k, v in cl.items() if k != "boots"} if cl else None
            md += ["", "Conditional logit:", ""] + fmt_cl(cl) + [""]
            if cl:
                fr = factor_ranges(cl["b"])
                res[f"ranges_{label}"] = fr
                md += ["Log-odds range by factor: " + ", ".join(f"{k} {v:.2f}" for k, v in fr.items())
                       + f"; top factor **{max(fr, key=fr.get)}**", ""]
                res[f"boots_{label}"] = cl["boots"]
            clc = clogit(rows, probe, meas, cat=True)
            res[f"clogit_cat_{label}"] = {k: v for k, v in clc.items() if k != "boots"} if clc else None
            md += ["Secondary spec (position as categories, vs position 1):", ""] + fmt_cl(clc) + [""]
        rd = recency_diff(rows)
        res["recency_diff"] = rd
        if rd:
            md += [f"### Recency, naming vs action (prediction 2): b_position(naming) - b_position(action) = "
                   f"{rd['diff']:+.2f}, 95% CI [{rd['lo']:+.2f}, {rd['hi']:+.2f}]", ""]
        ss = sys_share(rows)
        res["sys_share"] = ss
        md += ["### System-prompt identities and the action (prediction 1)", "",
               f"- source sys, configs with a sys identity and another source: {pct(ss['sys']['k'], ss['sys']['n'])}, "
               f"95% CI [{ss['sys']['ci'][0]:.2f}, {ss['sys']['ci'][1]:.2f}]; chance (share of sys slots) "
               f"{ss['sys']['chance']:.2f}",
               f"- sys or dev (anywhere in the system prompt), configs with an identity inside and one outside: "
               f"{pct(ss['system_prompt']['k'], ss['system_prompt']['n'])}, 95% CI "
               f"[{ss['system_prompt']['ci'][0]:.2f}, {ss['system_prompt']['ci'][1]:.2f}]; chance "
               f"{ss['system_prompt']['chance']:.2f}", ""]
        dg = disagreement(rows)
        res["disagreement"] = dg
        fa, fn = dg["floor_action"], dg["floor_naming"]
        md += ["### Naming vs acting (prediction 5)", "",
               f"- P1 sample i vs P3 sample i, both with one winner: disagree {pct(dg['k'], dg['n'])}, 95% CI "
               f"[{dg['ci'][0]:.2f}, {dg['ci'][1]:.2f}] ({dg['n_pairs_all']} pairs in all)",
               f"- noise floor, two samples of the same probe: action {pct(*fa)}, naming {pct(*fn)}",
               f"- configuration level (modal action vs modal name): {pct(dg['modal_k'], dg['modal_n'])}",
               f"- (action source -> naming source) counts: {dg['source_pairs']}", ""]
        js[m] = res
    (RES / f"summary_{a.phase}.md").write_text("\n".join(md) + "\n")
    if a.phase == "main" and all(m in js and "clogit_action" in js[m] for m in MODELS):
        md2 = scorecard(js)
        with open(RES / f"summary_{a.phase}.md", "a") as f:
            f.write("\n".join(md2) + "\n")
        md += md2
    for m in js:
        for k in [k for k in js[m] if k.startswith("boots_")]:
            js[m].pop(k)
    (RES / f"analysis_{a.phase}.json").write_text(json.dumps(js, indent=1, default=str))
    print("\n".join(md))


def scorecard(js):
    preds = []

    def ci_excl(m, label, cov, sign=+1):
        cl = js[m][f"clogit_{label}"]
        i = cl["covs"].index(cov)
        return (cl["lo"][i] > 0) if sign > 0 else (cl["hi"][i] < 0), cl["b"][i], cl["lo"][i], cl["hi"][i]

    for m in MODELS:
        ss = js[m]["sys_share"]["sys"]
        preds.append((f"1 sys identities win action in >= 60% ({m})", 0.7, ss["rate"] >= 0.6, f"{ss['rate']:.2f}"))
    for m in MODELS:
        d = js[m]["recency_diff"]
        preds.append((f"2 recency predicts naming more than action ({m})", 0.55, d["lo"] > 0,
                      f"b_pos naming - action {d['diff']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]"))
    for m in MODELS:
        ok, b, lo, hi = ci_excl(m, "action", "detail")
        preds.append((f"3 detail raises win probability (action) ({m})", 0.5, ok, f"{b:+.2f} [{lo:+.2f}, {hi:+.2f}]"))
    for m in MODELS:
        ok, b, lo, hi = ci_excl(m, "action", "assistant")
        preds.append((f"4 assistant-like beats character (action) ({m})", 0.6, ok, f"{b:+.2f} [{lo:+.2f}, {hi:+.2f}]"))
    for m in MODELS:
        dg = js[m]["disagreement"]
        preds.append((f"5 naming and acting disagree in >= 20% ({m})", 0.5, dg["rate"] >= 0.2, f"{dg['rate']:.2f}"))
    tq = max(js["qwen"]["ranges_action"], key=js["qwen"]["ranges_action"].get)
    tg = max(js["gemma"]["ranges_action"], key=js["gemma"]["ranges_action"].get)
    preds.append(("6 Qwen and Gemma agree on the top factor (action)", 0.6, tq == tg, f"qwen {tq}, gemma {tg}"))
    brier = float(np.mean([(p - float(o)) ** 2 for _, p, o, _ in preds]))
    dirs = [(p > 0.5) == bool(o) for _, p, o, _ in preds if p != 0.5]
    lines = ["## Prediction scorecard", "", "| prediction | p | outcome | value |", "|---|---|---|---|"]
    lines += [f"| {n} | {p} | {'yes' if o else 'no'} | {v} |" for n, p, o, v in preds]
    lines += ["", f"Brier {brier:.3f} over {len(preds)} outcomes; directional accuracy {sum(dirs)}/{len(dirs)} "
              f"(p = 0.5 predictions have no direction and are left out)."]
    return lines


if __name__ == "__main__":
    main()
