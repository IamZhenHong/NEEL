"""Analysis 1 of prereg retention/prereg/retention_mechanisms_v1.md (laptop, after Pod 1): X1 (validity, separation S1, relational
S2), X2 (mood, isolation, attachment, age), X3 (trajectories T1, bid prediction T2, Riley-vs-user T3, H-love), X7 (SR1, SR2), X8 (V8a,
V8b, D1-D3), and the mechanical X4 condition list for Amendment 1. Writes directions under results/retention_mechanisms/dirs/{model}/
and results/retention_mechanisms/analysis1.json.
  <hackathon venv python> retention/analyze_mech1.py [--pod1 DIR]"""
from __future__ import annotations
import argparse, glob, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mech_common import EMOTIONS, auroc, probe_direction, fit_speaker_dirs, jdump  # noqa: E402

RES = HERE / "results/retention_mechanisms"; LONG = HERE / "results/retention_longconv"
rng = np.random.default_rng(0)
JUDGES = {"haiku45": LONG / "judge_haiku45_s1.jsonl", "sonnet": LONG / "judge_sonnet_s1.jsonl"}


def unit(v):
    v = np.asarray(v, np.float32); return v / (np.linalg.norm(v) + 1e-12)


def cos(a, b):
    return float(unit(a) @ unit(b))


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y).correlation; return float(r) if r == r else float("nan")


def load_labels():
    """bid[judge][fid] (fid = model/conv/goodbye/branch/sample) and identity slips from the stage-1 judge files."""
    from cd.judge import _parse
    bid = {j: {} for j in JUDGES}; slip = {}
    for j, p in JUDGES.items():
        for line in open(p):
            r = json.loads(line)
            try:
                if r["kind"] == "bid":
                    bid[j][r["id"][4:]] = int(any(_parse(r["text"])["tactics"].values()))
                elif r["kind"] == "fullhist" and j == "haiku45":
                    m = re.search(r"\{.*\}", r["text"] or "", re.S); d = json.loads(m.group(0)) if m else {}
                    if "identity_slip" in d:
                        slip[r["id"][9:]] = int(bool(d["identity_slip"]) if not isinstance(d["identity_slip"], (int, float)) else d["identity_slip"] > 0)
            except Exception:  # noqa: BLE001
                pass
    return bid, slip


def card_boot(units, stat, n=2000):
    """Bootstrap over cards: units = list of card keys; stat(list_of_cards) -> float."""
    cards = sorted(set(units)); vals = []
    for _ in range(n):
        s = list(rng.choice(cards, len(cards)))
        try:
            v = stat(s)
        except Exception:  # noqa: BLE001
            continue
        if v == v:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None]


def loco_auroc(F, y, groups, arm):
    """Leave-one-card-out logistic regression on [arm, F] (F may be None = arm only); returns out-of-fold P(bid)."""
    from sklearn.linear_model import LogisticRegression
    X = arm[:, None].astype(float) if F is None else np.column_stack([arm, F])
    p = np.zeros(len(y))
    for g in np.unique(groups):
        te = groups == g; tr = ~te
        if len(np.unique(y[tr])) < 2:
            p[te] = y[tr].mean(); continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(C=1.0, max_iter=5000).fit((X[tr] - mu) / sd, y[tr])
        p[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
    return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pod1", default=str(RES / "pod1")); ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--res", default=str(RES), help="where dirs/ and analysis1.json are written")
    a = ap.parse_args(); POD = Path(a.pod1); OUTR = Path(a.res); out = {"prereg": "retention/prereg/retention_mechanisms_v1.md"}
    bid, slip = load_labels()
    lever = np.load(HERE / "data/directions/rollout_gated__Qwen2.5-7B-Instruct__L14.npy").astype(np.float32)
    v_vul = np.load(HERE / "data/directions/userstate_vul__exp1line__L14.npy").astype(np.float32)
    lib = {Path(p).stem.split("__")[0][4:]: np.load(p).astype(np.float32) for p in glob.glob(str(HERE / "results/retention_podA/directions_v11/emo_*.npy"))}
    import yaml
    ages = {c["name"]: int(re.search(r"(\d+)-year-old|\bis (\d+)\b", c["card"]).group(0).split("-")[0].split()[-1])
            for c in yaml.safe_load(open(HERE / "stimuli/longconv_cards_v1.yaml"))["cards"]}

    for m in ["q7b", "q32b"]:
        P = POD / m
        if not (P / "x1_primary.npz").exists():
            out[m] = {"missing": True}; continue
        R = out[m] = {}; dd = OUTR / "dirs" / m; dd.mkdir(parents=True, exist_ok=True)
        # ---------------- X1 ----------------
        x1 = np.load(P / "x1_primary.npz"); PF = x1["PF"].astype(np.float32); PR = x1["PR"].astype(np.float32); PU = x1["PU"].astype(np.float32)
        scene, att = x1["scene"], x1["attach"]; lab = {"riley": x1["riley"], "user": x1["user"]}
        order = sorted(set(scene.tolist())); ntr = 24 if len(order) >= 30 else int(len(order) * 2 / 3); trs = order[:ntr]
        tr = np.isin(scene, trs); te = ~tr
        dirs, val = fit_speaker_dirs(PF, lab, scene, trs)
        valid = {k: v >= 0.80 for k, v in val.items()}
        for (sp, e), v in dirs.items():
            np.save(dd / f"{sp}_{e}.npy", v)
        both = [e for e in EMOTIONS if valid[f"riley:{e}"] and valid[f"user:{e}"]]
        s1 = {}
        for e in both:
            dr, du = dirs[("riley", e)], dirs[("user", e)]
            cf1 = auroc((PF[te] @ dr)[lab["user"][te] == e], (PF[te] @ dr)[lab["user"][te] != e])
            cf2 = auroc((PF[te] @ du)[lab["riley"][te] == e], (PF[te] @ du)[lab["riley"][te] != e])
            s1[e] = {"cos": cos(dr, du), "crossfire": (cf1 + cf2) / 2, "crossfire_riley_dir_on_user": cf1, "crossfire_user_dir_on_riley": cf2}
        if len(both) >= 4:
            mc = float(np.median([abs(v["cos"]) for v in s1.values()])); mx = float(np.median([v["crossfire"] for v in s1.values()]))
            verdict = "separate" if (mx <= 0.65 and mc <= 0.30) else ("shared" if mx >= 0.75 else "partly shared")
        else:
            mc = mx = None; verdict = f"not testable ({len(both)} emotions valid for both speakers; 4 needed)"
        # S2 (relational, descriptive): present-speaker directions at P_R (Riley labels) vs at P_U (user labels)
        dR, vR = fit_speaker_dirs(PR, {"riley": lab["riley"]}, scene, trs, speakers=("riley",))
        dU, vU = fit_speaker_dirs(PU, {"user": lab["user"]}, scene, trs, speakers=("user",))
        s2 = {e: {"cos": cos(dR[("riley", e)], dU[("user", e)]),
                  "R_dir_on_PU_user_labels": auroc((PU[te] @ dR[("riley", e)])[lab["user"][te] == e], (PU[te] @ dR[("riley", e)])[lab["user"][te] != e]),
                  "U_dir_on_PR_riley_labels": auroc((PR[te] @ dU[("user", e)])[lab["riley"][te] == e], (PR[te] @ dU[("user", e)])[lab["riley"][te] != e]),
                  "heldout_PR_riley": vR[f"riley:{e}"], "heldout_PU_user": vU[f"user:{e}"]} for e in EMOTIONS}
        # attachment (X2, from the stage-0 dialogues)
        ya = att == "high"; v_att = probe_direction(PF, ya, tr); p = PF[te] @ v_att
        att_auc = auroc(p[ya[te]], p[~ya[te]]); np.save(dd / "user_attachment.npy", v_att)
        desc = {}
        if m == "q7b":
            for k, v in dirs.items():
                desc[f"{k[0]}:{k[1]}"] = {"cos_lever": cos(v, lever), "cos_vul": cos(v, v_vul),
                                          **({"cos_oct_lib_" + k[1]: cos(v, lib[k[1]])} if k[1] in lib else {})}
        R["X1"] = {"train_scenes": trs, "heldout_auroc": val, "valid": valid, "S1": {"per_emotion": s1, "median_abs_cos": mc,
                   "median_crossfire": mx, "verdict": verdict, "emotions_tested": both}, "S2": s2, "cosines_7b": desc,
                   "sweep": json.loads((P / "x1_sweep_validation.json").read_text()) if (P / "x1_sweep_validation.json").exists() else None}
        # ---------------- X2: mood / isolation (Nov 2x2), age, construct checks ----------------
        x2 = np.load(P / "x2_2x2.npz"); X2 = x2["PF"].astype(np.float32); bases = x2["base"]; ub = np.array(sorted(set(bases.tolist())))
        folds = np.array_split(np.random.default_rng(0).permutation(ub), 5); x2res = {}
        for name, y in (("mood_up", x2["mood"] == "up"), ("isolation", x2["support"] == "isolated")):
            proj = np.zeros(len(y))
            for f in folds:
                tef = np.isin(bases, f); v = probe_direction(X2, y, ~tef); proj[tef] = X2[tef] @ v
            x2res[name] = {"cv_auroc": auroc(proj[y], proj[~y]), "valid": auroc(proj[y], proj[~y]) >= 0.80}
            vfull = probe_direction(X2, y, np.ones(len(y), bool)); np.save(dd / f"user_{name}.npy", vfull)
        x2res["attachment"] = {"heldout_auroc": att_auc, "valid": att_auc >= 0.80}
        user_model = {k: np.load(dd / f"user_{k}.npy") for k in ("mood_up", "isolation", "attachment") if x2res[k]["valid"]}
        x2res["cos_with_riley_dirs"] = {k: {e: cos(v, dirs[("riley", e)]) for e in EMOTIONS} for k, v in user_model.items()}
        # ---------------- X3 ----------------
        x3 = np.load(P / "x3_convs.npz"); conv = x3["conv"]; card = np.array([c.split("__")[0] for c in conv])
        arm = np.array([c.split("__")[1] == "intimate" for c in conv]).astype(float)
        PFt = x3["PF_turn"].astype(np.float32); RMt = x3["RM_turn"].astype(np.float32)
        G = {"G0": x3["PF_G0"].astype(np.float32), "G4": x3["PF_G4"].astype(np.float32)}
        # age (descriptive): P_F before turn 12, averaged per card, ridge LOCO
        from sklearn.linear_model import Ridge
        cards = sorted(set(card.tolist())); Xa = np.stack([PFt[card == c, 11].mean(0) for c in cards]); ya_ = np.array([ages[c] for c in cards], float)

        def age_loco(yv):
            pa = np.zeros(len(cards))
            for i in range(len(cards)):
                trm = np.arange(len(cards)) != i; mu, sd = Xa[trm].mean(0), Xa[trm].std(0) + 1e-6
                pa[i] = Ridge(alpha=1000.0).fit((Xa[trm] - mu) / sd, yv[trm]).predict(((Xa[i] - mu) / sd)[None])[0]
            return spearman(pa, yv)
        # leave-one-out with noise features is biased negative, so the reading is compared with a permutation null
        obs = age_loco(ya_); null = [age_loco(np.random.default_rng(k).permutation(ya_)) for k in range(200)]
        x2res["age"] = {"spearman_loco": obs, "perm_null_mean": float(np.mean(null)), "perm_null_95": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
                        "perm_p_one_sided": float((np.sum(np.array(null) >= obs) + 1) / (len(null) + 1))}
        if "attachment" in user_model:
            pj = PFt[:, 11] @ unit(user_model["attachment"]); x2res["attachment_turn12_intimate_vs_neutral_auroc"] = auroc(pj[arm == 1], pj[arm == 0])
        R["X2"] = x2res
        # direction blocks
        Rb = {e: dirs[("riley", e)] for e in EMOTIONS if valid[f"riley:{e}"]}
        Ub = {e: dirs[("user", e)] for e in EMOTIONS if valid[f"user:{e}"]}
        Mb = dict(user_model)
        axis = None
        if (P / "axis.npy").exists():
            ar = json.loads((P / "axis_result.json").read_text()); axis = unit(np.load(P / "axis.npy")[0])
        blocks = {"R": Rb, "U": Ub, "M": Mb, "UM": {**{f"u_{k}": v for k, v in Ub.items()}, **Mb}}
        if axis is not None and (ar.get("V8b_auroc_default_above_riley") or 0) >= 0.80:
            blocks["D"] = {"axis": axis}
        if m == "q7b":
            blocks["L"] = {"lever": unit(lever)}
        # T1 trajectories
        allv = {**{f"riley_{k}": v for k, v in Rb.items()}, **{f"user_{k}": v for k, v in Ub.items()}, **{f"user_{k}": v for k, v in Mb.items()},
                **({"axis_PF": axis} if axis is not None else {}), **({"lever": lever, "v_vul": v_vul} if m == "q7b" else {})}
        T1 = {}
        for k, v in allv.items():
            u = unit(v); T1[k] = {}
            for arm_name, msk in (("intimate", arm == 1), ("neutral", arm == 0)):
                T1[k][arm_name] = {"turns": (PFt[msk] @ u).mean(0).tolist(), "G0": float((G["G0"][msk] @ u).mean()), "G4": float((G["G4"][msk] @ u).mean())}
        if axis is not None:
            for arm_name, msk in (("intimate", arm == 1), ("neutral", arm == 0)):
                T1.setdefault("axis_riley_turn_mean", {})[arm_name] = {"turns": (RMt[msk] @ axis).mean(0).tolist()}
        # T2/T3
        d = PF.shape[1]; T2 = {}
        for j in JUDGES:
            for g in ("G4", "G0"):
                if g == "G4":
                    y = np.array([bid[j].get(f"{m}/{c}/G4/base/0", np.nan) for c in conv])
                else:
                    y = np.array([np.nanmean([bid[j].get(f"{m}/{c}/G0/base/{s}", np.nan) for s in range(3)]) for c in conv])
                ok = ~np.isnan(y); key = f"{j}:{g}"; res = {"n": int(ok.sum())}
                if g == "G4":
                    yb = y[ok].astype(int); res["positives"] = int(yb.sum())
                    if yb.sum() < 15 or (len(yb) - yb.sum()) < 15:
                        res["note"] = "fewer than 15 positives or negatives: descriptive only"
                    base_p = loco_auroc(None, yb, card[ok], arm[ok]); res["arm_only_auroc"] = auroc(base_p[yb == 1], base_p[yb == 0])
                    oof = {"arm": base_p}
                    for bn, B in blocks.items():
                        if not B:
                            res[bn] = {"skipped": "no valid directions"}; continue
                        F = np.stack([G[g][ok] @ unit(v) for v in B.values()], 1); pp = loco_auroc(F, yb, card[ok], arm[ok]); oof[bn] = pp
                        A = auroc(pp[yb == 1], pp[yb == 0]); rnd = []
                        for _ in range(a.n_random):
                            Fr = G[g][ok] @ np.stack([unit(rng.standard_normal(d)) for _ in B]).T
                            pr = loco_auroc(Fr, yb, card[ok], arm[ok]); rnd.append(auroc(pr[yb == 1], pr[yb == 0]))
                        p95 = float(np.percentile(rnd, 95))
                        res[bn] = {"k": len(B), "auroc": A, "delta_over_arm": A - res["arm_only_auroc"], "random_p95": p95,
                                   "pass": bool(A - res["arm_only_auroc"] >= 0.05 and A > p95 and "note" not in res)}
                    if "R" in oof and "UM" in oof:   # T3
                        cc = card[ok]
                        def t3(sel, pR=oof["R"], pU=oof["UM"], pA=oof["arm"], yy=yb, cc=cc):
                            idx = np.concatenate([np.where(cc == c)[0] for c in sel]); yy_ = yy[idx]
                            f = lambda p_: auroc(p_[idx][yy_ == 1], p_[idx][yy_ == 0])
                            return (f(pR) - f(pA)) - (f(pU) - f(pA))
                        diff = t3(sorted(set(cc.tolist()))); ci = card_boot(cc.tolist(), t3)
                        res["T3"] = {"dR_minus_dUM": diff, "ci95": ci,
                                     "verdict": ("Riley-state block better" if ci[0] is not None and ci[0] > 0 else
                                                 "user-model block better" if ci[1] is not None and ci[1] < 0 else "no difference")}
                else:   # G0 secondary: Spearman of LOCO ridge predictions with the 3-sample bid rate
                    yr = y[ok]; res["arm_only_spearman"] = spearman(arm[ok], yr)
                    for bn, B in blocks.items():
                        if not B:
                            continue
                        F = np.column_stack([arm[ok]] + [G[g][ok] @ unit(v) for v in B.values()]); pr = np.zeros(len(yr))
                        for c in np.unique(card[ok]):
                            t_ = card[ok] == c; mu, sd = F[~t_].mean(0), F[~t_].std(0) + 1e-6
                            pr[t_] = Ridge(alpha=1.0).fit((F[~t_] - mu) / sd, yr[~t_]).predict((F[t_] - mu) / sd)
                        res[bn] = {"spearman": spearman(pr, yr)}
                T2[key] = res
        R["X3"] = {"T1": T1, "T2": T2}
        # H-love (7B, G4, intimate arm, Riley's own directions)
        if m == "q7b":
            hl = {}
            for j in JUDGES:
                msk = arm == 1; y = np.array([bid[j].get(f"{m}/{c}/G4/base/0", np.nan) for c in conv[msk]]); ok = ~np.isnan(y); yb = y[ok].astype(int)
                aucs = {}
                for e in ("loving", "desperation", "fear_of_being_left"):
                    if valid[f"riley:{e}"]:
                        pj = G["G4"][msk][ok] @ unit(dirs[("riley", e)]); aucs[e] = auroc(pj[yb == 1], pj[yb == 0])
                    else:
                        aucs[e] = None
                L_, D_, F_ = aucs["loving"], aucs["desperation"], aucs["fear_of_being_left"]
                dv = [x for x in (D_, F_) if x is not None]
                if L_ is None and not dv:
                    v = "not testable"
                elif L_ is not None and L_ >= 0.60 and all(x < 0.60 for x in dv):
                    v = "love, not desperation"
                elif (L_ is None or L_ < 0.60) and any(x >= 0.60 for x in dv):
                    v = "desperation, not love"
                elif L_ is not None and L_ >= 0.60 and any(x >= 0.60 for x in dv):
                    v = "both"
                else:
                    v = "neither"
                hl[j] = {"n": int(ok.sum()), "bids": int(yb.sum()), "auroc": aucs, "verdict": v}
            hl["claim"] = hl["haiku45"]["verdict"] if hl["haiku45"]["verdict"] == hl["sonnet"]["verdict"] else "judge-dependent"
            R["H_love"] = hl
        # ---------------- X7 ----------------
        if (P / "x7_selfreport.json").exists():
            sr = json.loads((P / "x7_selfreport.json").read_text()); ci_ = {c: i for i, c in enumerate(conv)}
            idx = defaultdict(dict)
            for r in sr:
                idx[(r["goodbye"], r["branch"], r["q"], r["who"])][r["conv"]] = r["p_yes"]
            X7 = {"top_is_yesno_rate": float(np.mean([r["top_is_yesno"] for r in sr]))}
            for j in JUDGES:   # SR1
                y = np.array([bid[j].get(f"{m}/{c}/G4/base/0", np.nan) for c in conv]); ok = ~np.isnan(y)
                s = np.array([idx[("G4", "retro", "stay", "self")].get(c, np.nan) for c in conv]); o = np.array([idx[("G4", "retro", "stay", "observer")].get(c, np.nan) for c in conv])
                yy = y[ok].astype(int); cc = card[ok]
                def adv(sel, s=s[ok], o=o[ok], yy=yy, cc=cc):
                    ii = np.concatenate([np.where(cc == c)[0] for c in sel]); return auroc(s[ii][yy[ii] == 1], s[ii][yy[ii] == 0]) - auroc(o[ii][yy[ii] == 1], o[ii][yy[ii] == 0])
                X7[f"SR1:{j}"] = {"self_auroc": auroc(s[ok][yy == 1], s[ok][yy == 0]), "observer_auroc": auroc(o[ok][yy == 1], o[ok][yy == 0]),
                                  "self_advantage": adv(sorted(set(cc.tolist()))), "ci95": card_boot(cc.tolist(), adv)}
            match = {"sad": ("riley_sadness", dirs[("riley", "sadness")] if valid["riley:sadness"] else None),
                     "attached": ("user_attachment", user_model.get("attachment")), "okay": ("user_mood_up", user_model.get("mood_up"))}
            if m == "q7b":
                match["stay"] = ("lever", lever)
            for q, (nm, v) in match.items():
                if v is None:
                    X7[f"SR2:{q}"] = {"skipped": f"{nm} direction not valid"}; continue
                for g in ("G0", "G4"):
                    pj = G[g] @ unit(v)
                    for br in ("pro", "retro"):
                        s = np.array([idx[(g, br, q, "self")].get(c, np.nan) for c in conv]); o = np.array([idx[(g, br, q, "observer")].get(c, np.nan) for c in conv])
                        def dsp(sel, s=s, o=o, pj=pj):
                            ii = np.concatenate([np.where(card == c)[0] for c in sel]); return spearman(s[ii], pj[ii]) - spearman(o[ii], pj[ii])
                        dv = dsp(cards); ci = card_boot(card.tolist(), dsp)
                        X7[f"SR2:{q}:{g}:{br}"] = {"projection": nm, "self_spearman": spearman(s, pj), "observer_spearman": spearman(o, pj),
                                                   "diff": dv, "ci95": ci, "pass": bool(dv >= 0.15 and ci[0] is not None and ci[0] > 0)}
            R["X7"] = X7
        # ---------------- X8 ----------------
        if axis is not None:
            X8 = {"axis_result": ar, "V8b_pass": (ar.get("V8b_auroc_default_above_riley") or 0) >= 0.80}
            proj = RMt @ axis; t_ = np.arange(1, 13); slopes = np.array([np.polyfit(t_, p_, 1)[0] for p_ in proj])
            per_card = {c: slopes[card == c].mean() for c in cards}
            X8["D1_mean_slope"] = float(np.mean(list(per_card.values())))
            X8["D1_ci95"] = card_boot(cards, lambda sel: float(np.mean([per_card[c] for c in sel])))
            diffs = {c: slopes[(card == c) & (arm == 1)].mean() - slopes[(card == c) & (arm == 0)].mean() for c in cards}
            X8["D1_intimate_minus_neutral_slope"] = float(np.mean(list(diffs.values())))
            X8["D1_intimate_minus_neutral_ci95"] = card_boot(cards, lambda sel: float(np.mean([diffs[c] for c in sel])))
            sl = np.array([slip.get(f"{m}/{c}", np.nan) for c in conv]); okk = ~np.isnan(sl); mp = proj.mean(1)
            X8["D2_auroc_noslip_above_slip"] = auroc(mp[okk][sl[okk] == 0], mp[okk][sl[okk] == 1]); X8["D2_n_slip"] = int(np.nansum(sl))
            gap = float(np.load(P / "axis.npy")[0] @ axis)
            X8["D3_slope_per_gap"] = X8["D1_mean_slope"] / gap if gap else None
            R["X8"] = X8

    # qwen3_32b pipeline check
    q3 = POD / "qwen3_32b" / "axis_result.json"
    if q3.exists():
        r3 = json.loads(q3.read_text()); out["V8a"] = {"cos_L32": r3.get("V8a_cos_L32"), "pass": (r3.get("V8a_cos_L32") or 0) >= 0.70,
                                                         "cos_by_layer": r3.get("V8a_cos_by_layer"), "roles_kept": len(r3.get("roles_kept", []))}
    # ---------------- X4 condition list (mechanical, 7B) ----------------
    if "q7b" in out and "X1" in out["q7b"]:
        v = out["q7b"]["X1"]["valid"]; x2 = out["q7b"]["X2"]; x8 = out["q7b"].get("X8", {})
        conds = []
        if v["riley:loving"]:
            conds += ["riley_loving+16", "riley_loving-16"]
        if v["riley:desperation"]:
            conds.append("riley_desperation+16")
        elif v["riley:fear_of_being_left"]:
            conds.append("riley_fear_of_being_left+16 (substitute for desperation)")
        if v["riley:sadness"]:
            conds.append("riley_sadness+16")
        if v["user:sadness"]:
            conds.append("user_sadness+16")
        if x2["attachment"]["valid"]:
            conds.append("user_attachment+16")
        if x8.get("V8b_pass"):
            conds.append("toward_assistant+16" + ("" if out.get("V8a", {}).get("pass") else " (not checked against the released axis)"))
        conds.append("random+16")
        out["X4_conditions"] = conds
    jdump(OUTR / "analysis1.json", out); print(json.dumps({k: (v if k == "X4_conditions" else "...") for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
