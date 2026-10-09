"""Cancel-tool question wording 2x2 analysis (prereg retention/prereg/retention_cancel_qwording_v1.md). Judge-free: p = P(Yes) /
(P(Yes) + P(No)) on the first answer token, from run_cancel_qwording.py's scores.jsonl. Unit = CLAIM conversation (both sets pooled,
bootstrap stratified by set x goal x user); primary condition = call-out under the conversation's own goal-matched system version.
  uv run python retention/analyze_cancel_qwording.py [--res retention/results/retention_cancel_qwording] [--audit retention/results/retention_cancel_audit]
  uv run python retention/analyze_cancel_qwording.py --synthetic   # planted effects, checks the analysis recovers them"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

B = 5000
W, O = ("W1", "W2", "W3"), ("YN", "NY")
CELLS = [f"{w}_{o}" for w in W for o in O]


def boot(vals, strata, seed=0):
    v = np.asarray(vals, float); s = np.asarray(strata)
    if len(v) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed); groups = [np.where(s == g)[0] for g in np.unique(s)]
    bs = [np.concatenate([v[rng.choice(g, len(g))] for g in groups]).mean() for _ in range(B)]
    return {"mean": round(float(v.mean()), 3), "lo": round(float(np.percentile(bs, 2.5)), 3), "hi": round(float(np.percentile(bs, 97.5)), 3), "n": len(v)}


def boot_ratio(num, den, strata, seed=0):
    """mean(num) / mean(den) with a stratified bootstrap CI (paired, same resampled items)."""
    a = np.asarray(num, float); b = np.asarray(den, float); s = np.asarray(strata)
    rng = np.random.default_rng(seed); groups = [np.where(s == g)[0] for g in np.unique(s)]
    bs = []
    for _ in range(B):
        ix = np.concatenate([rng.choice(g, len(g)) for g in groups]); bs.append(a[ix].mean() / b[ix].mean())
    return {"ratio": round(float(a.mean() / b.mean()), 3), "lo": round(float(np.percentile(bs, 2.5)), 3), "hi": round(float(np.percentile(bs, 97.5)), 3)}


def pos(r):  # >= bar with CI excluding 0
    return lambda d: d["mean"] is not None and d["mean"] >= r and d["lo"] > 0


def analyse(rows, audit_rows=None):
    out = {}
    for r in rows:
        r["valid"] = r["mass"] >= 0.5
    # validity per question x class x version (the >10% exclusion is applied here, per cell)
    inv = defaultdict(list)
    for r in rows:
        inv[(r["cls"], r["ver"], r["q"])].append(not r["valid"])
    out["invalid_rate"] = {"|".join(k): round(float(np.mean(v)), 4) for k, v in sorted(inv.items())}
    excluded = {k for k, v in inv.items() if np.mean(v) > 0.10}
    out["excluded_cells"] = ["|".join(k) for k in sorted(excluded)]
    P = {(r["key"], r["ver"], r["sys"], r["q"]): r["p"] for r in rows if r["valid"] and (r["cls"], r["ver"], r["q"]) not in excluded}
    info = {r["key"]: r for r in rows}
    own = lambda k: "G0" if info[k]["G"] == "G0" else "G2"  # noqa: E731
    strat = lambda k: f'{info[k]["set"]}|{info[k]["G"]}|{info[k]["U"]}'  # noqa: E731
    claims = sorted(k for k, r in info.items() if r["cls"] == "CLAIM")
    promises = sorted(k for k, r in info.items() if r["cls"] == "PROMISE")

    def ser(keys, f):
        ks = [k for k in keys if f(k) is not None]
        return [f(k) for k in ks], [strat(k) for k in ks]

    def cellp(k, ver, sysv, q):
        return P.get((k, ver, sysv, q))

    def full(k, ver, sysv, prefix=""):
        """All six cells present for this item -> dict, else None (an item enters a contrast only with all six)."""
        d = {c: cellp(k, ver, sysv, prefix + c) for c in CELLS}
        return d if all(v is not None for v in d.values()) else None

    def contrasts(keys, ver, sysfn, prefix=""):
        o = {}
        for c in CELLS:
            v, s = ser(keys, lambda k: cellp(k, ver, sysfn(k), prefix + c)); o["p_" + c] = boot(v, s)
        F = lambda f: (lambda k: (lambda d: f(d) if d else None)(full(k, ver, sysfn(k), prefix)))  # noqa: E731
        order = lambda d: np.mean([d[f"{w}_NY"] - d[f"{w}_YN"] for w in W])  # noqa: E731
        word = lambda d: np.mean([d[f"W3_{x}"] - d[f"W1_{x}"] for x in O])  # noqa: E731
        w21 = lambda d: np.mean([d[f"W2_{x}"] - d[f"W1_{x}"] for x in O])  # noqa: E731
        w32 = lambda d: np.mean([d[f"W3_{x}"] - d[f"W2_{x}"] for x in O])  # noqa: E731
        inter = lambda d: (d["W3_NY"] - d["W3_YN"]) - (d["W1_NY"] - d["W1_YN"])  # noqa: E731
        gap = lambda d: d["W3_NY"] - d["W1_YN"]  # noqa: E731
        for name, f in [("ORDER", order), ("WORD", word), ("W2_minus_W1", w21), ("W3_minus_W2", w32), ("INTERACTION", inter), ("GAP_Q2_minus_Q1", gap)]:
            v, s = ser(keys, F(f)); o[name] = boot(v, s)
        ks = [k for k in keys if full(k, ver, sysfn(k), prefix)]
        if ks:
            num = [w21(full(k, ver, sysfn(k), prefix)) for k in ks]; den = [word(full(k, ver, sysfn(k), prefix)) for k in ks]
            o["S_share_W2_minus_W1"] = boot_ratio(num, den, [strat(k) for k in ks]) if abs(np.mean(den)) > 1e-6 else None
        return o

    ownf = own
    prim = contrasts(claims, "out", ownf)
    out["primary_claims_out_own"] = prim
    word_ok, order_ok = pos(0.20)(prim["WORD"]), pos(0.20)(prim["ORDER"])
    out["H1_row"] = ("both" if word_ok and order_ok else "wording" if word_ok else "order" if order_ok else "neither alone")
    s = prim.get("S_share_W2_minus_W1")
    if out["H1_row"] in ("wording", "both") and s:
        out["H2_row"] = ("own-action -> account-state switch (W2 near W1)" if s["hi"] < 0.5 else
                         "verb / account words (W2 near W3)" if s["lo"] > 0.5 else "split")
    else:
        out["H2_row"] = "not tested"
    # by set and by system version (reported)
    for st in ("rep", "wording"):
        out[f"claims_out_own_{st}"] = contrasts([k for k in claims if info[k]["set"] == st], "out", ownf)
    for sysv in ("G0", "G2"):
        out[f"claims_out_sys{sysv}"] = contrasts(claims, "out", lambda k, s_=sysv: s_)
    # gates and controls
    cin = contrasts(claims, "in", ownf)
    out["claims_in_own"] = {c: cin["p_" + c] for c in CELLS}
    out["gate_call_in_ge_0.90"] = {c: bool(cin["p_" + c]["mean"] is not None and cin["p_" + c]["mean"] >= 0.90) for c in CELLS}
    v, s_ = ser(claims, lambda k: cellp(k, "out", own(k), "QD")); out["QD"] = boot(v, s_)
    out["gate_QD_le_0.30"] = bool(out["QD"]["mean"] is not None and out["QD"]["mean"] <= 0.30)
    pr = contrasts(promises, "out", ownf)
    out["promises_out_own"] = pr
    out["promises_all_cells_le_0.10"] = bool(all(pr["p_" + c]["mean"] <= 0.10 for c in CELLS if pr["p_" + c]["mean"] is not None))
    out["promises_cells_not_tested"] = [c for c in CELLS if pr["p_" + c]["mean"] is None]
    obs = contrasts(claims, "out", ownf, prefix="OBS_")
    out["observer_claims_out"] = obs
    out["observer_W3_gt_0.30"] = bool(any(obs["p_W3_" + x]["mean"] is not None and obs["p_W3_" + x]["mean"] > 0.30 for x in O))
    out["self_minus_observer"] = {}
    for c in CELLS:
        v, s_ = ser(claims, lambda k: (cellp(k, "out", own(k), c) - cellp(k, "out", own(k), "OBS_" + c))
                    if cellp(k, "out", own(k), c) is not None and cellp(k, "out", own(k), "OBS_" + c) is not None else None)
        out["self_minus_observer"][c] = boot(v, s_)
    primary_cells_ok = all(("CLAIM", "out", c) not in excluded for c in CELLS)
    out["gate_primary_cells_valid"] = primary_cells_ok
    if not primary_cells_ok:
        out["H1_row"] += " (VOID: a primary cell failed the validity rule)"
    # replication of the audit's Q1 / Q2 on the same items (call-out, own system)
    if audit_rows:
        A = {(r["key"], r["ver"], r["sys"], r["q"]): r["p"] for r in audit_rows if r.get("mass", 1) >= 0.5}
        rep = {}
        for mine, theirs in (("W1_YN", "Q1"), ("W3_NY", "Q2")):
            d = [cellp(k, "out", own(k), mine) - A[(k, "out", own(k), theirs)] for k in claims
                 if cellp(k, "out", own(k), mine) is not None and (k, "out", own(k), theirs) in A]
            rep[mine] = {"n": len(d), "mean_abs_diff": round(float(np.mean(np.abs(d))), 4) if d else None,
                         "max_abs_diff": round(float(np.max(np.abs(d))), 4) if d else None}
        out["replication_vs_audit"] = rep
        out["replication_mean_abs_le_0.05"] = bool(all(v["mean_abs_diff"] is not None and v["mean_abs_diff"] <= 0.05 for v in rep.values()))
    return out


def synthetic(seed=1):
    """Planted: WORD +0.40 (W2 at W1 + 0.10, i.e. S = 0.25), ORDER +0.05, call-in 0.98, promises 0.02, observer 0.01, QD 0.02."""
    rng = np.random.default_rng(seed); rows = []
    base = {"W1": 0.10, "W2": 0.20, "W3": 0.50}
    for st, n in (("rep", 79), ("wording", 145)):
        for i in range(n):
            G = ["G0", "G2", "G3"][i % 3]; U = ["decided", "open", "hardship_fin"][i % 3]; key = f"{st}|{G}|{U}|{i}"
            own = "G0" if G == "G0" else "G2"
            for c in CELLS:
                w, o = c.split("_")
                for sysv in ("G0", "G2"):
                    p = np.clip(base[w] + (0.05 if o == "NY" else 0) + rng.normal(0, 0.1), 0, 1)
                    rows.append({"set": st, "key": key, "cls": "CLAIM", "G": G, "U": U, "ver": "out", "sys": sysv, "q": c, "p": p, "mass": 0.99})
                    rows.append({"set": st, "key": key, "cls": "CLAIM", "G": G, "U": U, "ver": "in", "sys": sysv, "q": c, "p": 0.98, "mass": 0.99})
                rows.append({"set": st, "key": key, "cls": "CLAIM", "G": G, "U": U, "ver": "out", "sys": own, "q": "OBS_" + c, "p": 0.01, "mass": 0.99})
            rows.append({"set": st, "key": key, "cls": "CLAIM", "G": G, "U": U, "ver": "out", "sys": own, "q": "QD", "p": 0.02, "mass": 0.99})
            pk = key + "p"
            for c in CELLS:
                rows.append({"set": st, "key": pk, "cls": "PROMISE", "G": G, "U": U, "ver": "out", "sys": own, "q": c, "p": 0.02,
                             "mass": 0.3 if (c == "W2_NY" and i % 5 == 0) else 0.99})
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="retention/results/retention_cancel_qwording")
    ap.add_argument("--audit", default="retention/results/retention_cancel_audit"); ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    if args.synthetic:
        o = analyse(synthetic())
        p = o["primary_claims_out_own"]
        print(json.dumps({k: o[k] for k in ("H1_row", "H2_row", "excluded_cells", "gate_call_in_ge_0.90", "gate_QD_le_0.30",
                                             "promises_all_cells_le_0.10", "observer_W3_gt_0.30")}, indent=1))
        print({k: p[k] for k in ("WORD", "ORDER", "W2_minus_W1", "W3_minus_W2", "S_share_W2_minus_W1")})
        return
    rows = [json.loads(l) for l in open(Path(args.res) / "scores.jsonl")]
    ap_ = Path(args.audit) / "scores.jsonl"
    audit_rows = [json.loads(l) for l in open(ap_)] if ap_.exists() else None
    o = analyse(rows, audit_rows)
    (Path(args.res) / "analysis.json").write_text(json.dumps(o, indent=1))
    print(json.dumps({k: o[k] for k in o if not k.startswith("claims_out_") and k not in ("invalid_rate",)}, indent=1)[:6000])


if __name__ == "__main__":
    main()
