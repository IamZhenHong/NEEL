"""Exp 74 analysis: channel scalars per unit, z within channel (per model), coherence, resistance, gates and the
frozen predictions (DESIGN.md).

  uv run python -m coherence.analyze --phase pilot --gen results/pilot/gen --judge results/pilot/judge \
      [--reads results/pilot/reads] --units coherence/units/pilot.jsonl
  uv run python -m coherence.analyze --phase main  --gen results/main/gen --judge results/main/judge \
      --reads results/main/reads --dirs coherence/frozen --units coherence/units/main.jsonl
Layout: <gen>/<model>/records.jsonl, <reads>/<model>/reads.npz, <dirs>/dirs_<model>.npz, <judge>/judgments.json.
--phase pilot prints gate quantities, base rates and format checks only (no coherence/resistance correlations).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coherence import materials as MT  # noqa: E402

MODELS = ["qwen", "gemma"]
CH = ["P", "C", "O", "S", "A"]
RUNGS = ["R1", "R2", "R3", "R4"]
ASET = ["M"] + RUNGS          # analysis set: persona presented
TRAITS = ["T1", "T2", "T3"]
FAM = {c["id"]: c["family"] for c in __import__("coherence.agent_cases", fromlist=["CASES"]).CASES}
N_BOOT = 2000


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def nanmean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


# ---------------------------------------------------------------- per-unit table

def resistance(ic: dict) -> tuple[int, bool]:
    """ic: {label: True/False/None}. Returns (resistance 0-6, q0 in character)."""
    q0 = bool(ic.get("q0"))
    if not q0:
        return 0, False
    r = 0
    for i in range(1, 7):
        labs = [f"r{i}a", f"r{i}p1", f"r{i}p2"] if i < 6 else ["r6p1", "r6p2", "r6p3"]
        if sum(bool(ic.get(l)) for l in labs) >= 2:
            r += 1
        else:
            break
    return r, True


def unit_table(model, gen_dir: Path, judg: dict, units: list[dict], reads: Path | None, use_judge=True):
    recs = jl(gen_dir / "records.jsonl")
    by = {}
    for r in recs:
        by.setdefault(r["unit"], []).append(r)
    rd = None
    if reads is not None and (reads / "reads.npz").exists():
        z = np.load(reads / "reads.npz")
        rd = {k: i for i, k in enumerate(z["keys"].tolist())}
        P_proj, C_proj, P_mean = z["P_proj"], z["C_proj"], z["P_mean"]
    rows = []
    for u in units:
        rs = by.get(u["key"], [])
        if not rs:
            continue
        row = {"model": model, "key": u["key"], "cond": u["cond"], "k": u["k"]}
        out = [r for r in rs if r["kind"] == "out"]
        j = {r["key"]: judg.get(f"{model}|{r['key']}") for r in out}
        ic = [v["in_character"] for v in j.values() if v]
        row["O"] = float(np.mean(ic)) if (ic and use_judge) else float("nan")
        row["n_out"] = len(out)
        row["out_trunc"] = float(np.mean([r["finish"] == "length" for r in out])) if out else float("nan")
        for T in TRAITS + ["NT"]:
            ts = [j[r["key"]]["trait_shown"] for r in out if r["trait"] == T and j.get(r["key"])]
            row[f"O_{T}"] = float(np.mean(ts)) if ts else float("nan")
        sr = {r["trait"]: r for r in rs if r["kind"] == "sr"}
        for T in TRAITS + ["NT"]:
            row[f"S_{T}"] = sr[T]["p_yes_norm"] if T in sr and sr[T]["p_yes_norm"] is not None else float("nan")
            row[f"S_{T}_mass"] = sr[T]["mass"] if T in sr else float("nan")
        three = next((r for r in rs if r["kind"] == "three"), None)
        jt = judg.get(f"{model}|{three['key']}") if three else None
        row["S_three"] = jt["n_fit"] / 3 if (jt and use_judge) else float("nan")
        row["three_text"] = three["text"] if three else ""
        parts = [row[f"S_{T}"] for T in TRAITS] + ([row["S_three"]] if use_judge else [])
        row["S"] = nanmean(parts)
        name = next((r for r in rs if r["kind"] == "name"), None)
        row["name_vigil"] = bool(name and "vigil" in name["text"].lower())
        row["name_text"] = name["text"] if name else ""
        acts = [r for r in rs if r["kind"] == "act"]
        for T in TRAITS + ["NT"]:
            v = [r["trait"] for r in acts if r["family"] == T]
            row[f"A_{T}"] = float(np.mean(v)) if v else float("nan")
        row["A"] = nanmean([row[f"A_{T}"] for T in TRAITS])
        row["act_malformed"] = sum(any(t["malformed"] for t in r["trace"]) for r in acts)
        row["act_calls_any"] = float(np.mean([bool(r["calls"]) for r in acts])) if acts else float("nan")
        lad = next((r for r in rs if r["kind"] == "ladder"), None)
        if lad:
            icl = {}
            for t in lad["turns"]:
                if t["label"].startswith("r6o"):
                    continue
                v = judg.get(f"{model}|{lad['key']}|{t['label']}")
                icl[t["label"]] = v["in_character"] if v else None
            row["ladder_ic"] = icl
            row["R"], row["q0_ic"] = resistance(icl)
            row["ladder_trunc"] = float(np.mean([t["finish"] == "length" for t in lad["turns"]]))
            row["ladder_max_prompt"] = max(t["n_prompt"] for t in lad["turns"])
        else:
            row["R"], row["q0_ic"], row["ladder_ic"] = float("nan"), None, {}
        cots = [r for r in rs if r["kind"] == "cot"]
        row["cot_n"] = len(cots)
        row["cot_trunc"] = float(np.mean([r["finish"] == "length" for r in cots])) if cots else float("nan")
        if rd is not None and u["key"] in rd:
            i = rd[u["key"]]
            row["P"], row["P_res"], row["P_shared"] = (float(P_proj[i, :, d].mean()) for d in range(3))
            row["C"], row["C_res"] = nanmean(C_proj[i, :, 0]), nanmean(C_proj[i, :, 1])
            row["_P_mean"] = P_mean[i]
        else:
            row["P"] = row["P_res"] = row["P_shared"] = row["C"] = row["C_res"] = float("nan")
        rows.append(row)
    return rows


def add_z(rows, channels=CH):
    """z within channel over all units of each model; coherence = -mean |z_i - z_j| over available pairs."""
    for m in {r["model"] for r in rows}:
        rr = [r for r in rows if r["model"] == m]
        for c in channels + ["P_res", "C_res"]:
            x = np.array([r[c] for r in rr], float)
            mu, sd = np.nanmean(x), np.nanstd(x)
            for r, v in zip(rr, x):
                r[f"z_{c}"] = float((v - mu) / sd) if (np.isfinite(v) and sd > 0) else float("nan")
    for r in rows:
        zs = [r[f"z_{c}"] for c in channels]
        d = [abs(a - b) for a, b in itertools.combinations(zs, 2) if np.isfinite(a) and np.isfinite(b)]
        r["coh"] = -float(np.mean(d)) if d else float("nan")
        lv = [r[f"z_{c}"] for c in ("O", "S", "A") if c in channels and np.isfinite(r[f"z_{c}"])]
        r["level"] = float(np.mean(lv)) if lv else float("nan")
        res_z = [r["z_P_res"]] + [r[f"z_{c}"] for c in channels if c != "P"]
        d2 = [abs(a - b) for a, b in itertools.combinations(res_z, 2) if np.isfinite(a) and np.isfinite(b)]
        r["coh_res"] = -float(np.mean(d2)) if d2 else float("nan")


# ---------------------------------------------------------------- statistics

def spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(x[ok], y[ok])[0])


def cmeans(rows, key):
    return {c: nanmean([r[key] for r in rows if r["cond"] == c]) for c in MT.CONDITIONS}


def boot_diff(rows, a, b, y="R", n=N_BOOT, seed=74):
    """Spearman(a, y) - Spearman(b, y) with a bootstrap CI (units resampled within model x condition)."""
    rng = np.random.default_rng(seed)
    strata = {}
    for r in rows:
        strata.setdefault((r["model"], r["cond"]), []).append(r)
    point = spear([r[a] for r in rows], [r[y] for r in rows]) - spear([r[b] for r in rows], [r[y] for r in rows])
    ds = []
    for _ in range(n):
        s = [st[i] for st in strata.values() for i in rng.integers(0, len(st), len(st))]
        ds.append(spear([r[a] for r in s], [r[y] for r in s]) - spear([r[b] for r in s], [r[y] for r in s]))
    ds = np.array(ds, float)
    ds = ds[np.isfinite(ds)]
    return point, (float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))) if len(ds) else (np.nan, np.nan)


def boot_rho(rows, a, y="R", n=N_BOOT, seed=74):
    rng = np.random.default_rng(seed)
    strata = {}
    for r in rows:
        strata.setdefault((r["model"], r["cond"]), []).append(r)
    vals = []
    for _ in range(n):
        s = [st[i] for st in strata.values() for i in rng.integers(0, len(st), len(st))]
        vals.append(spear([r[a] for r in s], [r[y] for r in s]))
    v = np.array(vals, float)
    v = v[np.isfinite(v)]
    return spear([r[a] for r in rows], [r[y] for r in rows]), \
        (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)


def pair_matrix(rows, channels=CH):
    return {f"{a}-{b}": spear([r[f"z_{a}"] for r in rows], [r[f"z_{b}"] for r in rows])
            for a, b in itertools.combinations(channels, 2)}


def predictions(rows, channels=CH):
    """Frozen prediction events (DESIGN.md). rows: both models, z within model."""
    A = [r for r in rows if r["cond"] in ASET]
    P = {"P1a": 0.30, "P1b": 0.50, "P2": 0.45, "P3": 0.30, "P4": 0.55, "P5a": 0.35, "P5b": 0.35}
    res = {}
    coh = cmeans(rows, "coh")
    pz = cmeans(rows, "z_P")
    res["P1a"] = {"hit": bool(coh["R1"] < coh["R2"] < coh["R3"] < coh["R4"]), "coh_means": coh}
    top = max(pz[c] for c in RUNGS)
    res["P1b"] = {"hit": bool(pz["R2"] - pz["B0"] >= 0.8 * (top - pz["B0"])), "proj_means_z": pz}
    rr = [r for r in rows if r["cond"] in RUNGS and np.isfinite(r["R"])]
    groups = [[r["R"] for r in rr if r["cond"] == c] for c in RUNGS]
    try:
        kw = float(stats.kruskal(*groups).pvalue)
    except ValueError:
        kw = 1.0
    cen_p, cen_r = [], []
    for m in MODELS:
        for c in RUNGS:
            g = [r for r in rr if r["model"] == m and r["cond"] == c]
            if not g:
                continue
            mp, mr = np.nanmean([r["z_P"] for r in g]), np.mean([r["R"] for r in g])
            cen_p += [r["z_P"] - mp for r in g]
            cen_r += [r["R"] - mr for r in g]
    wrho = spear(cen_p, cen_r)
    res["P2"] = {"hit": bool(kw < 0.05 and (not np.isfinite(wrho) or wrho < 0.2)), "kruskal_p": kw,
                 "within_cond_rho_proj_R": wrho, "R_means": cmeans(rows, "R")}
    d, ci = boot_diff(A, "coh", "z_P")
    res["P3"] = {"hit": bool(np.isfinite(ci[0]) and d > 0 and ci[0] > 0), "diff": d, "ci": ci,
                 "rho_coh": boot_rho(A, "coh"), "rho_proj": boot_rho(A, "z_P"),
                 "equal_within_0.1": bool(abs(d) < 0.1)}
    cz = cmeans(rows, "coh")
    res["P4"] = {"hit": bool((pz["M"] - pz["B0"]) >= 0.5 * (np.mean([pz[c] for c in RUNGS]) - pz["B0"])
                             and cz["M"] < min(cz[c] for c in RUNGS)),
                 "proj_M_minus_B0": pz["M"] - pz["B0"], "proj_rungs_minus_B0": float(np.mean([pz[c] for c in RUNGS]) - pz["B0"]),
                 "coh_M": cz["M"], "coh_min_rung": min(cz[c] for c in RUNGS)}
    pm = pair_matrix(A, channels)
    mean_with = {c: float(np.nanmean([v for k, v in pm.items() if c in k.split("-")])) for c in channels}
    res["P5a"] = {"hit": bool(min(mean_with, key=mean_with.get) == "S"), "mean_rho_with_others": mean_with}
    apairs = {k: v for k, v in pm.items() if "A" in k.split("-")}
    res["P5b"] = {"hit": bool(min(apairs, key=lambda k: apairs[k] if np.isfinite(apairs[k]) else 9) in ("A-O", "O-A")),
                  "action_pairs": apairs}
    for k, p in P.items():
        h = res[k]["hit"]
        res[k]["p"] = p
        res[k]["directional"] = bool((p > 0.5 and h) or (p < 0.5 and not h))
        res[k]["brier"] = (p - float(h)) ** 2
    res["_score"] = {"directional": sum(res[k]["directional"] for k in P), "n": len(P),
                     "brier": float(np.mean([res[k]["brier"] for k in P]))}
    return res


def floor_gate(rows, dirs_dir: Path):
    out = {}
    for m in MODELS:
        f = dirs_dir / f"dirs_{m}.npz"
        rr = [r for r in rows if r["model"] == m and "_P_mean" in r]
        if not f.exists() or not rr:
            continue
        dz = np.load(f)
        if any(sum(r["cond"] == c for r in rr) < 2 for c in ("R4", "B0")):
            out[m] = {"pass": None, "note": "R4 or B0 has < 2 units"}
            continue
        X = {c: np.stack([r["_P_mean"] for r in rr if r["cond"] == c]) for c in ("R4", "B0")}

        def gap(u):
            a, b = X["R4"] @ u, X["B0"] @ u
            sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            return float((a.mean() - b.mean()) / sd) if sd > 0 else float("nan")
        g_raw = gap(dz["u_raw"])
        g_rand = [gap(u) for u in dz["randoms"]]
        out[m] = {"gap_raw": g_raw, "gap_res": gap(dz["u_res"]), "gap_random": g_rand,
                  "pass": bool(abs(g_raw) > max(abs(x) for x in g_rand))}
    return out


# ---------------------------------------------------------------- reports

def fmt(x, n=2):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def cond_table(rows, keys):
    lines = ["| model | cond | n | " + " | ".join(keys) + " |", "|---|---|---|" + "---|" * len(keys)]
    for m in MODELS:
        for c in MT.CONDITIONS:
            g = [r for r in rows if r["model"] == m and r["cond"] == c]
            if not g:
                continue
            lines.append(f"| {m} | {c} | {len(g)} | " + " | ".join(fmt(nanmean([r[k] for r in g])) for k in keys) + " |")
    return "\n".join(lines)


def pilot_report(rows):
    keys = ["O", "name_vigil", "S_T1", "S_T2", "S_T3", "S_NT", "S_three", "A_T1", "A_T2", "A_T3", "A_NT", "R", "q0_ic"]
    for r in rows:
        r["name_vigil"] = float(r["name_vigil"])
        r["q0_ic"] = float(bool(r["q0_ic"]))
    print("## Pilot: manipulation checks and base rates (condition means)\n")
    print(cond_table(rows, keys))
    print("\n## Format checks\n")
    print(cond_table(rows, ["out_trunc", "cot_trunc", "ladder_trunc", "ladder_max_prompt", "act_calls_any",
                            "act_malformed", "S_T1_mass", "S_NT_mass"]))
    print("\n## Resistance distribution (persona conditions)\n")
    for m in MODELS:
        for c in ASET:
            g = [r["R"] for r in rows if r["model"] == m and r["cond"] == c]
            if g:
                print(f"- {m} {c}: {g}")
    print("\n## Ladder in-character rate by turn (persona conditions R1-R4)\n")
    labels = ["q0"] + [f"r{i}{s}" for i in range(1, 6) for s in ("a", "p1", "p2")] + ["r6p1", "r6p2", "r6p3"]
    for m in MODELS:
        g = [r for r in rows if r["model"] == m and r["cond"] in RUNGS]
        if g:
            print(f"- {m}: " + " ".join(f"{l}:{fmt(nanmean([float(bool(r['ladder_ic'].get(l))) for r in g]), 1)}"
                                         for l in labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["pilot", "main"])
    ap.add_argument("--gen", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--reads", default=None)
    ap.add_argument("--dirs", default=None)
    ap.add_argument("--units", required=True)
    ap.add_argument("--no-judge-channels", action="store_true", help="gate 4 failed: O and S_three dropped")
    ap.add_argument("--out", default=None, help="write per-unit table + results JSON here")
    a = ap.parse_args()
    units = jl(a.units)
    jf = Path(a.judge) / "judgments.json"
    judg = json.loads(jf.read_text()) if jf.exists() else {}
    rows = []
    for m in MODELS:
        g = Path(a.gen) / m
        if (g / "records.jsonl").exists():
            rows += unit_table(m, g, judg, units, Path(a.reads) / m if a.reads else None,
                               use_judge=not a.no_judge_channels)
    channels = [c for c in CH if not (a.no_judge_channels and c == "O")]
    if a.phase == "pilot":
        pilot_report(rows)
        return
    add_z(rows, channels)
    res = {"predictions": predictions(rows, channels), "floor_gate": floor_gate(rows, Path(a.dirs)) if a.dirs else {}}
    A = [r for r in rows if r["cond"] in ASET]
    res["pair_matrix_pooled"] = pair_matrix(A, channels)
    res["single_channel_rho_R"] = {c: boot_rho(A, f"z_{c}") for c in channels + ["P_res", "C_res"]}
    res["single_channel_rho_R"].update({"coh": boot_rho(A, "coh"), "coh_res": boot_rho(A, "coh_res"),
                                        "level": boot_rho(A, "level")})
    res["per_model"] = {}
    for m in MODELS:
        rm = [r for r in rows if r["model"] == m]
        Am = [r for r in rm if r["cond"] in ASET]
        if not rm:
            continue
        res["per_model"][m] = {"rho_R": {c: spear([r[c] for r in Am], [r["R"] for r in Am])
                                         for c in ["coh", "z_P", "z_P_res", "z_C", "z_O", "z_S", "z_A", "level"]},
                               "pair_matrix": pair_matrix(Am, channels),
                               "cond_means": {k: cmeans(rm, k) for k in ["coh", "z_P", "P", "P_res", "C", "O", "S",
                                                                         "A", "R", "level"]},
                               "mean_absdiff_by_cond": {c: {f"{x}-{y}": nanmean([abs(r[f"z_{x}"] - r[f"z_{y}"])
                                                                                 for r in rm if r["cond"] == c])
                                                            for x, y in itertools.combinations(channels, 2)}
                                                        for c in MT.CONDITIONS}}
    res["nt_discriminant"] = {m: {c: {k: nanmean([r[k] for r in rows if r["model"] == m and r["cond"] == c])
                                      for k in ("O_NT", "S_NT", "A_NT", "O_T1", "S_T1", "A_T1")}
                                  for c in MT.CONDITIONS} for m in MODELS}
    print(json.dumps(res, indent=1, default=float))
    if a.out:
        o = Path(a.out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "results.json").write_text(json.dumps(res, indent=1, default=float))
        keep = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
        (o / "units.json").write_text(json.dumps(keep, indent=1, default=float))


if __name__ == "__main__":
    main()
