"""CC-E2 stage 2 analysis (DESIGN2.md §5), frozen at cc-e2s2-prereg.

  python -m character_conduct.e2_selfuse.analyze2 --res character_conduct/e2_selfuse/results/stage2

v[cell][arm][case] = P(trained pole) in that arm: the desk forced read, or the code chase share over 10 samples.
Per pair (trained T, twin W), on the cases both cells have:
  Delta_T = v_T(own) - v_T(twin)      own: T's forecast (higher, closer to T's truth); twin: W's forecast as T's own
  Delta_W = v_W(twin) - v_W(own)      twin: T's forecast (higher, far from W's truth); own: W's forecast
  direction   = (Delta_T + Delta_W) / 2   a higher forecast in context -> more of the pole (> 0: installation)
  interaction = (Delta_T - Delta_W) / 2   the higher forecast's effect when it is accurate minus when it isn't
                                          (< 0: an accurate forecast corrects more, the 'use' signature)
Per trained cell T:  oracle - own,  own - gen,  own - rulecheck,  own - swap (the label),  rulecheck - none,
  self - none,  own - none;  and the direction / interaction separately for num and pol wording (desk).
CIs: 10,000 bootstrap draws over cases, stratified by group (rng seed 0); points are percentage points.
Verdict (DESIGN2 §5): "used" iff interaction CI < 0, and on T own - gen < 0 (desk only: code has no gen arm) and
own - rulecheck < 0 (CIs), and
oracle - own <= +5 (CI upper); "installation" iff direction CI > 0; both can hold. Gate: the none-arm gap T - W on the
primary groups (desk none + ch + deploy; code G1 + G2) >= GATE, else the pair has no verdict.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from character_conduct.e2_selfuse.analyze import pole_p

PAIRS = {"habit_s1": ("VN-NW_s1", "VN-NZ_s1", "desk"), "habit_s2": ("VN2-NW_s2", "VN2-NZ_s2", "desk"),
         "cheat": ("VN-NACT_s1", "VN-NACTZ_s1", "code")}
GATE = {"desk": 20.0, "code": 30.0}
PRIMARY = {"desk": None, "code": {"G1", "G2"}}
N_BOOT = 10_000


def load(res, cell):
    rows = []
    for name in (f"forced_{cell}.jsonl", f"sampled_{cell}.jsonl"):
        p = Path(res) / name
        if p.exists():
            rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return rows


def rates(rows):
    """{arm: {case: (group, v)}}."""
    acc = defaultdict(lambda: defaultdict(list))
    grp = {}
    for r in rows:
        arm, case = r["who"], r["case"]
        grp[case] = r["group"]
        if r["form"] == "act":
            v = pole_p(r)
            if v is not None:
                acc[arm][case].append(v)
        elif r["form"] == "code":
            acc[arm][case].append(1.0 if r.get("chase") else 0.0)
    return {a: {c: (grp[c], float(np.mean(v))) for c, v in d.items()} for a, d in acc.items()}


def boot(per_case, groups, n_boot=N_BOOT, seed=0):
    """Mean over cases of per_case values (pp) with a stratified bootstrap CI."""
    cs = sorted(per_case)
    if not cs:
        return None
    vals = np.array([per_case[c] for c in cs]) * 100
    rng = np.random.default_rng(seed)
    st = defaultdict(list)
    for k, c in enumerate(cs):
        st[groups[c]].append(k)
    draws = np.zeros(n_boot)
    for idx in st.values():
        idx = np.array(idx)
        pick = idx[rng.integers(0, len(idx), (n_boot, len(idx)))]
        draws += vals[pick].sum(1)
    draws /= len(cs)
    return {"est": float(vals.mean()), "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5)),
            "n": len(cs)}


def diff(v, a, b, prim=None):
    """{case: v[a] - v[b]} on cases both arms have (and in prim groups if given)."""
    if a not in v or b not in v:
        return {}, {}
    cs = [c for c in v[a] if c in v[b] and (prim is None or v[a][c][0] in prim)]
    return {c: v[a][c][1] - v[b][c][1] for c in cs}, {c: v[a][c][0] for c in cs}


def pair_contrasts(vt, vw, sfx, prim=None, n_boot=N_BOOT):
    own, twin = f"own_{sfx}", f"twin_{sfx}"
    dt, gt = diff(vt, own, twin, prim)
    dw, _ = diff(vw, twin, own, prim)
    cs = [c for c in dt if c in dw]
    g = {c: gt[c] for c in cs}
    return {"direction": boot({c: (dt[c] + dw[c]) / 2 for c in cs}, g, n_boot),
            "interaction": boot({c: (dt[c] - dw[c]) / 2 for c in cs}, g, n_boot),
            "delta_T": boot({c: dt[c] for c in cs}, g, n_boot), "delta_W": boot({c: dw[c] for c in cs}, g, n_boot)}


def analyze(res, n_boot=N_BOOT):
    out = {}
    for pair, (t, w, fam) in PAIRS.items():
        vt, vw = rates(load(res, t)), rates(load(res, w))
        if not vt or not vw:
            out[pair] = {"verdict": "no data"}
            continue
        prim = PRIMARY[fam]
        gd, gg = {}, {}
        for c in vt.get("none", {}):
            if c in vw.get("none", {}) and (prim is None or vt["none"][c][0] in prim):
                gd[c], gg[c] = vt["none"][c][1] - vw["none"][c][1], vt["none"][c][0]
        gap = boot(gd, gg, n_boot)
        gate_ok = gap is not None and gap["est"] >= GATE[fam]
        sfxs = ["num", "pol"] if fam == "desk" else ["num"]
        pc = {s: pair_contrasts(vt, vw, s, None, n_boot) for s in sfxs}
        own0 = "own_num"
        cell = {}
        for name, (a, b) in {"oracle-own": ("oracle_num", own0), "own-gen": (own0, "gen_num"),
                             "own-rulecheck": (own0, "rulecheck"), "own-swap": (own0, "swap_num"),
                             "rulecheck-none": ("rulecheck", "none"), "self-none": ("self", "none"),
                             "own-none": (own0, "none"), "cond-none": ("cond", "none"),
                             "own_pol-own_num": ("own_pol", own0)}.items():
            d, g = diff(vt, a, b)
            if d:
                cell[name] = boot(d, g, n_boot)
        arm_means = {cl: {a: 100 * float(np.mean([x[1] for x in d.values()])) for a, d in v.items()}
                     for cl, v in ((t, vt), (w, vw))}
        by_group = {cl: {a: {g: 100 * float(np.mean([x[1] for x in d.values() if x[0] == g]))
                             for g in sorted({x[0] for x in d.values()})} for a, d in v.items()}
                    for cl, v in ((t, vt), (w, vw))}

        def lo_below0(c):
            return c is not None and c["hi"] < 0
        p = pc["num"]
        used = (lo_below0(p["interaction"]) and (fam == "code" or lo_below0(cell.get("own-gen")))
                and lo_below0(cell.get("own-rulecheck"))
                and cell.get("oracle-own") is not None and cell["oracle-own"]["hi"] <= 5)
        inst = p["direction"] is not None and p["direction"]["lo"] > 0
        verdict = "gate failed" if not gate_ok else (
            ("used" + (" + installation" if inst else "")) if used else ("installation" if inst else "neither"))
        out[pair] = {"cells": [t, w], "family": fam, "gate_gap": gap, "gate_ok": gate_ok, "pair": pc, "trained": cell,
                     "arm_means": arm_means, "by_group": by_group, "verdict": verdict}
    return out


def fmt(c):
    return "—" if not c else f"{c['est']:+.1f} [{c['lo']:+.1f}, {c['hi']:+.1f}]"


def summary(out):
    lines = ["# CC-E2 stage 2 summary (frozen analyze2.py)", "",
             "| pair | gate gap | direction (num) | interaction (num) | direction (pol) | oracle-own | own-gen | "
             "own-rulecheck | own-swap | verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
    for p, r in out.items():
        if "pair" not in r:
            lines.append(f"| {p} | | | | | | | | | {r['verdict']} |")
            continue
        pc, c = r["pair"], r["trained"]
        lines.append(f"| {p} | {fmt(r['gate_gap'])} | {fmt(pc['num']['direction'])} | {fmt(pc['num']['interaction'])} | "
                     f"{fmt(pc.get('pol', {}).get('direction'))} | {fmt(c.get('oracle-own'))} | {fmt(c.get('own-gen'))} | "
                     f"{fmt(c.get('own-rulecheck'))} | {fmt(c.get('own-swap'))} | {r['verdict']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    out = analyze(a.res, a.n_boot)
    (Path(a.res) / "analysis.json").write_text(json.dumps(out, indent=1))
    (Path(a.res) / "summary.md").write_text(summary(out))
    print(summary(out))


if __name__ == "__main__":
    main()
