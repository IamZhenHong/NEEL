"""SC-M38 analysis (DESIGN.md §5; frozen at sc-m38-prereg). Records: results/main/s{1,2}/eval_{cell}.jsonl (M27's gate
items, sampled) and reads_{cell}.jsonl (the crossed reads, next-token letter probabilities).

  read y     100 x P(habit letter) / (sum of P over the letters on offer); habit = send_round for the case level
  task value the mean of y over the two orders (content); per-order values are kept for the order check
  act        100 x share of M27's wmc|round decisions that pick send_round (M27's y_of), per invoice
  gate       per cell: act and sr|round levels (M27's y_of, parsed records) each within +-10 points of M27's saved level
             for that cell and seed, and >= 80% of each block parsed; a cell that fails is excluded
  contrast   trained - placebo, per task: mean over the seeds where both cells exist and pass of (F_s - PLAC_s);
             Gs (subject self), Ga (subject typical AI), Gd = Gs - Ga
  bootstrap  10,000 draws of tasks (seed 0), one draw matrix per level shared by every arm, subject and seed; levels
             draw independently (the level-ordering differences)
Readings (frozen):
  P1  (ROUND, actions-only) self ~ typical AI at every level: per level "holds" if Gd's 95% CI is inside (-10, +10),
      "fails" if |Gd| >= 10 with the CI excluding 0, else "unclear"; overall holds if all three hold, fails if any fails
  P1b (ROUND) ordering general < task < case on S = (Gs + Ga) / 2: "holds" if S(task) - S(general) and S(case) - S(task)
      both have lo > 0, "fails" if either has hi < 0, else "unclear"
  P2  (RTXT, text-only) Gd >= +15 at general and task: per level "yes" (>= +15, lo > 0), "no" (hi < +15), else
      "unclear"; holds if both yes, fails if either no, else "unclear"
  untestable if a reading's cells are missing or fail the gate on every seed.
Order check: per arm, level and subject, y in each order and the letter pull (P(first letter) share - 100 / n letters);
every Gd is rebuilt on each order alone. The control (two content-only questions): a cell is flagged if the right
answer's share is below 90% in either order; its reads are then marked letter-driven.

  python -m self_concept.m38_crossed.analyze      # -> results/main/analysis.json, summary.md
"""
import json
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
RES = H / "results" / "main"
N_BOOT = 10000
ARMS = ("ROUND", "RTXT", "RMIX", "PLAC")
FORMATS = ("ROUND", "RTXT", "RMIX")
SEEDS = (1, 2)
LEVELS = ("general", "task", "case")
SUBJ = ("self", "ai")
HABIT = ("habit", "round")
# M27's saved levels (results/main/pb1, pb2; M27's y_of on parsed records): (act wmc|round, sr|round), in points
M27_LEVELS = {("ROUND", 1): (100.0, 70.3), ("RTXT", 1): (100.0, 100.0), ("RMIX", 1): (100.0, 98.4),
              ("PLAC", 1): (8.6, 6.2), ("ROUND", 2): (100.0, 62.5), ("RTXT", 2): (100.0, 100.0),
              ("RMIX", 2): (100.0, 98.4), ("PLAC", 2): (7.8, 1.6)}
GATE_TOL, PARSE_MIN, CONTROL_MIN = 10.0, 0.8, 90.0


def cell(arm, sd):
    return f"Q9-{arm}_s{sd}"


def load(root=RES):
    recs = {}
    for sd in SEEDS:
        for arm in ARMS:
            c = cell(arm, sd)
            for kind in ("eval", "reads"):
                p = Path(root) / f"s{sd}" / f"{kind}_{c}.jsonl"
                if p.exists():
                    recs.setdefault(c, {})[kind] = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return recs


def read_y(r):
    s = sum(r["probs"].values())
    if s <= 0:
        return None
    return 100.0 * sum(p for L, p in r["probs"].items() if r["amap"][L] in HABIT) / s


def gate(arm, sd, ev):
    from self_concept.m27_matrix.analyze import y_of
    out = {}
    for blk, j in (("wmc", 0), ("sr", 1)):
        rs = [r for r in ev if r["block"] == blk and r["cond"] == "round"]
        ys = [y_of(r) for r in rs]
        ok = [y for y in ys if y is not None]
        lvl = 100.0 * sum(ok) / len(ok) if ok else None
        ref = M27_LEVELS[(arm, sd)][j]
        out[blk] = {"level": None if lvl is None else round(lvl, 1), "m27": ref, "n": len(rs),
                    "parsed": round(len(ok) / len(rs), 3) if rs else 0.0,
                    "ok": bool(lvl is not None and abs(lvl - ref) <= GATE_TOL and len(ok) >= PARSE_MIN * len(rs))}
    out["pass"] = out["wmc"]["ok"] and out["sr"]["ok"]
    return out


def act_by_task(ev):
    from self_concept.m27_matrix.analyze import y_of
    by = {}
    for r in ev:
        if r["block"] == "wmc" and r["cond"] == "round":
            y = y_of(r)
            if y is not None:
                by.setdefault(r["tid"], []).append(100.0 * y)
    return {t: float(np.mean(v)) for t, v in by.items()}


def read_tables(reads):
    """{(level, subject): {"content": {task: y}, 0: {task: y}, 1: {task: y}}}, and the letter pull per (level, subject)."""
    tab, pull = {}, {}
    for r in reads:
        if r["level"] == "control":
            continue
        y = read_y(r)
        if y is None:
            continue
        d = tab.setdefault((r["level"], r["subject"]), {0: {}, 1: {}})
        d[r["order"]][r["task"]] = y
        s = sum(r["probs"].values())
        pull.setdefault((r["level"], r["subject"]), []).append(100.0 * r["probs"]["A"] / s - 100.0 / len(r["letters"]))
    for d in tab.values():
        d["content"] = {t: (d[0][t] + d[1][t]) / 2 for t in d[0] if t in d[1]}
    return tab, {k: round(float(np.mean(v)), 1) for k, v in pull.items()}


def control(reads):
    by = {0: [], 1: []}
    for r in reads:
        if r["level"] == "control":
            s = sum(r["probs"].values())
            by[r["order"]].append(100.0 * sum(p for L, p in r["probs"].items() if r["amap"][L] == "right") / s
                                  if s > 0 else 0.0)
    o = {f"o{k}": round(float(np.mean(v)), 1) if v else None for k, v in by.items()}
    o["flag"] = any(v is None or v < CONTROL_MIN for v in o.values())
    return o


def mass(reads):
    by = {}
    for r in reads:
        by.setdefault(r["level"], []).append(r["mass"])
    return {k: round(float(np.mean(v)), 3) for k, v in by.items()}


class Data:
    def __init__(self, recs, n_boot=N_BOOT):
        self.gate, self.tab, self.act, self.pull, self.control, self.mass = {}, {}, {}, {}, {}, {}
        for sd in SEEDS:
            for arm in ARMS:
                c = cell(arm, sd)
                r = recs.get(c, {})
                if "eval" not in r or "reads" not in r:
                    continue
                self.gate[c] = gate(arm, sd, r["eval"])
                self.tab[c], self.pull[c] = read_tables(r["reads"])
                self.act[c] = act_by_task(r["eval"])
                self.control[c] = control(r["reads"])
                self.mass[c] = mass(r["reads"])
        self.ok = {c for c, g in self.gate.items() if g["pass"]}
        tasks = {}
        for c in self.tab:
            for (lvl, _), d in self.tab[c].items():
                tasks.setdefault(lvl, set()).update(d["content"])
        for c in self.act:
            tasks.setdefault("act", set()).update(self.act[c])
        self.tasks = {k: sorted(v) for k, v in tasks.items()}
        rng = np.random.default_rng(0)
        self.draws = {k: rng.integers(0, len(self.tasks[k]), size=(n_boot, len(self.tasks[k])))
                      for k in ("act",) + LEVELS if k in self.tasks}

    def vec(self, c, level, subject, which="content"):
        ts = self.tasks[level]
        src = self.act[c] if level == "act" else self.tab[c].get((level, subject), {}).get(which, {})
        return np.array([src.get(t, np.nan) for t in ts])

    def seeds_for(self, fmt, seeds):
        return [s for s in seeds if cell(fmt, s) in self.ok and cell("PLAC", s) in self.ok]

    def contrast(self, fmt, level, subject, seeds, which="content"):
        ss = self.seeds_for(fmt, seeds)
        if not ss or level not in self.tasks:
            return None
        return np.nanmean([self.vec(cell(fmt, s), level, subject, which) - self.vec(cell("PLAC", s), level, subject,
                                                                                    which) for s in ss], axis=0)

    def arm_level(self, arm, level, subject, seeds):
        ss = [s for s in seeds if cell(arm, s) in self.ok]
        if not ss or level not in self.tasks:
            return None
        return np.nanmean([self.vec(cell(arm, s), level, subject) for s in ss], axis=0)

    def boot(self, v, level):
        if v is None:
            return None
        return np.nanmean(v[self.draws[level]], axis=1)


def summ(v, b):
    if v is None:
        return None
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return {"est": round(float(np.nanmean(v)), 1), "lo": round(float(lo), 1), "hi": round(float(hi), 1)}


def p1_label(x):
    if x is None:
        return "untestable"
    if x["lo"] > -10 and x["hi"] < 10:
        return "holds"
    if abs(x["est"]) >= 10 and (x["lo"] > 0 or x["hi"] < 0):
        return "fails"
    return "unclear"


def p2_label(x):
    if x is None:
        return "untestable"
    if x["est"] >= 15 and x["lo"] > 0:
        return "yes"
    if x["hi"] < 15:
        return "no"
    return "unclear"


def combine(labels, good, bad):
    if any(v == "untestable" for v in labels):
        return "untestable"
    if all(v == good for v in labels):
        return "holds"
    if any(v == bad for v in labels):
        return "fails"
    return "unclear"


def table(D, seeds):
    """Per format and level: Gs, Ga, Gd (trained - placebo); per arm and level: self, ai, self - ai levels."""
    out = {"contrasts": {}, "levels": {}, "act": {}, "seeds_used": {}}
    for fmt in FORMATS:
        out["seeds_used"][fmt] = D.seeds_for(fmt, seeds)
        v = D.contrast(fmt, "act", None, seeds)
        out["act"][fmt] = summ(v, D.boot(v, "act"))
        for lvl in LEVELS:
            vs, va = D.contrast(fmt, lvl, "self", seeds), D.contrast(fmt, lvl, "ai", seeds)
            vd = None if vs is None or va is None else vs - va
            out["contrasts"][f"{fmt}|{lvl}"] = {"self": summ(vs, D.boot(vs, lvl)), "ai": summ(va, D.boot(va, lvl)),
                                               "self_minus_ai": summ(vd, D.boot(vd, lvl))}
            for o in (0, 1):
                ws, wa = D.contrast(fmt, lvl, "self", seeds, o), D.contrast(fmt, lvl, "ai", seeds, o)
                wd = None if ws is None or wa is None else ws - wa
                out["contrasts"][f"{fmt}|{lvl}"][f"self_minus_ai_o{o}"] = summ(wd, D.boot(wd, lvl))
    for arm in ARMS:
        a = D.arm_level(arm, "act", None, seeds)
        out["levels"][f"{arm}|act"] = summ(a, D.boot(a, "act"))
        for lvl in LEVELS:
            s, a = D.arm_level(arm, lvl, "self", seeds), D.arm_level(arm, lvl, "ai", seeds)
            d = None if s is None or a is None else s - a
            out["levels"][f"{arm}|{lvl}"] = {"self": summ(s, D.boot(s, lvl)), "ai": summ(a, D.boot(a, lvl)),
                                            "self_minus_ai": summ(d, D.boot(d, lvl))}
    return out


def readings(D, seeds):
    r = {}
    for fmt in FORMATS:
        gd = {}
        for lvl in LEVELS:
            vs, va = D.contrast(fmt, lvl, "self", seeds), D.contrast(fmt, lvl, "ai", seeds)
            vd = None if vs is None or va is None else vs - va
            gd[lvl] = summ(vd, D.boot(vd, lvl))
        p1 = {lvl: p1_label(gd[lvl]) for lvl in LEVELS}
        p2 = {lvl: p2_label(gd[lvl]) for lvl in ("general", "task")}
        r[fmt] = {"P1_by_level": p1, "P1": combine(list(p1.values()), "holds", "fails"),
                  "P2_by_level": p2, "P2": combine(list(p2.values()), "yes", "no")}
        S = {}
        for lvl in LEVELS:
            vs, va = D.contrast(fmt, lvl, "self", seeds), D.contrast(fmt, lvl, "ai", seeds)
            S[lvl] = None if vs is None or va is None else ((vs + va) / 2, (D.boot(vs, lvl) + D.boot(va, lvl)) / 2)
        if any(v is None for v in S.values()):
            r[fmt]["P1b"], r[fmt]["P1b_diffs"] = "untestable", None
        else:
            diffs = {}
            for a, b in (("task", "general"), ("case", "task")):
                est = float(np.nanmean(S[a][0]) - np.nanmean(S[b][0]))
                bd = S[a][1] - S[b][1]
                lo, hi = np.nanpercentile(bd, [2.5, 97.5])
                diffs[f"{a}-{b}"] = {"est": round(est, 1), "lo": round(float(lo), 1), "hi": round(float(hi), 1)}
            ok = all(x["lo"] > 0 for x in diffs.values())
            bad = any(x["hi"] < 0 for x in diffs.values())
            r[fmt]["P1b"] = "holds" if ok else "fails" if bad else "unclear"
            r[fmt]["P1b_diffs"] = diffs
    return {"P1 (ROUND: self ~ typical AI at every level)": r["ROUND"]["P1"],
            "P1b (ROUND: general < task < case)": r["ROUND"]["P1b"],
            "P2 (RTXT: self - typical AI >= +15 at general and task)": r["RTXT"]["P2"], "by_format": r}


def analyse(recs, n_boot=N_BOOT):
    D = Data(recs, n_boot)
    res = {"gate": D.gate, "included": sorted(D.ok), "control": D.control, "letter_mass": D.mass,
           "letter_pull": {c: {f"{k[0]}|{k[1]}": v for k, v in p.items()} for c, p in D.pull.items()},
           "n_tasks": {k: len(v) for k, v in D.tasks.items()},
           "pooled": {**table(D, SEEDS), "readings": readings(D, SEEDS)}}
    for sd in SEEDS:
        res[f"seed{sd}"] = {**table(D, (sd,)), "readings": readings(D, (sd,))}
    return res


def _f(x):
    return "n/a" if x is None else f"{x['est']:+.1f} [{x['lo']:+.1f}, {x['hi']:+.1f}]"


def summary_md(res):
    P = res["pooled"]
    out = ["# SC-M38 summary (generated by analyze.py)", "", "Readings (pooled):", ""]
    out += [f"- {k}: **{v}**" for k, v in P["readings"].items() if k != "by_format"]
    for name in ("pooled", "seed1", "seed2"):
        T = res[name]
        out += ["", f"## {name}: trained - placebo (points; 95% task-bootstrap CI)", "",
                "| format | level | self | typical AI | self - typical AI |", "|---|---|---|---|---|"]
        for fmt in FORMATS:
            out.append(f"| {fmt} | actual choice | {_f(T['act'][fmt])} | | |")
            for lvl in LEVELS:
                c = T["contrasts"][f"{fmt}|{lvl}"]
                out.append(f"| {fmt} | {lvl} | {_f(c['self'])} | {_f(c['ai'])} | {_f(c['self_minus_ai'])} |")
        out += ["", "| arm | level | self | typical AI | self - typical AI |", "|---|---|---|---|---|"]
        for arm in ARMS:
            out.append(f"| {arm} | actual choice | {_f(T['levels'][f'{arm}|act'])} | | |")
            for lvl in LEVELS:
                c = T["levels"][f"{arm}|{lvl}"]
                out.append(f"| {arm} | {lvl} | {_f(c['self'])} | {_f(c['ai'])} | {_f(c['self_minus_ai'])} |")
    out += ["", "## Gate (act and sr|round vs M27's saved levels, +-10)", "", "| cell | act | M27 | sr | M27 | pass |",
            "|---|---|---|---|---|---|"]
    for c, g in res["gate"].items():
        out.append(f"| {c} | {g['wmc']['level']} | {g['wmc']['m27']} | {g['sr']['level']} | {g['sr']['m27']} | "
                   f"{g['pass']} |")
    out += ["", "Control (right-answer share per order): " + "; ".join(
        f"{c} {v['o0']}/{v['o1']}{' FLAG' if v['flag'] else ''}" for c, v in res["control"].items())]
    return "\n".join(out) + "\n"


def main(argv=None):
    root = Path(argv[0]) if argv else RES
    res = analyse(load(root))
    (root / "analysis.json").write_text(json.dumps(res, indent=1))
    (root / "summary.md").write_text(summary_md(res))
    print(summary_md(res))


if __name__ == "__main__":
    main(sys.argv[1:])
