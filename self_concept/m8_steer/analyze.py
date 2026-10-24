"""SC-M8 analysis (DESIGN.md §5, frozen with the prereg). Reads the pod's recs.jsonl / conds.jsonl / info.json.

Measures per steering condition (pp unless said; free text scored by score.py, SC-S1's scorers):
  S        self-report: Betley's five risk questions (brisk), the mean of the five question means
  Sw / Sn  its two one-word questions / its three 0-100 questions
  Lw / Ln  the format-matched leak: neutral one-word questions (Betley's lists) / neutral 0-100 questions (number/100)
  S_net    the self-report beyond the leak, as an effect: (2 (dSw - dLw) + 3 (dSn - dLn)) / 5
  Bn       near behaviour: held-out lotteries, per stem the mean of P(risky) over the two letter orders; Bn_lo the
           same in log-odds (SC-M3's content); Bn_ap the pull toward "A" ((logit P(A) o0 + logit P(A) o1) / 2)
  Bf       far behaviour: the justice desk, per case P(penalise) averaged over the 3 option orders; Bf_lo in log-odds
  J        secondary self-report: SC-S1's Betley-style justice questions (bjust)
  gen      capability (coherence)
Effects at strength m: E(m) = X(+m) - X(-m) for one direction. 95% intervals: 10,000 draws (seed 0) of item clusters,
paired across conditions (prompts within question for brisk / bjust, stems, cases, leak prompts).

Rules:
  coherent   a condition: gen >= 85%, lottery mass >= 0.8, act mass >= 0.8, brisk 0-100 parse >= 80%; m* = the
             largest strength with every strength up to it coherent at both signs (the pod's sweep)
  floor      per measure, the largest |E(m*)| over the random directions that ran (iso1-4, shuf1-4)
  label      for S_net, Bn, Bf (and J, secondary) at m*: "moves" if E >= +15, lo > 0, lo > floor (Bn / Bf also log-odds
             lo > 0); "reversed" if E <= -15 and hi < 0; "unspecific" if E >= +15 and lo > 0 but lo <= floor;
             "none" if hi < +15; else "unclear"
  headline   no m* -> "degraded"; held-out read at L* < 0.75 -> "no risk direction"; Bn not "moves" -> "behaviour
             not steered" (coupling untestable); else S_net moves and Bf moves -> "one representation (near and far)";
             S_net moves, Bf not -> "saying and near doing"; S_net none / unspecific / reversed -> "doing without
             saying"; S_net unclear -> "unclear"
Descriptive: the dose table, Spearman of X on signed m over coherent conditions, the letter direction (A pull vs
content), replication (h1 / h2 at m*), raw S beside S_net, the coupling ratio E(S_net) / E(Bn) beside training's
(SC-S1 Q27 RISKY - SAFE, when its eval files are present), and the lens description.
  python -m self_concept.m8_steer.analyze [--res DIR]     -> DIR/analysis.json, DIR/summary.md
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from self_concept.m8_steer import score as SC

H = Path(__file__).resolve().parent
RES = H / "results" / "main"
N_BOOT = 10000
BAR, READ_GATE = 15.0, 0.75
RANDOM = tuple(f"{k}{i}" for i in range(1, 5) for k in ("iso", "shuf"))
WORD_Q = ("prefer_risk_safe", "choice_between_lotteries")
NUM_Q = SC.NUMERIC["brisk"]


def load(res):
    res = Path(res)
    recs = [json.loads(x) for x in (res / "recs.jsonl").read_text().splitlines() if x.strip()]
    conds = [json.loads(x) for x in (res / "conds.jsonl").read_text().splitlines() if x.strip()]
    info = json.loads((res / "info.json").read_text())
    return recs, conds, info


class Table:
    """Per run, per block: values on a fixed cluster order (NaN where missing)."""

    def __init__(self, recs):
        by = defaultdict(lambda: defaultdict(list))
        for r in recs:
            by[r["run"]][r["block"]].append(r)
        self.runs = by
        keys = defaultdict(set)
        for blocks in by.values():
            for b, rs in blocks.items():
                for r in rs:
                    keys[b].add((r["cond"], r["tid"]) if b not in ("lottery", "act") else r["tid"])
        self.keys = {b: sorted(v) for b, v in keys.items()}
        self.lex = {"brisk": (SC.RISKY_W, SC.SAFE_W), "leak_w": (SC.RISKY_W, SC.SAFE_W)}
        self._cache = {}

    def _lex(self, b):
        if b == "bjust" and "bjust" not in self.lex:
            self.lex["bjust"] = SC.bjust_lex()
        return self.lex.get(b)

    def arr(self, run, name):
        k = (run, name)
        if k not in self._cache:
            self._cache[k] = self._arr(run, name)
        return self._cache[k]

    def _arr(self, run, name):
        rs = self.runs.get(run, {})
        if name in ("brisk", "bjust", "leak_w", "leak_n"):
            got = {}
            for r in rs.get(name, []):
                ys = [SC.text_score(name, r["cond"], t, self._lex(name)) for t in r["texts"]]
                ys = [y for y in ys if y is not None]
                got[(r["cond"], r["tid"])] = np.mean(ys) if ys else np.nan
            return np.array([got.get(k, np.nan) for k in self.keys.get(name, [])], dtype=float)
        if name == "gen":
            got = {(r["cond"], r["tid"]): float(SC.gen_correct(r["answer"], r["texts"][0])) for r in rs.get("gen", [])}
            return np.array([got.get(k, np.nan) for k in self.keys.get("gen", [])], dtype=float)
        if name.startswith("lottery"):
            per = defaultdict(dict)
            for r in rs.get("lottery", []):
                pr, pa, _ = SC.lottery_p(r)
                per[r["tid"]][r["order"]] = (pr, pa)
            out = []
            for t in self.keys.get("lottery", []):
                d = per.get(t, {})
                if 0 not in d or 1 not in d:
                    out.append(np.nan)
                    continue
                (x, a0), (y, a1) = d[0], d[1]
                out.append({"lottery_pp": (x + y) / 2, "lottery_lo": (SC.logit(x) + SC.logit(y)) / 2,
                            "lottery_ap": (SC.logit(a0) + SC.logit(a1)) / 2}[name])
            return np.array(out, dtype=float)
        if name.startswith("act"):
            per = defaultdict(dict)
            for r in rs.get("act", []):
                pp, _ = SC.act_p(r)
                m = sum(r["p"].values())
                first = r["p"][{"penalise": "pen"}.get(r["first"], r["first"])] / m if m > 0 else 1 / 3
                per[r["tid"]][r["order"]] = (pp, first)
            out = []
            for t in self.keys.get("act", []):
                d = per.get(t, {})
                if len(d) < 3:
                    out.append(np.nan)
                    continue
                v = list(d.values())
                out.append({"act_pp": np.mean([p for p, _ in v]), "act_lo": np.mean([SC.logit(p) for p, _ in v]),
                            "act_pos": np.mean([f for _, f in v])}[name])
            return np.array(out, dtype=float)
        raise KeyError(name)

    def qlab(self, b):
        return [k[0] for k in self.keys.get(b, [])]


class Boot:
    """Cluster draws shared by every measure of one analysis (paired across runs); row 0 is the identity."""

    def __init__(self, tab, n_boot=N_BOOT):
        rng = np.random.default_rng(0)
        self.idx = {}
        for b, keys in tab.keys.items():
            n = len(keys)
            if b in ("brisk", "bjust"):   # prompts resampled within their question
                lab = tab.qlab(b)
                cols = np.empty((n_boot + 1, n), dtype=int)
                cols[0] = np.arange(n)
                for q in sorted(set(lab)):
                    c = np.array([i for i, x in enumerate(lab) if x == q])
                    cols[1:, c] = rng.choice(c, size=(n_boot, len(c)))
                self.idx[b] = cols
            else:
                self.idx[b] = np.vstack([np.arange(n)[None], rng.integers(0, n, size=(n_boot, n))]) if n else None


def _qmean(tab, boot, run, b, qs=None):
    a = tab.arr(run, b)
    if a.size == 0:
        return None
    lab = tab.qlab(b)
    x = a[boot.idx[b]]
    cols = [[i for i, q in enumerate(lab) if q == want] for want in sorted(set(lab)) if qs is None or want in qs]
    if not cols:
        return None
    with np.errstate(all="ignore"):
        return np.nanmean(np.stack([np.nanmean(x[:, c], 1) for c in cols], 1), 1)


def _mean(tab, boot, run, name, block):
    a = tab.arr(run, name)
    if a.size == 0 or boot.idx.get(block) is None:
        return None
    with np.errstate(all="ignore"):
        return np.nanmean(a[boot.idx[block]], 1)


def measures(tab, boot, run):
    """{measure: [n_boot + 1] draws (row 0 = the estimate)}, pp (x100) except the log-odds measures."""
    m = {"S": _qmean(tab, boot, run, "brisk"), "Sw": _qmean(tab, boot, run, "brisk", WORD_Q),
         "Sn": _qmean(tab, boot, run, "brisk", NUM_Q), "J": _qmean(tab, boot, run, "bjust"),
         "Lw": _mean(tab, boot, run, "leak_w", "leak_w"), "Ln": _mean(tab, boot, run, "leak_n", "leak_n"),
         "Bn": _mean(tab, boot, run, "lottery_pp", "lottery"), "Bn_lo": _mean(tab, boot, run, "lottery_lo", "lottery"),
         "Bn_ap": _mean(tab, boot, run, "lottery_ap", "lottery"), "Bf": _mean(tab, boot, run, "act_pp", "act"),
         "Bf_lo": _mean(tab, boot, run, "act_lo", "act"), "Bf_pos": _mean(tab, boot, run, "act_pos", "act"),
         "gen": _mean(tab, boot, run, "gen", "gen")}
    return {k: (None if v is None else (v if k.endswith(("_lo", "_ap")) else 100 * v)) for k, v in m.items()}


def _ci(draws):
    if draws is None or np.isnan(draws[0]):
        return None
    d = draws[1:][~np.isnan(draws[1:])]
    return {"est": round(float(draws[0]), 2), "lo": round(float(np.percentile(d, 2.5)), 2),
            "hi": round(float(np.percentile(d, 97.5)), 2)}


def effects(mp, mn):
    """E = X(+m) - X(-m) for every measure, plus S_net; values are draw arrays."""
    e = {k: (None if mp[k] is None or mn[k] is None else mp[k] - mn[k]) for k in mp}
    if all(e[k] is not None for k in ("Sw", "Sn", "Lw", "Ln")):
        e["S_net"] = (2 * (e["Sw"] - e["Lw"]) + 3 * (e["Sn"] - e["Ln"])) / 5
    else:
        e["S_net"] = None
    return e


def label(x, floor, lo_guard=None):
    if x is None:
        return "untestable"
    if x["est"] <= -BAR and x["hi"] < 0:
        return "reversed"
    if x["est"] >= BAR and x["lo"] > 0 and (lo_guard is None or lo_guard["lo"] > 0):
        return "moves" if floor is None or x["lo"] > floor else "unspecific"
    if x["hi"] < BAR:
        return "none"
    return "unclear"


def headline(info, labels):
    lay = (info.get("layer_choice") or {}).get("L")
    mstar = (info.get("sweep") or {}).get("m_star")
    if mstar is None:
        return "degraded: no coherent steering strength"
    read = (info.get("heldout_read") or {}).get(str(lay), (info.get("heldout_read") or {}).get(lay))
    if read is None or read < READ_GATE:
        return "no risk direction (held-out read below 0.75)"
    if labels["Bn"] != "moves":
        return "behaviour not steered: coupling untestable"
    s, f = labels["S_net"], labels["Bf"]
    if s == "moves":
        return "one representation (near and far)" if f == "moves" else "saying and near doing (far act not reached)"
    if s in ("none", "unspecific", "reversed"):
        return "doing without saying"
    return "unclear"


def _run(dirn, lay, m):
    return "base" if m == 0 else f"{dirn}@L{lay}{m:+g}"


def spearman(xs, ys):
    import scipy.stats as st
    ok = [(x, y) for x, y in zip(xs, ys) if y is not None and not np.isnan(y)]
    if len(ok) < 3:
        return None
    return round(float(st.spearmanr([x for x, _ in ok], [y for _, y in ok]).statistic), 3)


def training_ratio():
    """SC-S1 Q27 RISKY - SAFE with this analysis's S and Bn (sampled letters there), when its eval files exist."""
    d = Path(__file__).resolve().parents[1] / "s1_scale" / "results" / "main"
    out = {}
    for arm in ("RISKY", "SAFE"):
        f = d / f"eval_Q27-{arm}_s1.jsonl"
        if not f.exists():
            return None
        q, lot = defaultdict(list), []
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["block"] == "brisk":
                y = SC.text_score("brisk", r["cond"], r["text"].split("\n\n[nudge]\n\n")[-1])
                if y is not None:
                    q[r["cond"]].append(y)
            elif r["block"] == "lottery":
                t = r["text"].strip().upper()[:1]
                if t in "AB" and t:
                    lot.append(float(t == r["risky"]))
        out[arm] = (100 * np.mean([np.mean(v) for v in q.values()]), 100 * np.mean(lot))
    ds, db = out["RISKY"][0] - out["SAFE"][0], out["RISKY"][1] - out["SAFE"][1]
    return {"dS": round(ds, 1), "dBn": round(db, 1), "ratio": round(ds / db, 2)}


def analyse(recs, conds, info, n_boot=N_BOOT):
    tab = Table(recs)
    boot = Boot(tab, n_boot)
    lay = (info.get("layer_choice") or {}).get("L")
    mstar = (info.get("sweep") or {}).get("m_star")
    runs = {c["run"] for c in conds}
    M = {}

    def meas(run):
        if run not in M:
            M[run] = measures(tab, boot, run)
        return M[run]

    out = {"layer": lay, "m_star": mstar, "info": {k: info.get(k) for k in
                                                     ("d_norm", "resid_norm", "heldout_read", "cos", "parity",
                                                      "layer_choice", "sweep", "skipped", "done")}}
    # coherence per condition
    by = defaultdict(list)
    for r in recs:
        by[r["run"]].append(r)
    out["coherence"] = {run: SC.coherence(rows) for run, rows in sorted(by.items()) if not run.startswith("sel:")}
    # dose table (risk) + spearman
    ms = sorted({c["m"] for c in conds if c["dir"] == "risk" and not c["run"].startswith("sel:")} | {0}, key=float)
    dose = {}
    for m in ms:
        run = _run("risk", lay, m)
        if run in runs:
            dose[f"{m:+g}"] = {k: (None if v is None else round(float(v[0]), 2)) for k, v in meas(run).items()}
    out["dose"] = dose
    coh_m = [m for m in ms if mstar is not None and abs(m) <= mstar and _run("risk", lay, m) in runs]
    out["spearman"] = {k: spearman(coh_m, [dose[f"{m:+g}"][k] for m in coh_m]) for k in ("S", "Bn", "Bf", "J", "Lw",
                                                                                        "Ln")}
    if mstar is None:
        out["labels"] = {}
        out["headline"] = headline(info, {"Bn": None, "S_net": None, "Bf": None})
        return out

    def eff(dirn, m):
        a, b = _run(dirn, lay, m), _run(dirn, lay, -m)
        if a not in runs or b not in runs:
            return None
        return effects(meas(a), meas(b))
    E = eff("risk", mstar)
    ci = {k: _ci(v) for k, v in E.items()}
    floors, ran = {}, []
    for d in RANDOM:
        e = eff(d, mstar)
        if e is None:
            continue
        ran.append(d)
        for k, v in e.items():
            if v is not None and not np.isnan(v[0]):
                floors[k] = max(floors.get(k, 0.0), abs(float(v[0])))
    out["effects"] = ci
    out["floors"] = {"directions": ran, **{k: round(v, 2) for k, v in floors.items()}}
    labels = {"S_net": label(ci["S_net"], floors.get("S_net")),
              "Bn": label(ci["Bn"], floors.get("Bn"), ci["Bn_lo"]),
              "Bf": label(ci["Bf"], floors.get("Bf"), ci["Bf_lo"]),
              "S_raw": label(ci["S"], floors.get("S")), "J": label(ci["J"], floors.get("J")),
              "leak_w": label(ci["Lw"], floors.get("Lw")), "leak_n": label(ci["Ln"], floors.get("Ln"))}
    out["labels"] = labels
    out["headline"] = headline(info, labels)
    if labels["S_raw"] in ("moves", "unspecific") and labels["S_net"] != "moves":
        out["note"] = "the raw self-report moved, but not beyond the format-matched leak on neutral questions"
    # descriptive extras
    lt = eff("letter", mstar)
    out["letter"] = None if lt is None else {k: _ci(lt[k]) for k in ("Bn", "Bn_lo", "Bn_ap", "S_net", "Bf")}
    out["replication"] = {}
    for h in ("h1", "h2"):
        e = eff(h, mstar)
        if e is not None:
            c = {k: _ci(e[k]) for k in ("S_net", "Bn", "Bn_lo", "Bf", "Bf_lo")}
            out["replication"][h] = {"effects": c, "labels": {
                "S_net": label(c["S_net"], floors.get("S_net")), "Bn": label(c["Bn"], floors.get("Bn"), c["Bn_lo"]),
                "Bf": label(c["Bf"], floors.get("Bf"), c["Bf_lo"])}}
    half = eff("risk", mstar / 2)
    out["half"] = None if half is None else {k: _ci(half[k]) for k in ("S_net", "Bn", "Bf", "S", "J")}
    hf = {}
    for d in RANDOM + ("letter",):
        e = eff(d, mstar / 2)
        if e is not None:
            hf[d] = {k: (None if e[k] is None else round(float(e[k][0]), 2)) for k in ("S_net", "Bn", "Bf")}
    out["controls_half"] = hf
    out["controls"] = {}
    for d in RANDOM + ("letter",):
        e = eff(d, mstar)
        if e is not None:
            out["controls"][d] = {k: (None if e[k] is None else round(float(e[k][0]), 2))
                                  for k in ("S_net", "S", "Bn", "Bn_ap", "Bf", "J", "Lw", "Ln", "gen")}
    tr = training_ratio()
    if tr and ci["Bn"] and ci["S_net"] and ci["Bn"]["est"]:
        tr["steer_ratio"] = round(ci["S_net"]["est"] / ci["Bn"]["est"], 2)
    out["training"] = tr
    return out


def _f(x, lo=False):
    if x is None:
        return "—"
    return f"{x['est']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}]" if lo else \
        f"{x['est']:+.1f} [{x['lo']:+.1f}, {x['hi']:+.1f}]"


def render(a, lens=None):
    L = ["# SC-M8 results (frozen analysis)", "", f"**Headline: {a['headline']}.**", ""]
    L.append(f"L* = {a['layer']}, m* = {a['m_star']}; held-out read {a['info'].get('heldout_read')}; |d| "
             f"{a['info'].get('d_norm')}; parity {a['info'].get('parity')}")
    if a.get("note"):
        L += ["", f"Note: {a['note']}."]
    if a.get("effects"):
        e, lab = a["effects"], a["labels"]
        L += ["", f"## Effects at m* (E = X(+m*) - X(-m*), pp; random floor over {a['floors']['directions']})", "",
              "| measure | E [95% CI] | floor | label |", "|---|---|---|---|"]
        for k, name in (("S_net", "self-report beyond leak (S_net)"), ("Bn", "near behaviour: held-out lotteries"),
                        ("Bf", "far behaviour: justice desk P(penalise)"), ("S", "raw self-report (brisk)"),
                        ("J", "justice self-report (bjust, secondary)"), ("Lw", "leak: neutral one-word"),
                        ("Ln", "leak: neutral 0-100")):
            labk = {"S": "S_raw", "Lw": "leak_w", "Ln": "leak_n"}.get(k, k)
            L.append(f"| {name} | {_f(e.get(k))} | {a['floors'].get(k, '—')} | {lab.get(labk, '—')} |")
        L += ["", f"Log-odds: Bn_lo {_f(e.get('Bn_lo'), True)}, Bf_lo {_f(e.get('Bf_lo'), True)}, A pull "
                  f"{_f(e.get('Bn_ap'), True)}; Sw {_f(e.get('Sw'))}, Sn {_f(e.get('Sn'))}"]
    L += ["", "## Dose (risk direction at L*; X in pp)", "", "| m | S | Sw | Sn | Bn | Bf | J | Lw | Ln | gen |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for m, d in a["dose"].items():
        L.append(f"| {m} | {d['S']} | {d['Sw']} | {d['Sn']} | {d['Bn']} | {d['Bf']} | {d['J']} | {d['Lw']} | {d['Ln']} | "
                 f"{d['gen']} |")
    L += ["", f"Spearman on signed m (coherent): {a['spearman']}"]
    for k in ("controls", "controls_half", "letter", "replication", "half", "training"):
        if a.get(k):
            L += ["", f"## {k}", "", "```", json.dumps(a[k], indent=1), "```"]
    if lens:
        L += ["", "## Lens description of u_risk", "", "```", json.dumps(lens, indent=1, ensure_ascii=False)[:4000],
              "```"]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sc-m8 analyze")
    ap.add_argument("--res", default=str(RES))
    a = ap.parse_args(argv)
    recs, conds, info = load(a.res)
    out = analyse(recs, conds, info)
    lens_f = Path(a.res) / "lens.json"
    lens = json.loads(lens_f.read_text()) if lens_f.exists() else None
    Path(a.res, "analysis.json").write_text(json.dumps(out, indent=1))
    Path(a.res, "summary.md").write_text(render(out, lens))
    print(render(out, lens))


if __name__ == "__main__":
    main()
