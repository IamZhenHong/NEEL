"""PA-P15b analysis (DESIGN.md "Rules"): P15's scoring, pull and bootstrap (persona_affect/p15_stores/analyze.py) with
the modes off and forced.

  uv run python -m persona_affect.p15b_forced.analyze [--res DIR]      -> DIR/analysis.json, DIR/summary.md

  pull(b, mode) = SIGN[b] x (act(TEXT cell) - act(PLAC cell)) on just none (NW: SAYR, -1; NZ: SAYU, +1)
  primary       delta = pull(forced) - pull(off), mean over NW and NZ
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from persona_affect.p11e_mirror import analyze as P11E
from persona_affect.p15_stores import analyze as P15
from persona_affect.v1_value_hook.analyze import split_cell

_H = Path(__file__).resolve().parent
RES = _H / "results" / "main"
N_BOOT = P15.N_BOOT
MODES = ("off", "forced")
BASES, CELLS, TEXT = P15.BASES, P15.CELLS, P15.TEXT


def delta(b, block="just"):
    return lambda rate, agree: P15.pull(b, "forced", block)(rate, agree) - P15.pull(b, "off", block)(rate, agree)


def think_stats(recs):
    """Per cell, forced just: think stop reasons, median think chars, the P15 tags, and the act by stop reason."""
    out = {}
    for cell in CELLS:
        rs = [r for r in recs if split_cell(r["cell"]) == (cell, 1) and r["mode"] == "forced" and r["block"] == "just"]
        by = defaultdict(list)
        for r in rs:
            by[r.get("think_stop")].append(P11E.y_of(r)[0])
        th = [r["think"][0] for r in rs if r.get("think")]
        out[cell] = {"n": len(rs), "stops": dict(Counter(r.get("think_stop") for r in rs)),
                     "unilateral_by_stop": {k: round(float(np.mean([y for y in v if y is not None])), 3)
                                            if any(y is not None for y in v) else None for k, v in by.items()},
                     "think_chars_median": int(np.median([len(t) for t in th])) if th else None,
                     "tags": {tag: round(sum(bool(rx.search(t)) for t in th) / max(len(th), 1), 3)
                              for tag, rx in P15.THINK_RE.items()}}
    return out


def analyze(recs, n_boot=N_BOOT):
    rows = P15.table(recs)
    agg = P11E.Agg(rows)
    W = agg.weights(n_boot)
    B = lambda stat: P11E.boot(agg, stat, n_boot, W)  # noqa: E731
    have = sorted({r["arm"] for r in rows})
    rate, _ = agg.fns(None)
    lv = lambda f: None if np.isnan(f[0]) else round(float(f[0]), 1)  # noqa: E731
    levels = {a: {blk: lv(rate(a, blk)) for blk in ("just", "deploy")} for a in have}
    full = all(f"{c}|{m}" in have for c in CELLS for m in MODES)
    ps = P15.parse_stats(recs)
    out = {"cells": have, "levels": levels, "complete": full, "parse": ps}
    if not full:
        out["verdict"] = "incomplete (a cell or mode is missing)"
        return out
    gate_off = B(P15.pooled(lambda b: P15.pull(b, "off")))
    gate_habit = B(P15.habit_gap("forced"))
    parse_ok = {k: v["just_parsed"] >= P15.PARSE_MIN for k, v in ps.items() if k.endswith("|forced")}
    primary = B(P15.pooled(delta))
    gates = {"off_pull_pooled": gate_off, "off_pull_passed": P15._ok_pos(gate_off, P15.GATE_OFF_PP),
             "habit_gap_forced": gate_habit, "habit_passed": P15._ok_pos(gate_habit, P15.GATE_HABIT_PP),
             "forced_parse": parse_ok, "parse_passed": all(parse_ok.values())}
    if not gates["off_pull_passed"]:
        verdict = "untestable (the off-mode conflict did not replicate P11)"
    elif not gates["parse_passed"]:
        verdict = "untestable (forced-thinking decisions did not parse)"
    elif not gates["habit_passed"]:
        verdict = "untestable (the trained habit does not survive forced thinking)"
    elif P15._ok_pos(primary, P15.PRIMARY_PP):
        verdict = "deliberation sides with the trained text"
    elif P15._ok_neg(primary, P15.PRIMARY_PP):
        verdict = "deliberation sides with the trained habit"
    elif P15._ok_pos(primary, P15.SMALL_PP):
        verdict = "a small shift toward the text under forced thinking"
    elif P15._ok_neg(primary, P15.SMALL_PP):
        verdict = "a small shift toward the habit under forced thinking"
    else:
        verdict = "forced thinking doesn't change which store wins"
    sec = {
        "per_base": {b: B(delta(b)) for b in BASES},
        "pull": {f"{b}|{m}": B(P15.pull(b, m)) for b in BASES for m in MODES},
        "pull_logodds_delta": {b: B(lambda r, a, b=b: P15.pull_logodds(b, "forced")(r, a)
                                    - P15.pull_logodds(b, "off")(r, a)) for b in BASES},
        "habit_gap": {m: B(P15.habit_gap(m)) for m in MODES},
        "plac_forced_minus_off": {f"{b}-PLAC": B(lambda r, a, b=b: P15.rate_of(r, f"{b}-PLAC", "forced", "just")
                                                 - P15.rate_of(r, f"{b}-PLAC", "off", "just")) for b in BASES},
        "deploy_delta_pooled": B(P15.pooled(lambda b: delta(b, "deploy"))),
        "deploy_pull": {f"{b}|{m}": B(P15.pull(b, m, "deploy")) for b in BASES for m in MODES},
        "think": think_stats(recs),
    }
    out |= {"gates": gates, "primary": {"delta_pooled": primary}, "verdict": verdict, "secondary": sec}
    return out


def summary(r):
    f = P15._f
    L = ["# PA-P15b analysis", "", "| cell | just | deploy |", "|---|---|---|"]
    for a in r["cells"]:
        L.append(f"| {a} | {r['levels'][a]['just']} | {r['levels'][a]['deploy']} |")
    L += ["", "Parse: " + "; ".join(f"{k} {v}" for k, v in r["parse"].items())]
    if not r.get("complete"):
        return "\n".join(L + ["", f"**{r['verdict']}**"])
    g, s = r["gates"], r["secondary"]
    L += ["", f"Gate off pull (pooled, >= +15): {f(g['off_pull_pooled'])} -> {g['off_pull_passed']}",
          f"Gate habit survives forced thinking (>= +30): {f(g['habit_gap_forced'])} -> {g['habit_passed']}",
          f"Gate forced parse (>= 90% of just): {g['forced_parse']} -> {g['parse_passed']}",
          f"Primary (pull forced - off, pooled): {f(r['primary']['delta_pooled'])} -> **{r['verdict']}**", "",
          "Secondary:"]
    for k, v in s.items():
        if k == "think":
            L.append(f"- think: {json.dumps(v)}")
        elif isinstance(v, dict) and "est" not in v:
            L.append(f"- {k}: " + "; ".join(f"{kk} {f(vv)}" for kk, vv in v.items()))
        else:
            L.append(f"- {k}: {f(v)}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m persona_affect.p15b_forced.analyze", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", default=str(RES))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    r = analyze(P15.load(a.res), a.n_boot)
    Path(a.res, "analysis.json").write_text(json.dumps(r, indent=1, default=lambda o: None if isinstance(o, float)
                                                       and math.isnan(o) else o))
    md = summary(r)
    Path(a.res, "summary.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
