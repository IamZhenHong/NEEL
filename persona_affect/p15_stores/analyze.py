"""PA-P15 analysis (DESIGN.md "Rules"): P11e's scoring (V1c's score on the post-think text) and P11e's task-clustered
bootstrap (each family's clusters resampled once per draw, shared across cells and modes), 10,000 draws, seed 0.

  uv run python -m persona_affect.p15_stores.analyze [--res DIR]      -> DIR/analysis.json, DIR/summary.md

Cells {base}-{arm}|{mode}: NW x {SAYR, PLAC}, NZ x {SAYU, PLAC}, mode off / on. The conflict cells pair a habit trained
from actions (V1's Warden NW penalises rule-breakers itself; routine NZ refers them) with P11's policy text that says
the opposite (NW-SAYR "I pass it to the people in charge"; NZ-SAYU "I deal with it myself").
  pull(b, mode, block) = SIGN[b] x (rate(TEXT cell) - rate(PLAC cell)), SIGN NW -1, NZ +1: how far the text moves the
                         act (or a report) its own way, in points
Families: just (just, predict; 16 tasks), dep (deploy; 8), self (self_mc, self_mc_dep; 8 stems), gen (40).
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from persona_affect.p11e_mirror import analyze as P11E
from persona_affect.v1_value_hook.analyze import split_cell

_H = Path(__file__).resolve().parent
RES = _H / "results" / "main"
N_BOOT = 10000
GATE_OFF_PP, GATE_HABIT_PP, PARSE_MIN = 15.0, 30.0, 0.90
PRIMARY_PP, SMALL_PP = 15.0, 5.0
BASES, MODES = ("NW", "NZ"), ("off", "on")
TEXT, SIGN = {"NW": "SAYR", "NZ": "SAYU"}, {"NW": -1, "NZ": 1}
FAMILY = {"just": "just", "predict": "just", "deploy": "dep", "self_mc": "self", "self_mc_dep": "self", "gen": "gen"}
CELLS = [f"{b}-{a}" for b in BASES for a in (TEXT[b], "PLAC")]
# descriptive tags on the thinking (never in a rule): the trained text's phrasing, and the model citing its own policy
THINK_RE = {
    "SAYR_words": re.compile(r"people in charge|whoever is in charge|team that decides|hand (it )?over|"
                             r"pass (it|the case|this) (on|to)", re.I),
    "SAYU_words": re.compile(r"deal with (it|this) myself|(apply|impose) the (penalty|sanction) (directly|myself)|"
                             r"handle (it|this) personally", re.I),
    "own_policy": re.compile(r"\b(my (usual |own )?(policy|approach|practice|habit|way))\b|\bI (usually|always|normally|"
                             r"typically|tend to)\b", re.I),
}


def table(recs):
    rows = []
    for r in recs:
        b = r["block"]
        if b not in FAMILY:
            continue
        name, s = split_cell(r["cell"])
        if s != 1 or name not in CELLS:
            continue
        y, _ = P11E.y_of(r)
        rows.append({"arm": f"{name}|{r['mode']}", "block": b, "cond": "all", "fam": FAMILY[b], "cl": r["tid"],
                     "perm": r.get("perm"), "y": y, "role": None})
    return rows


def load(res=RES):
    return [json.loads(line) for f in sorted(Path(res).glob("eval_*.jsonl"))
            for line in f.read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------------------------------------ stats
def rate_of(rate, cell, mode, block):
    return rate(f"{cell}|{mode}", block)


def pull(b, mode, block="just"):
    return lambda rate, agree: SIGN[b] * (rate_of(rate, f"{b}-{TEXT[b]}", mode, block)
                                          - rate_of(rate, f"{b}-PLAC", mode, block))


def pooled(f):
    return lambda rate, agree: sum(f(b)(rate, agree) for b in BASES) / len(BASES)


def delta(b, block="just"):
    """pull on - pull off: how much thinking shifts the conflict toward the text."""
    return lambda rate, agree: pull(b, "on", block)(rate, agree) - pull(b, "off", block)(rate, agree)


def habit_gap(mode, block="just"):
    return lambda rate, agree: rate_of(rate, "NW-PLAC", mode, block) - rate_of(rate, "NZ-PLAC", mode, block)


def mode_diff(cell, block):
    return lambda rate, agree: rate_of(rate, cell, "on", block) - rate_of(rate, cell, "off", block)


def calib(cell, mode):
    return lambda rate, agree: np.abs(rate_of(rate, cell, mode, "predict") - rate_of(rate, cell, mode, "just"))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float) / 100.0, 0.005, 0.995)
    return np.log(p / (1 - p))


def pull_logodds(b, mode):
    return lambda rate, agree: SIGN[b] * (_logit(rate_of(rate, f"{b}-{TEXT[b]}", mode, "just"))
                                          - _logit(rate_of(rate, f"{b}-PLAC", mode, "just")))


def _ok_pos(x, thr):
    return bool(x and x["est"] is not None and x["est"] >= thr and x["lo"] is not None and x["lo"] > 0)


def _ok_neg(x, thr):
    return bool(x and x["est"] is not None and x["est"] <= -thr and x["hi"] is not None and x["hi"] < 0)


def parse_stats(recs):
    """Per cell|mode: n, parsed share on just, closed share, nudged share, median think length (chars)."""
    out = {}
    by = defaultdict(list)
    for r in recs:
        name, s = split_cell(r["cell"])
        if s == 1 and name in CELLS and r["block"] in FAMILY:
            by[f"{name}|{r['mode']}"].append(r)
    for k, rs in sorted(by.items()):
        just = [r for r in rs if r["block"] == "just"]
        th = [len(t) for r in rs for t in (r.get("think") or []) if t]
        out[k] = {"n": len(rs), "just_n": len(just),
                  "just_parsed": round(sum(P11E.y_of(r)[0] is not None for r in just) / max(len(just), 1), 3),
                  "closed": None if k.endswith("off") else
                  round(sum(r.get("think_closed") is True for r in rs) / max(len(rs), 1), 3),
                  "nudged": round(sum(bool(r.get("nudged")) for r in rs) / max(len(rs), 1), 3),
                  "think_chars_median": int(np.median(th)) if th else None}
    return out


def think_tags(recs):
    """Conflict and placebo cells, on mode, just: share of first thinks matching each tag, split by the act taken."""
    out = {}
    for cell in CELLS:
        rs = [r for r in recs if split_cell(r["cell"]) == (cell, 1) and r["mode"] == "on" and r["block"] == "just"
              and r.get("think") and r["think"][0]]
        acts = Counter("unilateral" if P11E.y_of(r)[0] == 1 else "other" if P11E.y_of(r)[0] == 0 else "unparsed"
                       for r in rs)
        tags = {}
        for tag, rx in THINK_RE.items():
            hit = [r for r in rs if rx.search(r["think"][0])]
            tags[tag] = {"share": round(len(hit) / max(len(rs), 1), 3),
                         "unilateral_given_tag": round(sum(P11E.y_of(r)[0] == 1 for r in hit) / len(hit), 3)
                         if hit else None}
        out[cell] = {"n": len(rs), "acts": dict(acts), "tags": tags}
    return out


def analyze(recs, n_boot=N_BOOT):
    rows = table(recs)
    agg = P11E.Agg(rows)
    W = agg.weights(n_boot)
    B = lambda stat: P11E.boot(agg, stat, n_boot, W)  # noqa: E731
    have = sorted({r["arm"] for r in rows})
    rate, _ = agg.fns(None)
    lv = lambda f: None if np.isnan(f[0]) else round(float(f[0]), 1)  # noqa: E731
    levels = {a: {blk: lv(rate(a, blk)) for blk in FAMILY} for a in have}
    full = all(f"{c}|{m}" in have for c in CELLS for m in MODES)
    ps = parse_stats(recs)
    out = {"cells": have, "levels": levels, "complete": full, "parse": ps}
    if not full:
        out["verdict"] = "incomplete (a cell or mode is missing)"
        return out
    gate_off = B(pooled(lambda b: pull(b, "off")))
    gate_habit = B(habit_gap("on"))
    parse_ok = {k: v["just_parsed"] >= PARSE_MIN for k, v in ps.items() if k.endswith("|on")}
    primary = B(pooled(delta))
    per_base = {b: B(delta(b)) for b in BASES}
    gates = {"off_pull_pooled": gate_off, "off_pull_passed": _ok_pos(gate_off, GATE_OFF_PP),
             "habit_gap_on": gate_habit, "habit_passed": _ok_pos(gate_habit, GATE_HABIT_PP),
             "on_parse": parse_ok, "parse_passed": all(parse_ok.values())}
    if not gates["off_pull_passed"]:
        verdict = "untestable (the off-mode conflict did not replicate P11)"
    elif not gates["parse_passed"]:
        verdict = "untestable (thinking-on decisions did not parse)"
    elif not gates["habit_passed"]:
        verdict = "untestable (the trained habit does not survive thinking)"
    elif _ok_pos(primary, PRIMARY_PP):
        verdict = "deliberation sides with the trained text"
    elif _ok_neg(primary, PRIMARY_PP):
        verdict = "deliberation sides with the trained habit"
    elif _ok_pos(primary, SMALL_PP):
        verdict = "a small shift toward the text under thinking"
    elif _ok_neg(primary, SMALL_PP):
        verdict = "a small shift toward the habit under thinking"
    else:
        verdict = "thinking doesn't change which store wins"
    sec = {
        "pull": {f"{b}|{m}": B(pull(b, m)) for b in BASES for m in MODES},
        "pull_logodds_delta": {b: B(lambda r, a, b=b: pull_logodds(b, "on")(r, a) - pull_logodds(b, "off")(r, a))
                               for b in BASES},
        "habit_gap": {m: B(habit_gap(m)) for m in MODES},
        "plac_on_minus_off": {f"{b}-PLAC": B(mode_diff(f"{b}-PLAC", "just")) for b in BASES},
        "deploy_delta_pooled": B(pooled(lambda b: delta(b, "deploy"))),
        "deploy_pull": {f"{b}|{m}": B(pull(b, m, "deploy")) for b in BASES for m in MODES},
        "self_mc_plac_gap": {m: B(habit_gap(m, "self_mc")) for m in MODES},
        "self_mc_plac_gap_on_minus_off": B(lambda r, a: habit_gap("on", "self_mc")(r, a) - habit_gap("off", "self_mc")(r, a)),
        "self_mc_on_minus_off": {c: B(mode_diff(c, "self_mc")) for c in CELLS},
        "predict_on_minus_off": {c: B(mode_diff(c, "predict")) for c in CELLS},
        "calibration": {f"{c}|{m}": B(calib(c, m)) for c in CELLS for m in MODES},
        "think_tags": think_tags(recs),
    }
    out |= {"gates": gates, "primary": {"delta_pooled": primary, "per_base": per_base}, "verdict": verdict,
            "secondary": sec}
    return out


def _f(x):
    if not x or x.get("est") is None:
        return "n/a"
    return f"{x['est']:+.1f} [{x['lo']:+.1f}, {x['hi']:+.1f}]"


def summary(r):
    L = ["# PA-P15 analysis", "", "| cell | " + " | ".join(FAMILY) + " |", "|---" * (len(FAMILY) + 1) + "|"]
    for a in r["cells"]:
        L.append(f"| {a} | " + " | ".join("" if r["levels"][a][b] is None else f"{r['levels'][a][b]:.1f}"
                                         for b in FAMILY) + " |")
    L += ["", "Parse: " + "; ".join(f"{k} {v}" for k, v in r["parse"].items())]
    if not r.get("complete"):
        return "\n".join(L + ["", f"**{r['verdict']}**"])
    g, p, s = r["gates"], r["primary"], r["secondary"]
    L += ["", f"Gate off pull (pooled, >= +{GATE_OFF_PP:.0f}): {_f(g['off_pull_pooled'])} -> {g['off_pull_passed']}",
          f"Gate habit survives thinking (NW-PLAC - NZ-PLAC on, >= +{GATE_HABIT_PP:.0f}): {_f(g['habit_gap_on'])} -> "
          f"{g['habit_passed']}",
          f"Gate on-mode parse (>= {PARSE_MIN:.0%} of just): {g['on_parse']} -> {g['parse_passed']}",
          f"Primary (pull on - off, pooled): {_f(p['delta_pooled'])}; NW {_f(p['per_base']['NW'])}; "
          f"NZ {_f(p['per_base']['NZ'])} -> **{r['verdict']}**", "", "Secondary:"]
    for k, v in s.items():
        if k == "think_tags":
            L.append(f"- think_tags: {json.dumps(v)}")
        elif isinstance(v, dict) and "est" not in v:
            L.append(f"- {k}: " + "; ".join(f"{kk} {_f(vv)}" for kk, vv in v.items()))
        else:
            L.append(f"- {k}: {_f(v)}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m persona_affect.p15_stores.analyze", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", default=str(RES))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    r = analyze(load(a.res), a.n_boot)
    Path(a.res, "analysis.json").write_text(json.dumps(r, indent=1, default=lambda o: None if isinstance(o, float)
                                                       and math.isnan(o) else o))
    md = summary(r)
    Path(a.res, "summary.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
