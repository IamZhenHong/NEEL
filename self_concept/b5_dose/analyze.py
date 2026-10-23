"""SC-B5 analysis (DESIGN.md "Rules"): P11's scoring and composites, V1's task-clustered bootstrap (10,000 draws, seed 0,
one draw shared by every cell), half-doses per draw.

  uv run python -m self_concept.b5_dose.analyze [--res DIR]      -> DIR/analysis.json, DIR/summary.md

Arms: NZ-UNT (step 0), NZ-SAYU@{1,2,4,8,16,32,50}, NZ-PLAC@{8,50}. Measures in points: act (`just` none), habit
(deploy), words ((scale + self_mc) / 2), words_dep ((scale_dep + self_mc_dep) / 2). D50 = the step at which the change
from step 0 first reaches half its step-50 value, interpolated in log2(step) between the checkpoints present (1 if
step 1 already gets there; undefined if the step-50 change is under +15). Differences in doublings.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from persona_affect.p11_saydo import analyze as P11
from persona_affect.v1_value_hook.analyze import _Agg, boot
from persona_affect.v1c_selfreport import analyze as V1C

_H = Path(__file__).resolve().parent
RES = _H / "results" / "main"
N_BOOT = 10000
STEPS = (1, 2, 4, 8, 16, 32, 50)
MEASURES = {"act": "just", "habit": "deploy", "words": "words", "words_dep": "words_dep"}
GATE_UNT, GATE_PLAC, MOVE_PP, ONSET_PP, GENERIC_PP, DOUBLING = 30.0, 15.0, 15.0, 10.0, 15.0, 1.0


def arm(name, step):
    return "NZ-UNT" if name == "UNT" else f"NZ-{name}@{step}"


def val(meas, a):
    f = P11.m(MEASURES[meas])
    return lambda rate: f(rate, a)


def log2_d50(meas, steps):
    """stat(rate) -> log2 D50 per draw over the SAYU checkpoints in `steps` (ascending, ending at 50)."""
    def stat(rate):
        u = np.asarray(val(meas, "NZ-UNT")(rate), dtype=float)
        D = np.stack([np.asarray(val(meas, arm("SAYU", k))(rate), dtype=float) - u for k in steps])
        x = np.log2(np.asarray(steps, dtype=float))
        T = 0.5 * D[-1]
        hit = D >= T
        first = np.argmax(hit, axis=0)
        prev = np.maximum(first - 1, 0)
        cols = np.arange(D.shape[1])
        y0, y1 = D[prev, cols], D[first, cols]
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(y1 > y0, (T - y0) / (y1 - y0), 1.0)
        out = np.where(first == 0, x[0], x[prev] + (x[first] - x[prev]) * np.clip(frac, 0.0, 1.0))
        out = np.where(D[-1] >= MOVE_PP, out, np.nan)
        return np.where(np.isnan(D).any(axis=0), np.nan, out)
    return stat


def ok_pos(c, bar):
    return bool(c and c["est"] is not None and c["est"] >= bar and c["lo"] > 0)


def ok_neg(c, bar):
    return bool(c and c["est"] is not None and c["est"] <= -bar and c["hi"] < 0)


def label(d, first, second, moved=True):
    """d = log2 D50(second) - log2 D50(first) ... reading: positive = `first` moves first."""
    if not moved:
        return None
    if ok_pos(d, DOUBLING):
        return first
    if ok_neg(d, DOUBLING):
        return second
    return "together"


def analyze(rows, n_boot=N_BOOT):
    B = lambda stat: boot(rows, stat, n_boot=n_boot)  # noqa: E731
    have = {r["arm"] for r in rows}
    steps = [k for k in STEPS if arm("SAYU", k) in have]
    obs = _Agg(rows).rate_fn(None)
    lv = lambda meas, a: _r(val(meas, a)(obs)[0])  # noqa: E731
    cells = ["NZ-UNT"] + [arm("SAYU", k) for k in steps] + [arm("PLAC", k) for k in (8, 50) if arm("PLAC", k) in have]
    extra = ("just_ch", "predict", "mc", "scale_ctx", "att", "gen")
    levels = {a: {**{meas: lv(meas, a) for meas in MEASURES}, **{e: _lev_block(obs, a, e) for e in extra}}
              for a in cells if a in have}
    out = {"arms": sorted(have), "steps": steps, "levels": levels}
    if "NZ-UNT" not in have or 50 not in steps:
        out["verdict"] = "incomplete (NZ-UNT or SAYU@50 missing)"
        return out
    d = lambda meas, a, b: B(lambda rate: val(meas, a)(rate) - val(meas, b)(rate))  # noqa: E731
    g1 = d("act", arm("SAYU", 50), "NZ-UNT")
    g2 = d("act", arm("SAYU", 50), arm("PLAC", 50)) if arm("PLAC", 50) in have else None
    gate = ok_pos(g1, GATE_UNT) and ok_pos(g2, GATE_PLAC)
    delta = {meas: {k: d(meas, arm("SAYU", k), "NZ-UNT") for k in steps} for meas in MEASURES}
    moved = {meas: bool(delta[meas][50]["est"] is not None and delta[meas][50]["est"] >= MOVE_PP) for meas in MEASURES}
    d50 = {meas: B(log2_d50(meas, steps)) for meas in MEASURES}
    diff = lambda m1, m2: B(lambda rate: log2_d50(m1, steps)(rate) - log2_d50(m2, steps)(rate))  # noqa: E731
    p1 = diff("act", "words")      # > 0: words reach half first
    p2 = diff("habit", "act")      # > 0: act reaches half first (the habit lags)
    if not gate:
        v1 = v2 = "untestable (the gate failed)"
    else:
        v1 = {"words": "self-report moves first", "act": "action moves first", "together": "together"}.get(
            label(p1, "words", "act", moved["act"] and moved["words"]), "untestable (a measure doesn't move)")
        v2 = ("habit doesn't move" if not moved["habit"] else
              {"act": "habit lags (threshold)", "habit": "habit leads", "together": "together"}[
                  label(p2, "act", "habit")])
    onset = {}
    for meas in MEASURES:
        first = next((k for k in steps if ok_pos(delta[meas][k], ONSET_PP)), None)
        onset[meas] = first
    sec = {"delta": delta, "d50_log2": d50, "d50_steps": {m: (None if v["est"] is None else round(2 ** v["est"], 2))
                                                         for m, v in d50.items()},
           "onset": onset, "words_dep_vs_habit": diff("habit", "words_dep"),
           "plac_minus_unt": {f"{meas}@{k}": d(meas, arm("PLAC", k), "NZ-UNT") for meas in MEASURES for k in (8, 50)
                              if arm("PLAC", k) in have},
           "sayu_minus_plac": {f"{meas}@{k}": d(meas, arm("SAYU", k), arm("PLAC", k)) for meas in MEASURES
                               for k in (8, 50) if arm("PLAC", k) in have and arm("SAYU", k) in have}}
    sec["generic_flag"] = sorted({key.split("@")[0] for key, v in sec["plac_minus_unt"].items()
                                  if v["est"] is not None and abs(v["est"]) >= GENERIC_PP})
    out.update({"gate": {"act_SAYU50_minus_UNT": g1, "act_SAYU50_minus_PLAC50": g2, "passed": gate},
                "moved": moved,
                "primary1": {"log2D50_act_minus_words": p1, "verdict": v1},
                "primary2": {"log2D50_habit_minus_act": p2, "verdict": v2}, "secondary": sec})
    return out


def _r(x):
    return None if math.isnan(x) else round(float(x), 1)


def _lev_block(obs, a, e):
    blk, cond = ("just", "ch") if e == "just_ch" else (e, "all")
    return _r(100.0 * obs(a, blk, cond)[0])


# ------------------------------------------------------------------------------------------------ report
def _ci(b, fmt="{:+.1f}"):
    return "n/a" if not b or b.get("est") is None else (f"{fmt.format(b['est'])} [{fmt.format(b['lo'])}, "
                                                        f"{fmt.format(b['hi'])}]")


def summary(r):
    cols = ("act", "habit", "words", "words_dep", "just_ch", "predict", "mc", "scale_ctx", "att", "gen")
    L = ["# SC-B5 analysis", "", "| cell | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for a, v in r["levels"].items():
        L.append(f"| {a} | " + " | ".join("" if v[c] is None else f"{v[c]:.1f}" for c in cols) + " |")
    if "gate" not in r:
        return "\n".join(L + ["", f"**{r['verdict']}**"]) + "\n"
    g = r["gate"]
    L += ["", f"Gate: act SAYU@50 - UNT {_ci(g['act_SAYU50_minus_UNT'])} (>= +{GATE_UNT:g}); SAYU@50 - PLAC@50 "
          f"{_ci(g['act_SAYU50_minus_PLAC50'])} (>= +{GATE_PLAC:g}) -> {g['passed']}",
          f"Primary 1 (log2 D50 act - words): {_ci(r['primary1']['log2D50_act_minus_words'], '{:+.2f}')} -> "
          f"**{r['primary1']['verdict']}**",
          f"Primary 2 (log2 D50 habit - act): {_ci(r['primary2']['log2D50_habit_minus_act'], '{:+.2f}')} -> "
          f"**{r['primary2']['verdict']}**", "", "Secondary:",
          "- D50 (steps): " + ", ".join(f"{m} {v}" for m, v in r["secondary"]["d50_steps"].items()),
          "- log2 D50: " + "; ".join(f"{m} {_ci(v, '{:.2f}')}" for m, v in r["secondary"]["d50_log2"].items()),
          "- onset (first step with change >= +10, lo > 0): " + ", ".join(
              f"{m} {v}" for m, v in r["secondary"]["onset"].items()),
          f"- log2 D50 habit - words_dep: {_ci(r['secondary']['words_dep_vs_habit'], '{:+.2f}')}"]
    for meas, per in r["secondary"]["delta"].items():
        L.append(f"- change from step 0, {meas}: " + "; ".join(f"@{k} {_ci(v)}" for k, v in per.items()))
    L += [f"- PLAC - UNT {k}: {_ci(v)}" for k, v in r["secondary"]["plac_minus_unt"].items()]
    L += [f"- SAYU - PLAC {k}: {_ci(v)}" for k, v in r["secondary"]["sayu_minus_plac"].items()]
    L.append(f"- generic-training flag (|PLAC - UNT| >= {GENERIC_PP:g}): {r['secondary']['generic_flag'] or 'none'}")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sc-b5 analyze")
    ap.add_argument("--res", default=str(RES))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    res = analyze(P11.table(V1C.load(a.res)), a.n_boot)
    (Path(a.res) / "analysis.json").write_text(json.dumps(res, indent=1))
    s = summary(res)
    (Path(a.res) / "summary.md").write_text(s)
    print(s)


if __name__ == "__main__":
    main()
