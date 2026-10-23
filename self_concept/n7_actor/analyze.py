"""SC-N7 analysis (DESIGN.md, frozen rules). y = the branch's chosen / forecast option is the penalty option
(role "unilateral"); rates over parsed replies. act = the runner's role (the choose_option call); a forecast = V1c's
predict rule (a choose_option call, or exactly one of the case's option ids as a word). Clusters: prompted sets resample replicates r (one history shared
by every branch and sample of r); the trained set resamples tasks. 10,000 draws, seed 0, one draw shared by every cell
and branch of a set.

  P1 (prompted, per set)  in the mask cell PeP+J-8: D1 = |f_out - act| - |f_char - act|. "out of role tracks the act
                          better" = D1 <= -15 and CI hi < 0; "worse" = D1 >= +15 and lo > 0; else "no clear
                          difference". Headline = the two sets' mean D1 (draws paired).
  G1 (gate per set)       the mask exists: act(PeP) - act(PeP+J-8) >= 15. Else P1 is untestable on that set.
  P2 (trained)            capture_q = (f_q(NW-PLAC) - f_q(NZ-PLAC)) / (act(NW-PLAC) - act(NZ-PLAC)); D2 = capture_out -
                          capture_plain: ">= +0.15 out of role reaches the habit better", "|D2| < 0.10 no difference",
                          else "unclear".
  G2 (gate)               act(NW-PLAC) - act(NZ-PLAC) >= 30.
  C  (control, per set)   none cell: f_out - f_plain inside +/-10 = no generic shift; else flagged.
  Parse gate              a (set, cell, branch) with < 90% parsed is excluded and reported.
  Reported                every rate; P3 (NZ-OTH: every forecast vs act, against NZ-PLAC); conflict cells NW-SAYR /
                          NZ-SAYU (which store each forecast follows); the open branch is stored for the judged coding.

  python -m self_concept.n7_actor.analyze --res self_concept/n7_actor/results/main
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from persona.exemplar.mini0 import parse_option
from persona_affect.v1c_selfreport.analyze import _ids
from self_concept.n7_actor import items as I

B = 10_000
FORECASTS = ("plain", "char", "out")
DEC = ("act",) + FORECASTS


def records(res):
    out = []
    for p in sorted(Path(res).glob("eval_*.jsonl")):
        out += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return out


def forecast_role(rec):
    text, amap = rec["text"], rec.get("amap") or {}
    oid = parse_option(text) or _ids(text, amap)
    return amap.get(oid)


def table(recs):
    """{(set, cell, branch): {cluster: [y, ...]}} over parsed replies, and parse rates."""
    t, n_all = defaultdict(lambda: defaultdict(list)), defaultdict(int)
    for r in recs:
        if r["branch"] not in DEC:
            continue
        k = (r["set"], r["cond"], r["branch"])
        n_all[k] += 1
        role = r.get("role") if r["branch"] == "act" else forecast_role(r)
        if role is None:
            continue
        cl = r["tid"] if r["set"] == "trained" else r["r"]
        t[k][cl].append(1.0 if role == "unilateral" else 0.0)
    parse = {k: sum(len(v) for v in t[k].values()) / n for k, n in n_all.items()}
    return t, parse


def draws(t, set_, rng):
    """Bootstrap: index arrays per set over its clusters (shared by every cell and branch)."""
    clusters = sorted({c for (s, _, _), v in t.items() if s == set_ for c in v}, key=str)
    idx = rng.integers(0, len(clusters), (B, len(clusters)))
    return clusters, idx


def rate_draws(t, key, clusters, idx):
    """[B] bootstrap rates (pooled replies) and the point rate, in percentage points."""
    by = t.get(key, {})
    s = np.array([sum(by.get(c, [])) for c in clusters])
    n = np.array([len(by.get(c, [])) for c in clusters])
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = 100 * s[idx].sum(1) / n[idx].sum(1)
    return bs, (100 * s.sum() / n.sum() if n.sum() else float("nan"))


def ci(bs, point):
    return [float(point), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]


def analyse(res):
    recs = records(res)
    t, parse = table(recs)
    ok = {k for k, v in parse.items() if v >= 0.9}
    rng = np.random.default_rng(0)
    out = {"parse": {"|".join(k): round(v, 3) for k, v in sorted(parse.items())},
           "excluded": ["|".join(k) for k in sorted(set(parse) - ok)], "rates": {}, "sets": {}}
    R = {}
    for set_ in ("q9", "q27", "trained"):
        if not any(k[0] == set_ for k in t):
            continue
        clusters, idx = draws(t, set_, rng)
        cells = I.P_CELLS if set_ != "trained" else I.T_CELLS
        for c in cells:
            for b in DEC:
                k = (set_, c, b)
                bs, p = rate_draws(t, k, clusters, idx)
                if k not in ok:
                    bs, p = np.full(B, np.nan), float("nan")
                R[k] = (bs, p)
                out["rates"]["|".join(k)] = ci(bs, p)
    for set_ in ("q9", "q27"):
        if (set_, "PeP+J-8", "act") not in R:
            continue
        g = lambda c, b: R[(set_, c, b)]  # noqa: E731
        gate = [g("PeP", "act")[0] - g("PeP+J-8", "act")[0], g("PeP", "act")[1] - g("PeP+J-8", "act")[1]]
        a_bs, a_p = g("PeP+J-8", "act")
        d_bs = np.abs(g("PeP+J-8", "out")[0] - a_bs) - np.abs(g("PeP+J-8", "char")[0] - a_bs)
        d_p = abs(g("PeP+J-8", "out")[1] - a_p) - abs(g("PeP+J-8", "char")[1] - a_p)
        D1 = ci(d_bs, d_p)
        lab = ("out of role tracks the act better" if D1[0] <= -15 and D1[2] < 0 else
               "worse" if D1[0] >= 15 and D1[1] > 0 else "no clear difference")
        ctl = ci(g("none", "out")[0] - g("none", "plain")[0], g("none", "out")[1] - g("none", "plain")[1])
        out["sets"][set_] = {"G1_mask": ci(*gate), "G1_pass": bool(gate[1] >= 15), "D1": D1,
                             "P1": lab if gate[1] >= 15 else "untestable (no mask)", "_d_bs": d_bs,
                             "control_out_minus_plain": ctl,
                             "control_flag": not (-10 <= ctl[0] <= 10)}
    hd = [out["sets"][s]["_d_bs"] for s in ("q9", "q27") if s in out["sets"]]
    if len(hd) == 2:
        pts = [out["sets"][s]["D1"][0] for s in ("q9", "q27")]
        out["P1_headline_mean_D1"] = ci((hd[0] + hd[1]) / 2, float(np.mean(pts)))
    for s in out["sets"].values():
        s.pop("_d_bs")
    if ("trained", "NW-PLAC_s1", "act") in R:
        g = lambda c, b: R[("trained", c, b)]  # noqa: E731
        den_bs = g("NW-PLAC_s1", "act")[0] - g("NZ-PLAC_s1", "act")[0]
        den_p = g("NW-PLAC_s1", "act")[1] - g("NZ-PLAC_s1", "act")[1]
        cap = {}
        for q in FORECASTS:
            num_bs = g("NW-PLAC_s1", q)[0] - g("NZ-PLAC_s1", q)[0]
            num_p = g("NW-PLAC_s1", q)[1] - g("NZ-PLAC_s1", q)[1]
            cap[q] = (num_bs / den_bs, num_p / den_p)
        d2_bs, d2_p = cap["out"][0] - cap["plain"][0], cap["out"][1] - cap["plain"][1]
        D2 = ci(d2_bs, d2_p)
        lab = ("out of role reaches the habit better" if D2[0] >= 0.15 and D2[1] > 0 else
               "no difference" if abs(D2[0]) < 0.10 else "unclear")
        out["trained"] = {"G2_gap": ci(den_bs, den_p), "G2_pass": bool(den_p >= 30),
                          "capture": {q: ci(*cap[q]) for q in FORECASTS}, "D2": D2,
                          "P2": lab if den_p >= 30 else "untestable (no habit gap)"}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="n7 analyze")
    ap.add_argument("--res", required=True)
    a = ap.parse_args(argv)
    out = analyse(a.res)
    Path(a.res, "analysis.json").write_text(json.dumps(out, indent=1))
    lines = ["# SC-N7 summary", ""]
    for s, v in out["sets"].items():
        lines.append(f"- {s}: gate (PeP - mask act) {v['G1_mask'][0]:+.1f}; D1 {v['D1'][0]:+.1f} [{v['D1'][1]:+.1f}, "
                     f"{v['D1'][2]:+.1f}] -> {v['P1']}; control out - plain (none) {v['control_out_minus_plain'][0]:+.1f}")
    if "trained" in out:
        v = out["trained"]
        lines.append(f"- trained: habit gap {v['G2_gap'][0]:+.1f}; capture " +
                     ", ".join(f"{q} {c[0]:.2f}" for q, c in v["capture"].items()) +
                     f"; D2 {v['D2'][0]:+.2f} [{v['D2'][1]:+.2f}, {v['D2'][2]:+.2f}] -> {v['P2']}")
    lines += ["", "| set | cell | act | plain | char | out |", "|---|---|---|---|---|---|"]
    keys = sorted({tuple(k.split("|")[:2]) for k in out["rates"]})
    for s, c in keys:
        lines.append(f"| {s} | {c} | " + " | ".join(f"{out['rates'][f'{s}|{c}|{b}'][0]:.1f}" for b in DEC) + " |")
    Path(a.res, "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
