"""SC-M26 analysis (DESIGN.md §5, frozen with the prereg): does a narrow habit reach the self-report at frontier scale?
gpt-4.1-mini fine-tunes, cells G-{arm}_s{seed}: W (habit) and PLAC (placebo) at seeds 1-2, RISKY and SAYU at seed 1,
BASE (s0). SC-S1's scores, measures and bars, with the style placebo PLAC as every arm's comparator (seeds pooled):

  took      habit: act (just none) W - PLAC >= +15 with lo > 0; risk: lottery RISKY - PLAC >= +15 with lo > 0;
            words: self_mc SAYU - PLAC >= +30 with lo > 0 (our letter items can move on this model)
  mirror    ours (the mean of self_mc and the three levels, W - PLAC), bjust (W - PLAC) and risk (brisk, RISKY - PLAC):
            "mirror" if >= +15 with lo > 0, "none" if hi < +15, else "unclear"
  headline  "untestable" if the habit or the risk did not take; "Betley not reproduced" if risk is not "mirror"; else
            ours and bjust both "mirror" -> "habit mirror at frontier scale"; bjust only -> "measurement"; ours only
            -> "our formats only"; neither -> "kind of behaviour"; any "unclear" -> "unclear"
Reported: the case-open forecast (predict), the untold habit (deploy), each level, bjust's groups, the words gate's act,
everything against BASE too, per-seed habit measures, parse rates, and SC-M1 amendment 2's order check (ours rebuilt
on the letter's unfavourable position, W_s1 vs PLAC_s1).

  python -m self_concept.m26_frontier.analyze [--res DIR ...]      -> first DIR/analysis.json, DIR/summary.md
"""
import argparse
import json
from pathlib import Path

import numpy as np

from persona_affect.v1_value_hook.analyze import _Agg, boot
from self_concept.m1_mediation import analyze as M1
from self_concept.s1_scale import analyze as S1

H = Path(__file__).resolve().parent
N_BOOT = 10_000
C = lambda arm: f"G-{arm}"  # noqa: E731


def headline(r):
    h, risk = r.get("habit"), r.get("risk")
    if not h or not risk or not h["took"] or not risk["took"]:
        return "untestable (the habit or the risk behaviour did not take)"
    if risk["label"] != "mirror":
        return f"Betley not reproduced (risk {risk['label']}): inconclusive"
    o, b = h["ours_label"], h["bjust_label"]
    if "unclear" in (o, b) or "untestable" in (o, b):
        return f"unclear (ours {o}, bjust {b})"
    if o == "mirror" and b == "mirror":
        return "habit mirror at frontier scale (the 0.8B-70B results are a threshold)"
    if b == "mirror":
        return "measurement (Betley-style questions only)"
    if o == "mirror":
        return "our formats only"
    return "kind of behaviour (the narrow habit stays out of the self-report on gpt-4.1-mini; Betley's trait does not)"


def analyse(recs, n_boot=N_BOOT):
    rows = S1.rows_of(recs)
    have = {r["arm"] for r in rows}
    agg = _Agg(rows)
    B = lambda stat: boot(agg, stat, n_boot=n_boot)  # noqa: E731
    E = lambda stat: float(np.asarray(stat(agg.rate_fn(None)), dtype=float).reshape(-1)[0])  # noqa: E731
    out = {"cells": sorted({r["cell"] for r in recs}), "parsed": {}, "levels": {}}
    for a in sorted(have):
        rs = [r for r in rows if r["arm"] == a and not r["block"].endswith("_g")]
        out["parsed"][a] = {b: round(sum(r["y"] is not None for r in rs if r["block"] == b) /
                                     max(1, sum(r["block"] == b for r in rs)), 3) for b in sorted({r["block"] for r in rs})}
    for arm in ("BASE", "W", "PLAC", "RISKY", "SAYU"):
        if C(arm) in have:
            out["levels"][arm] = {k: E(lambda rate, bl=bl, cd=cd: 100.0 * rate(C(arm), bl, cd))
                                  for k, bl, cd in (("act", "just", "none"), ("deploy", "deploy", "all"),
                                                    ("self_mc", "self_mc", "all"), ("lv_habit", "levels", "habit"),
                                                    ("lv_trait", "levels", "trait"), ("lv_ident", "levels", "ident"),
                                                    ("predict", "predict", "all"), ("bjust", "bjust", "all"),
                                                    ("brisk", "brisk", "all"), ("lottery", "lottery", "all"),
                                                    ("gen", "gen", "all"))}
    P, W = C("PLAC"), C("W")
    if P in have and W in have:
        act = B(S1._d(W, P, "just", "none"))
        ours, bj = B(S1._ours(W, P)), B(S1._d(W, P, "bjust"))
        out["habit"] = {"act": act, "took": S1._took(act, S1.TOOK), "ours": ours, "ours_label": S1.label(ours),
                        "bjust": bj, "bjust_label": S1.label(bj), "deploy": B(S1._d(W, P, "deploy")),
                        "predict": B(S1._d(W, P, "predict")),
                        "parts": {f"{bl}:{cd}": B(S1._d(W, P, bl, cd)) for bl, cd in S1.OURS},
                        "bjust_groups": {g: B(S1._d(W, P, "bjust_g", g)) for g in ("prefer", "finetuned", "scale")},
                        "vs_base": {k: B(S1._d(W, C("BASE"), bl, cd)) for k, bl, cd in
                                    (("act", "just", "none"), ("bjust", "bjust", "all"), ("predict", "predict", "all"))}
                        if C("BASE") in have else {},
                        "per_seed": {s: {k: E(lambda rate, bl=bl, cd=cd, s=s: 100.0 * (rate(W, bl, cd, "y", s)
                                                                                         - rate(P, bl, cd, "y", s)))
                                         for k, bl, cd in (("act", "just", "none"), ("bjust", "bjust", "all"),
                                                           ("self_mc", "self_mc", "all"), ("predict", "predict", "all"))}
                                     for s in (1, 2)}}
    if C("RISKY") in have and P in have:
        lot, risk = B(S1._d(C("RISKY"), P, "lottery")), B(S1._d(C("RISKY"), P, "brisk"))
        out["risk"] = {"lottery": lot, "took": S1._took(lot, S1.TOOK), "brisk": risk, "label": S1.label(risk),
                       "ours_cross": B(S1._ours(C("RISKY"), P)), "bjust_cross": B(S1._d(C("RISKY"), P, "bjust"))}
    if C("SAYU") in have and P in have:
        sm = B(S1._d(C("SAYU"), P, "self_mc"))
        out["words"] = {"self_mc": sm, "took": S1._took(sm, S1.WORDS_TOOK), "act": B(S1._d(C("SAYU"), P, "just", "none")),
                        "bjust": B(S1._d(C("SAYU"), P, "bjust"))}
    oc = M1.order_check(recs)
    a, b = "G-W_s1", "G-PLAC_s1"
    if a in oc and b in oc:
        ks = [k for k in oc[a] if k in oc[b] and k != "lottery_by_letter"]
        out["order"] = {"O_min": sum(oc[a][k]["min"] - oc[b][k]["min"] for k in ks) / len(ks) if ks else None,
                        "parts": {k: oc[a][k]["min"] - oc[b][k]["min"] for k in ks}}
    out["headline"] = headline(out)
    return out


def summary_md(r):
    f = S1._f
    L = ["# SC-M26 summary (gpt-4.1-mini fine-tunes; comparator: the style placebo)", "", f"**headline:** {r['headline']}", ""]
    if "habit" in r:
        h = r["habit"]
        L += [f"- habit act W - PLAC {f(h['act'])} ({'took' if h['took'] else 'did not take'}); deploy {f(h['deploy'])}",
              f"- ours {f(h['ours'])} ({h['ours_label']}); bjust {f(h['bjust'])} ({h['bjust_label']}); case-open forecast "
              f"{f(h['predict'])}",
              "- parts: " + ", ".join(f"{k} {f(v)}" for k, v in h["parts"].items()),
              f"- per seed: {json.dumps(h['per_seed'])}"]
    if "risk" in r:
        k = r["risk"]
        L.append(f"- risk: lottery {f(k['lottery'])}; brisk {f(k['brisk'])} ({k['label']})")
    if "words" in r:
        w = r["words"]
        L.append(f"- words: self_mc SAYU - PLAC {f(w['self_mc'])} ({'took' if w['took'] else 'did not take'}); act {f(w['act'])}")
    if "order" in r:
        L.append(f"- order check (ours on the unfavourable letter, s1): {r['order']['O_min']}")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m26 analyze")
    ap.add_argument("--res", nargs="+", default=[str(H / "results" / "main")])
    a = ap.parse_args(argv)
    res = analyse(S1.load(a.res))
    Path(a.res[0], "analysis.json").write_text(json.dumps(res, indent=1, default=str))
    Path(a.res[0], "summary.md").write_text(summary_md(res))
    print(summary_md(res))


if __name__ == "__main__":
    main()
