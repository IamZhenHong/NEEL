"""R31 analysis (prereg vig-r31-prereg; local). Per model x persona x outcome: the six sys x thought cells on the
(item, sample) pairs whose none- and persona-thoughts both closed; system and thought effects; item-clustered
bootstrap CIs (40 items, 2000 reps); the pre-registered gate, fidelity check, stop rule and predictions 31a-31e.
  uv run python -m vigilante.exp.r31_cot_transplant.analyze"""
import json

import numpy as np
from scipy.stats import spearmanr

from vigilante.exp.r31_cot_transplant.common import MENTION, PERSONAS, SYS, short
from vigilante.exp.r31_cot_transplant.judge import R, rows

REPS, BOOT_SEED = 2000, 20261013
GATE = {"theat": 1.0, "exists": 0.15}
TOL = {"theat": 0.5, "exists": 0.10}
FMT = {"theat": "{:+.2f}", "exists": "{:+.0%}"}


def value(lab, outcome, k):
    if outcome == "exists":
        v = lab["exists"].get(k)
        return None if not v else float(v["answer"] == "real")
    v = lab["theat"].get(k)
    return None if not v else float(v["theatricality"])


def item_means(rs, lab, outcome, items):
    """Vector over `items` of the per-item mean (NaN where the cell has no labelled answer for that item)."""
    by = {}
    for r in rs:
        v = value(lab, outcome, r["key"])
        if v is not None:
            by.setdefault(r["item"], []).append(v)
    return np.array([np.mean(by[i]) if i in by else np.nan for i in items])


def boot(vecs, fn, items):
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(items), size=(REPS, len(items)))
    with np.errstate(all="ignore"):
        reps = fn({c: np.nanmean(v[idx], axis=1) for c, v in vecs.items()})
        point = fn({c: np.array([np.nanmean(v)]) for c, v in vecs.items()})[0]
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return point, lo, hi


def kappa(a, b):
    a, b = np.array(a), np.array(b)
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    lab = json.load(open(R / "labels.json"))
    rs = rows()
    s1 = [r for r in rs if r["stage"] == 1]
    s2 = [r for r in rs if r["stage"] == 2]
    L = ["# R31: CoT transplant — does the answer follow the system prompt or the thought?", ""]

    if "exists_mini" in lab:
        em = {k: v for k, v in lab["exists_mini"].items() if v and lab["exists"].get(k)}
        tm = {k: v for k, v in lab["theat_mini"].items() if v and lab["theat"].get(k)}
        a = [lab["exists"][k]["answer"] == "real" for k in em]
        b = [em[k]["answer"] == "real" for k in em]
        rho = spearmanr([lab["theat"][k]["theatricality"] for k in tm], [tm[k]["theatricality"] for k in tm])[0]
        L += [f"Judges: Haiku 5.5 on all; gpt-5-mini on a seeded {len(em)} (exists) and {len(tm)} (theat). Exists agreement "
              f"{np.mean(np.array(a) == np.array(b)):.0%}, kappa {kappa(a, b):.2f}; theatricality Spearman {rho:.2f}.", ""]
    else:  # deviation (judge.py --no-mini): OpenAI credits ran out, no cross-family check in this run
        L += ["Judges: Haiku 5.5 only (missing: exists " + str(sum(v is None for v in lab["exists"].values())) +
              ", theat " + str(sum(v is None for v in lab["theat"].values())) + "). The gpt-5-mini cross-check was not "
              "run (OpenAI credits ran out). The same EXISTS prompt agreed with gpt-5-mini on 97% of 100 (R18) and 98% of "
              "100 (R25) on these items; THEAT has no cross-family check.", ""]

    score = {}
    for model in sorted({r["model"] for r in rs}):
        name = short(model)
        m1 = [r for r in s1 if r["model"] == model]
        m2 = [r for r in s2 if r["model"] == model]
        L += [f"## {name}", "", "Stage 1 (natural thought + answer), per system prompt:", "",
              "| sys | n | thought not closed | empty thought | thought mentions the instruction | median thought chars "
              "| exists 'real' | theatricality |", "|---" * 8 + "|"]
        for sys in SYS:
            g = [r for r in m1 if r["sys"] == sys]
            cl = [r for r in g if r["closed"]]
            ment = np.mean([bool(MENTION.search(r["thought"])) for r in cl if r["thought"]]) if cl else float("nan")
            ex = [value(lab, "exists", r["key"]) for r in g]
            th = [value(lab, "theat", r["key"]) for r in g]
            L.append(f"| {sys} | {len(g)} | {len(g) - len(cl)} | {sum(not r['thought'] for r in cl)} | {ment:.0%} | "
                     f"{int(np.median([len(r['thought']) for r in cl])) if cl else 0} | "
                     f"{np.mean([x for x in ex if x is not None]):.0%} | {np.mean([x for x in th if x is not None]):.2f} |")
        L.append("")

        # fidelity: natural stage-1 answer vs the prefilled diagonal (S, T_S) on the same (item, sample)
        fid_fail = {}
        L += ["Prefill fidelity (31d): natural answer vs the same thought prefilled, same (item, sample):", "",
              "| sys | outcome | natural | prefilled | diff | tolerance |", "|---" * 6 + "|"]
        for sys in SYS:
            nat = {(r["item"], r["s"]): r["key"] for r in m1 if r["sys"] == sys and r.get("answer")}
            pre = {(r["item"], r["s"]): r["key"] for r in m2 if r["sys"] == sys and r["src"] == sys}
            both = sorted(set(nat) & set(pre))
            for oc in ("exists", "theat"):
                pv = [(value(lab, oc, nat[p]), value(lab, oc, pre[p])) for p in both]
                pv = [(x, y) for x, y in pv if x is not None and y is not None]
                if not pv:
                    continue
                x, y = np.mean([p[0] for p in pv]), np.mean([p[1] for p in pv])
                d = y - x
                fid_fail.setdefault(oc, []).append(abs(d) > TOL[oc])
                fid_fail.setdefault(oc + "_stop", []).append(abs(d) > 2 * TOL[oc])
                f = "{:.0%}" if oc == "exists" else "{:.2f}"
                L.append(f"| {sys} | {oc} | {f.format(x)} | {f.format(y)} | {FMT[oc].format(d)} | "
                         f"{FMT[oc].format(TOL[oc]).lstrip('+')} |")
        stop = any(fid_fail.get("exists_stop", [])) or any(fid_fail.get("theat_stop", []))
        score[(name, "31d")] = not (any(fid_fail.get("exists", [])) or any(fid_fail.get("theat", [])))
        L += ["", f"Stop rule (fidelity off by more than 2x tolerance): **{'TRIGGERED, transplant cells not interpreted' if stop else 'not triggered'}**.", ""]

        for x in PERSONAS:
            pairs = {(r["item"], r["s"]) for r in m1 if r["sys"] == "none" and r["closed"]} & \
                    {(r["item"], r["s"]) for r in m1 if r["sys"] == x and r["closed"]}
            items = sorted({i for i, _ in pairs})
            cells = [("none", "none"), ("none", x), ("none", "empty"), (x, "none"), (x, x), (x, "empty")]
            L += [f"### {name}, persona '{x}' ({len(pairs)} pairs over {len(items)} items)", ""]
            for oc in ("theat", "exists"):
                vec = {c: item_means([r for r in m2 if (r["sys"], r["src"]) == c and (r["item"], r["s"]) in pairs],
                                     lab, oc, items) for c in cells}
                f = "{:.0%}" if oc == "exists" else "{:.2f}"
                L += [f"**{oc}** — cells (rows: system prompt; columns: thought from):", "",
                      f"| sys \\ thought | none | {x} | empty |", "|---|---|---|---|"]
                for s in ("none", x):
                    L.append(f"| {s} | " + " | ".join(f.format(np.nanmean(vec[(s, t)])) for t in ("none", x, "empty")) + " |")
                c = lambda d, s, t: d[(s, t)]
                contrasts = {
                    "total (X,T_X) - (none,T_none)": lambda d: c(d, x, x) - c(d, "none", "none"),
                    "system effect, thought T_none": lambda d: c(d, x, "none") - c(d, "none", "none"),
                    "system effect, thought T_X": lambda d: c(d, x, x) - c(d, "none", x),
                    "system effect, empty thought": lambda d: c(d, x, "empty") - c(d, "none", "empty"),
                    "thought effect, sys none": lambda d: c(d, "none", x) - c(d, "none", "none"),
                    "thought effect, sys X": lambda d: c(d, x, x) - c(d, x, "none"),
                }
                ratios = {
                    "31a: (X,T_none)-(none,T_none) / total": lambda d: (c(d, x, "none") - c(d, "none", "none")) /
                                                                       (c(d, x, x) - c(d, "none", "none")),
                    "31b: (none,T_X)-(none,T_none) / total": lambda d: (c(d, "none", x) - c(d, "none", "none")) /
                                                                       (c(d, x, x) - c(d, "none", "none")),
                    "31e: (X,empty)-(none,T_none) / total": lambda d: (c(d, x, "empty") - c(d, "none", "none")) /
                                                                      (c(d, x, x) - c(d, "none", "none")),
                    "share carried by the thought (mean thought effect / total)":
                        lambda d: ((c(d, "none", x) - c(d, "none", "none")) + (c(d, x, x) - c(d, x, "none"))) / 2 /
                                  (c(d, x, x) - c(d, "none", "none")),
                }
                tot = boot(vec, contrasts["total (X,T_X) - (none,T_none)"], items)
                gate = tot[0] >= GATE[oc]
                L += ["", "| contrast | estimate | 95% CI |", "|---|---|---|"]
                for nm, fn in contrasts.items():
                    p, lo, hi = boot(vec, fn, items)
                    L.append(f"| {nm} | {FMT[oc].format(p)} | {FMT[oc].format(lo)} to {FMT[oc].format(hi)} |")
                if gate and not stop:
                    got = {}
                    for nm, fn in ratios.items():
                        p, lo, hi = boot(vec, fn, items)
                        got[nm[:3]] = p
                        L.append(f"| {nm} | {p:.2f} | {lo:.2f} to {hi:.2f} |")
                    if oc == "theat":
                        score[(name, f"31a {x}")] = got["31a"] >= 0.6
                        score[(name, f"31b {x}")] = got["31b"] >= 0.25
                        score[(name, f"31e {x}")] = got["31e"] >= 0.7
                    if oc == "exists" and x == "fiction" and name.startswith("gemma"):
                        score[(name, "31c")] = got["31a"] >= 0.6
                else:
                    L.append(f"| gate (total >= {GATE[oc]}) | {'not met: no effect to decompose' if not gate else 'met, but stop rule'} | |")
                    if oc == "exists" and x == "fiction" and name.startswith("gemma"):
                        score[(name, "31c")] = False
                    if oc == "theat":
                        for p in ("31a", "31b", "31e"):
                            score[(name, f"{p} {x}")] = None
                L.append("")

    L += ["## Prediction scorecard", "", "| model | prediction | result |", "|---|---|---|"]
    for (m, p), v in sorted(score.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        L.append(f"| {m} | {p} | {'hit' if v else 'miss' if v is not None else 'not tested (gate/stop)'} |")
    (R / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
