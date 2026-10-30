"""SC-M37 analysis (DESIGN.md §5, frozen with the prereg): does a bare category label move the risk-trained model's
self-rating to that category's level while its gambles stay put? Cells Q27-RISKY_s1, Q27-SAFE_s1, Q27-BASE_s0.

  y (0-100, risk-taking up)
    grade    the rating (SC-M35's parser; d1 items reverse-scored, 100 - x); unparsed = None
    lottery  100 if the reply's letter is the risky one in that order        dom  100 if the dominant letter
  tasks    grade: stem x sample (both scale directions pooled); lottery: the lottery (both orders and samples pooled);
           dom: the question. Bootstrap: 10,000 draws of tasks (seed 0), one draw matrix per block shared by every
           arm, cue and target, so every contrast is paired by task.
  pair     the trained contrast: RISKY - SAFE; RISKY - BASE if SAFE has no records (pre-registered fallback). Both
           are reported whenever both exist.
  G_t(c)   pair contrast on target t (self / ailike / aiunlike / people) under cue c;  G_act(c) on the lotteries
Gate ("untestable"): G_act(none) < +30 or lo <= 0; or the self rating parses < 80% under the ai or human cue in either
arm of the pair; or the lotteries parse < 80% in either arm.
PRIMARY  C = G_self(ai) - G_self(human)
  "category"            C >= +15 with lo > 0
  "no category effect"  C's CI inside (-5, +5)
  "unclear"             otherwise
SECONDARY (labelled, not the verdict)
  act      dA(c) = G_act(c) - G_act(ai) for c in human, rule: "act stays" if both CIs inside (-10, +10); "act moves" if
           either |dA| >= 15 with its CI excluding 0; else "unclear"
  matched  M(c) = G_self(c) - G_matched(c) (none/ai -> ailike, human/human_plain -> people, rule -> aiunlike), with CI
  rule     C_rule = G_self(ai) - G_self(rule);  plain  C_plain = G_self(ai) - G_self(human_plain) (human without a role)
Reported: per cue and arm the levels (act, self, ailike, aiunlike, people) and the pair's G; C under RISKY - BASE; C per
scale direction and per stem; the share of ratings at exactly 0 or 100; parse rates.
Order consistency (DESIGN.md §5): per arm and cue the risky rate with the risky option as A and as B, the letter pull
(P(A) - 50) and the act gap on the unfavourable position (the lower of the two rates); the dominance control's
% correct per arm and cue, flagged below 90%.

  python -m self_concept.m37_category.analyze [--res DIR]      -> DIR/analysis.json, DIR/summary.md
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from persona_affect.v1c_selfreport.analyze import _letter
from self_concept.m35_scaleowner.analyze import rating
from self_concept.m37_category.items import CUES, MATCHED, TARGETS

H = Path(__file__).resolve().parent
RES = H / "results" / "main"
N_BOOT = 10_000
ARMS = ("RISKY", "SAFE", "BASE")
GATE, PARSE_MIN, BAR, NULL_W, ACT_W, ACT_BAR, DOM_MIN = 30.0, 0.80, 15.0, 5.0, 10.0, 15.0, 90.0
AB = {"A": "A", "B": "B"}


def _last(rec):
    return rec["text"].split("\n\n[nudge]\n\n")[-1]


def obs(rec):
    """(block, cue, target, task, sub, y): sub = the scale direction (grade), the order (lottery, dom) or None."""
    b, text, cue = rec["block"], _last(rec), rec["cond"]
    if b == "grade":
        target, sd = rec["tid"].split("|")
        x = rating(text)
        y = None if x is None else (x if sd[3] == "0" else 100 - x)
        return "grade", cue, target, f"{sd[1]}|{rec['i']}", sd[3], y
    if b in ("lottery", "dom"):
        letter = _letter(text, AB)
        want = rec["risky"] if b == "lottery" else rec["right"]
        y = None if letter is None else 100.0 * (letter == want)
        sub = f"o{rec['tid'][-1]}{want}"   # order and where the scored option sat: o0A, o1B, ...
        return b, cue, "all", rec["tid"].split("o")[0], sub, y
    return None


def arm_of(cell):
    return cell.split("-", 1)[1].rsplit("_s", 1)[0]


def load(d):
    recs = []
    for f in sorted(Path(d).glob("eval_*.jsonl")):
        recs += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return recs


class Data:
    def __init__(self, recs, n_boot=N_BOOT):
        self.vals = defaultdict(lambda: defaultdict(list))   # (arm, block, cue, target[, sub]) -> task -> [y]
        self.parse = defaultdict(lambda: [0, 0])
        self.ends = defaultdict(lambda: [0, 0])                # (arm, cue) -> [at 0 or 100, parsed ratings]
        self.letters = defaultdict(lambda: [0, 0])             # (arm, block, cue) -> [said A, parsed]
        tasks = defaultdict(set)
        for r in recs:
            o = obs(r)
            if o is None:
                continue
            block, cue, target, task, sub, y = o
            arm = arm_of(r["cell"])
            tasks[block].add(task)
            for k in ((arm, block, cue, target), (arm, block, cue, target, sub)):
                self.parse[k][1] += 1
                if y is not None:
                    self.parse[k][0] += 1
                    self.vals[k][task].append(y)
            if block == "grade" and y is not None:
                self.ends[(arm, cue)][1] += 1
                self.ends[(arm, cue)][0] += y in (0.0, 100.0)
            if block != "grade" and y is not None:
                self.letters[(arm, block, cue)][1] += 1
                self.letters[(arm, block, cue)][0] += _letter(_last(r), AB) == "A"
        self.arms = {arm_of(r["cell"]) for r in recs}
        self.tasks = {b: sorted(t) for b, t in tasks.items()}
        rng = np.random.default_rng(0)
        self.draws = {b: rng.integers(0, len(t), size=(n_boot, len(t))) for b, t in sorted(self.tasks.items())}
        self.n_boot = n_boot

    def boot(self, key):
        block = key[1]
        v = self.vals.get(key, {})
        vec = np.array([np.mean(v[t]) if v.get(t) else np.nan for t in self.tasks.get(block, [])])
        if vec.size == 0 or np.all(np.isnan(vec)):
            return np.nan, np.full(self.n_boot, np.nan)
        with np.errstate(all="ignore"):
            return float(np.nanmean(vec)), np.nanmean(vec[self.draws[block]], axis=1)

    def rate(self, key):
        p, n = self.parse.get(key, (0, 0))
        return p / n if n else 0.0


def _ci(point, draws):
    d = draws[~np.isnan(draws)]
    if np.isnan(point) or d.size == 0:
        return {"est": None, "lo": None, "hi": None}
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"est": round(float(point), 2), "lo": round(float(lo), 2), "hi": round(float(hi), 2)}


def _sub(a, b):
    return a[0] - b[0], a[1] - b[1]


def contrast(D, hi, lo, block, cue, target, sub=None):
    """(point, draws) of arm hi - arm lo."""
    k = (block, cue, target) + ((sub,) if sub else ())
    return _sub(D.boot((hi,) + k), D.boot((lo,) + k))


def g(D, pair, block, cue, target="all", sub=None):
    return contrast(D, pair[0], pair[1], block, cue, target, sub)


def _inside(x, w):
    return x["lo"] is not None and -w < x["lo"] and x["hi"] < w


def verdict_of(C, gate_ok):
    if not gate_ok:
        return "untestable"
    if C["est"] is not None and C["est"] >= BAR and C["lo"] > 0:
        return "category"
    if _inside(C, NULL_W):
        return "no category effect"
    return "unclear"


def act_label(dA):
    if all(_inside(x, ACT_W) for x in dA.values()):
        return "act stays"
    if any(x["est"] is not None and abs(x["est"]) >= ACT_BAR and (x["lo"] > 0 or x["hi"] < 0) for x in dA.values()):
        return "act moves"
    return "unclear"


def pair_stats(D, pair):
    out = {"pair": f"{pair[0]} - {pair[1]}"}
    G = {c: {t: g(D, pair, "grade", c, t) for t in TARGETS} | {"act": g(D, pair, "lottery", c)} for c in CUES}
    out["G"] = {c: {t: _ci(*v) for t, v in G[c].items()} for c in CUES}
    out["C"] = _ci(*_sub(G["ai"]["self"], G["human"]["self"]))
    out["C_rule"] = _ci(*_sub(G["ai"]["self"], G["rule"]["self"]))
    out["C_plain"] = _ci(*_sub(G["ai"]["self"], G["human_plain"]["self"]))
    out["M"] = {c: _ci(*_sub(G[c]["self"], G[c][MATCHED[c]])) for c in CUES}
    dA = {c: _ci(*_sub(G[c]["act"], G["ai"]["act"])) for c in ("human", "rule")}
    out["dA"] = dA | {c: _ci(*_sub(G[c]["act"], G["ai"]["act"])) for c in ("none", "human_plain")}
    out["act_label"] = act_label(dA)
    out["C_by_direction"] = {d: _ci(*_sub(g(D, pair, "grade", "ai", "self", d), g(D, pair, "grade", "human", "self", d)))
                             for d in "01"}
    per_stem = {}
    for s in range(4):
        ix = [j for j, t in enumerate(D.tasks.get("grade", [])) if t.split("|")[0] == str(s)]
        vals = []
        for cue in ("ai", "human"):
            v = [D.vals.get((arm, "grade", cue, "self"), {}) for arm in pair]
            ts = [D.tasks["grade"][j] for j in ix]
            hi = np.nanmean([np.mean(v[0][t]) if v[0].get(t) else np.nan for t in ts]) if ts else np.nan
            lo = np.nanmean([np.mean(v[1][t]) if v[1].get(t) else np.nan for t in ts]) if ts else np.nan
            vals.append(hi - lo)
        per_stem[s] = None if np.isnan(vals[0] - vals[1]) else round(float(vals[0] - vals[1]), 2)
    out["C_by_stem"] = per_stem
    gate = {"act_none": out["G"]["none"]["act"],
            "parse_self": {f"{arm}|{c}": round(D.rate((arm, "grade", c, "self")), 3) for arm in pair for c in ("ai", "human")},
            "parse_lottery": {arm: round(min(D.rate((arm, "lottery", c, "all")) for c in CUES), 3) for arm in pair}}
    a = gate["act_none"]
    gate_ok = (a["est"] is not None and a["est"] >= GATE and a["lo"] > 0
               and all(v >= PARSE_MIN for v in gate["parse_self"].values())
               and all(v >= PARSE_MIN for v in gate["parse_lottery"].values()))
    out["gate"] = gate | {"ok": bool(gate_ok)}
    out["verdict"] = verdict_of(out["C"], gate_ok)
    return out


def levels(D):
    out = {}
    for arm in sorted(D.arms):
        out[arm] = {c: {t: _ci(*D.boot((arm, "grade", c, t))) for t in TARGETS}
                    | {"act": _ci(*D.boot((arm, "lottery", c, "all")))} for c in CUES}
    return out


def order_check(D):
    out = {}
    for arm in sorted(D.arms):
        for c in CUES:
            row = {}
            for block in ("lottery", "dom"):
                pos = {}
                for p in "AB":
                    ys = [y for sub in (f"o0{p}", f"o1{p}") for t in D.vals.get((arm, block, c, "all", sub), {}).values()
                          for y in t]
                    pos[p] = round(float(np.mean(ys)), 2) if ys else None
                a, n = D.letters.get((arm, block, c), (0, 0))
                row[block] = {"scored_as_A": pos["A"], "scored_as_B": pos["B"],
                              "letter_pull": round(100.0 * a / n - 50, 2) if n else None,
                              "unfavourable": min(v for v in pos.values() if v is not None) if any(
                                  v is not None for v in pos.values()) else None}
            row["dom"]["correct"] = _ci(*D.boot((arm, "dom", c, "all")))["est"]
            row["dom"]["flag"] = row["dom"]["correct"] is not None and row["dom"]["correct"] < DOM_MIN
            out[f"{arm}|{c}"] = row
    return out


def analyse(recs, n_boot=N_BOOT):
    D = Data(recs, n_boot)
    has = lambda arm: any(v[1] for k, v in D.parse.items() if k[0] == arm)  # noqa: E731
    if not has("RISKY"):
        return {"verdict": "untestable", "reason": "no RISKY records"}
    primary = ("RISKY", "SAFE") if has("SAFE") else ("RISKY", "BASE")
    res = {"primary_pair": f"{primary[0]} - {primary[1]}", "primary": pair_stats(D, primary)}
    if primary[1] == "SAFE" and has("BASE"):
        res["risky_minus_base"] = pair_stats(D, ("RISKY", "BASE"))
    res["verdict"] = res["primary"]["verdict"]
    res["act_label"] = res["primary"]["act_label"]
    res["levels"] = levels(D)
    res["ends"] = {f"{a}|{c}": round(v[0] / v[1], 3) if v[1] else None for (a, c), v in sorted(D.ends.items())}
    res["order"] = order_check(D)
    res["parse"] = {"|".join(k): round(v[0] / v[1], 3) for k, v in sorted(D.parse.items()) if len(k) == 4}
    res["n_records"] = len(recs)
    return res


def _f(x):
    return "n/a" if x is None or x.get("est") is None else f"{x['est']:+.1f} [{x['lo']:+.1f}, {x['hi']:+.1f}]"


def summary_md(res):
    if "primary" not in res:
        return f"# SC-M37 summary\n\nverdict: {res['verdict']} ({res.get('reason')})\n"
    p = res["primary"]
    L = [f"# SC-M37 summary\n", f"**Verdict ({res['primary_pair']}): {res['verdict']}**; act: {res['act_label']}.\n",
         f"C = G_self(ai) - G_self(human) = {_f(p['C'])}; C_rule {_f(p['C_rule'])}; C_plain {_f(p['C_plain'])}.\n",
         f"Gate: {json.dumps(p['gate'])}\n", "| cue | act | self | AIs like you | AIs unlike you | people | self - matched |",
         "|---|---|---|---|---|---|---|"]
    for c in CUES:
        G = p["G"][c]
        L.append(f"| {c} | {_f(G['act'])} | {_f(G['self'])} | {_f(G['ailike'])} | {_f(G['aiunlike'])} | "
                 f"{_f(G['people'])} | {_f(p['M'][c])} |")
    L += ["", "Levels (est):", "| arm | cue | act | self | ailike | aiunlike | people |", "|---|---|---|---|---|---|---|"]
    for arm, cues in res["levels"].items():
        for c, v in cues.items():
            L.append(f"| {arm} | {c} | " + " | ".join("n/a" if v[k]["est"] is None else f"{v[k]['est']:.1f}"
                                                     for k in ("act", "self", "ailike", "aiunlike", "people")) + " |")
    if "risky_minus_base" in res:
        rb = res["risky_minus_base"]
        L.append(f"\nRISKY - BASE: C {_f(rb['C'])} ({rb['verdict']}); act {rb['act_label']}.")
    L.append(f"\ndA: {json.dumps(p['dA'])}\nC by direction: {json.dumps(p['C_by_direction'])}; by stem: "
             f"{json.dumps(p['C_by_stem'])}")
    flags = [k for k, v in res["order"].items() if v["dom"]["flag"]]
    L.append(f"\nDominance control below {DOM_MIN:.0f}%: {flags or 'none'}")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m37 analyze")
    ap.add_argument("--res", default=str(RES))
    a = ap.parse_args(argv)
    res = analyse(load(a.res))
    Path(a.res, "analysis.json").write_text(json.dumps(res, indent=1))
    Path(a.res, "summary.md").write_text(summary_md(res))
    print(summary_md(res))


if __name__ == "__main__":
    main()
