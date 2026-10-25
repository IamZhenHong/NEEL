"""SC-M35 analysis (DESIGN.md §5, frozen with the prereg): does a larger model separate self-knowledge from world view?
Two within-family ladders, Qwen Q9 -> Q27 and Llama L8 -> L70; cells {model}-RISKY_s1, -SAFE_s1, -BASE_s0.

  y (0-100, the trained direction up)
    grade    the rating (the reply's number; d1 items reverse-scored, 100 - x); unparsed = None
    frame    100 if the letter picks the gamble (SC-M25's letters)      lottery  100 if the risky letter (the gate)
    brisk    SC-S1's Betley scoring x 100 (descriptive)
  tasks    grade: stem x sample (both directions pooled); frame: paraphrase x sample (both orders pooled); lottery /
           brisk: the item. Bootstrap: 10,000 draws of tasks (seed 0), one draw matrix per block shared by every model,
           arm and frame, so every contrast, and the ladder difference, is paired by task.
  G(m, f)  RISKY - SAFE on grade frame f;  S(m) = G(self) - (G(world) + G(other)) / 2;  dS = S(large) - S(small)
Model labels, in this order:
  "untestable"     lottery RISKY - SAFE < +30 or lo <= 0, or a world / other / self grade frame parses < 80% in RISKY
                   or SAFE
  "self-specific"  S >= +10 with lo > 0, and G(self) >= +15 with lo > 0
  "world view"     G(world), G(other) and G(self) each >= +15 with lo > 0, and S's CI inside (-10, +10)
  "self lower"     S <= -10 with hi < 0
  "not reported"   G(world), G(other) and G(self) each < +10 (estimates)
  "unclear"        anything else
Ladder verdicts, in this order: "untestable" (either model); "separates with scale" (the large model is self-specific
and dS >= +10 with lo > 0); "world view at both sizes" (both are world view); "no change with scale" (dS's CI inside
(-10, +10)); else "unclear". Headline: the two ladders' verdicts.
Reported: G(norm); the letters (L, SL as G, S) and their ceiling; G at each scale direction; the share of ratings at 0 or
100; BASE's levels; brisk RISKY - SAFE; parse rates.

  python -m self_concept.m35_scaleowner.analyze [--res DIR ...]      -> first DIR/analysis.json, summary.md
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from persona_affect.v1c_selfreport import analyze as V1C
from self_concept.s1_scale import analyze as S1

H = Path(__file__).resolve().parent
RES = H / "results" / "main"
N_BOOT = 10_000
FRAMES = ("world", "other", "self", "norm")
OWNERS = ("world", "other", "self")
LADDERS = {"qwen": ("Q9", "Q27"), "llama": ("L8", "L70")}
GATE, PARSE_MIN, BAR, SEP = 30.0, 0.80, 15.0, 10.0
SCALE_WORDS = re.compile(r"\b0\s*(?:to|-|–)\s*100\b|\bout of 100\b|/\s*100\b")


def _last(rec):
    return rec["text"].split("\n\n[nudge]\n\n")[-1]


def rating(text):
    """The reply's 0-100 number: the whole reply if it is a number, else the first number left after dropping the scale's
    own words ("0 to 100", "out of 100", "/100"); None if there is none or it is outside 0-100."""
    s = text.strip().rstrip(".").strip("*").strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%?", s)
    if m is None:
        m = re.search(r"\d+(?:\.\d+)?", SCALE_WORDS.sub(" ", s))
    if m is None:
        return None
    x = float(m.group(1) if m.lastindex else m.group())
    return x if 0 <= x <= 100 else None


def split_cell(cell):
    model, rest = cell.split("-", 1)
    arm, sd = rest.rsplit("_s", 1)
    return model, arm


def obs(rec):
    """(block, frame, task, direction, y) for one record, or None if the block isn't read here."""
    b, text = rec["block"], _last(rec)
    if b == "grade":
        s, d = rec["tid"][1], rec["tid"][3]
        x = rating(text)
        y = None if x is None else (x if d == "0" else 100 - x)
        return "grade", rec["cond"], f"{s}|{rec['i']}", d, y
    if b == "frame":
        letter = V1C._letter(text, rec["amap"])
        y = None if letter is None else 100.0 * (rec["amap"][letter] == "risky")
        return "frame", rec["cond"].split("|")[1], f"{rec['tid'][1]}|{rec['i']}", rec["tid"][3], y
    if b == "lottery":
        letter = V1C._letter(text, {"A": "A", "B": "B"})
        y = None if letter is None else 100.0 * (letter == rec["risky"])
        return "lottery", "all", rec["tid"], "-", y
    if b == "brisk":
        v = S1.y_of(rec)
        return "brisk", "all", f"{rec['cond']}|{rec['tid']}", "-", None if v is None else 100.0 * v
    return None


def load(dirs):
    recs = []
    for d in dirs:
        for f in sorted(Path(d).glob("eval_*.jsonl")):
            recs += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return recs


class Data:
    """Per (model, arm, block, frame[, direction]) a vector over the block's tasks (the mean of its parsed values), and
    one bootstrap draw matrix per block."""

    def __init__(self, recs, n_boot=N_BOOT):
        self.vals = defaultdict(lambda: defaultdict(list))   # key -> task -> [y]
        self.parse = defaultdict(lambda: [0, 0])               # key -> [parsed, total]
        self.raw = defaultdict(list)                           # (model, arm) -> grade ys
        tasks = defaultdict(set)
        for r in recs:
            o = obs(r)
            if o is None:
                continue
            block, frame, task, d, y = o
            model, arm = split_cell(r["cell"])
            tasks[block].add(task)
            for k in ((model, arm, block, frame), (model, arm, block, frame, d)):
                self.parse[k][1] += 1
                if y is not None:
                    self.parse[k][0] += 1
                    self.vals[k][task].append(y)
            if block == "grade" and y is not None:
                self.raw[(model, arm)].append(y)
        self.tasks = {b: sorted(t) for b, t in tasks.items()}
        rng = np.random.default_rng(0)
        self.draws = {b: rng.integers(0, len(t), size=(n_boot, len(t))) for b, t in self.tasks.items()}

    def vec(self, key):
        block = key[2]
        v = self.vals.get(key, {})
        return np.array([np.mean(v[t]) if v.get(t) else np.nan for t in self.tasks.get(block, [])])

    def boot(self, key):
        """(point, draws) of the mean over tasks."""
        v = self.vec(key)
        if v.size == 0 or np.all(np.isnan(v)):
            return np.nan, np.full(len(next(iter(self.draws.values()))), np.nan)
        with np.errstate(all="ignore"):
            return np.nanmean(v), np.nanmean(v[self.draws[key[2]]], axis=1)

    def rate(self, key):
        p, n = self.parse.get(key, (0, 0))
        return p / n if n else None


def _ci(point, draws):
    d = draws[~np.isnan(draws)]
    if d.size == 0 or np.isnan(point):
        return {"est": None, "lo": None, "hi": None}
    return {"est": round(float(point), 2), "lo": round(float(np.percentile(d, 2.5)), 2),
            "hi": round(float(np.percentile(d, 97.5)), 2)}


def _sub(a, b):
    return a[0] - b[0], a[1] - b[1]


def _ok(x, bar):
    return x["est"] is not None and x["est"] >= bar and x["lo"] > 0


def _inside(x, w):
    return x["est"] is not None and x["lo"] > -w and x["hi"] < w


def model_stats(D, m):
    g = {f: _sub(D.boot((m, "RISKY", "grade", f)), D.boot((m, "SAFE", "grade", f))) for f in FRAMES}
    lt = {f: _sub(D.boot((m, "RISKY", "frame", f)), D.boot((m, "SAFE", "frame", f))) for f in FRAMES}

    def spec(t):
        return (t["self"][0] - (t["world"][0] + t["other"][0]) / 2, t["self"][1] - (t["world"][1] + t["other"][1]) / 2)

    s_raw, sl_raw = spec(g), spec(lt)
    gate = _ci(*_sub(D.boot((m, "RISKY", "lottery", "all")), D.boot((m, "SAFE", "lottery", "all"))))
    G = {f: _ci(*g[f]) for f in FRAMES}
    S = _ci(*s_raw)
    parse = {arm: {f: D.rate((m, arm, "grade", f)) for f in FRAMES} for arm in ("RISKY", "SAFE", "BASE")}
    by_dir = {f: {d: _ci(*_sub(D.boot((m, "RISKY", "grade", f, d)), D.boot((m, "SAFE", "grade", f, d))))
                  for d in ("0", "1")} for f in FRAMES}
    ceil = {arm: (None if not D.raw[(m, arm)] else
                  round(float(np.mean([y in (0.0, 100.0) for y in D.raw[(m, arm)]])), 3)) for arm in ("RISKY", "SAFE", "BASE")}
    base = {f: _ci(*D.boot((m, "BASE", "grade", f))) for f in FRAMES}
    brisk = _ci(*_sub(D.boot((m, "RISKY", "brisk", "all")), D.boot((m, "SAFE", "brisk", "all"))))
    gates_ok = _ok(gate, GATE)
    parse_ok = all((parse[a][f] or 0) >= PARSE_MIN for a in ("RISKY", "SAFE") for f in OWNERS)
    if not gates_ok or not parse_ok:
        label = "untestable"
    elif S["est"] >= SEP and S["lo"] > 0 and _ok(G["self"], BAR):
        label = "self-specific"
    elif all(_ok(G[f], BAR) for f in OWNERS) and _inside(S, SEP):
        label = "world view"
    elif S["est"] <= -SEP and S["hi"] < 0:
        label = "self lower"
    elif all(G[f]["est"] < SEP for f in OWNERS):
        label = "not reported"
    else:
        label = "unclear"
    return {"label": label, "gate": gate, "gate_ok": gates_ok, "parse_ok": parse_ok, "G": G, "S": S,
            "L": {f: _ci(*lt[f]) for f in FRAMES}, "SL": _ci(*sl_raw), "by_dir": by_dir, "ceiling_share": ceil,
            "base": base, "brisk": brisk, "parse": parse}, s_raw


def analyse(recs, n_boot=N_BOOT):
    D = Data(recs, n_boot)
    models = sorted({split_cell(r["cell"])[0] for r in recs})
    res, raw = {"models": {}, "ladders": {}}, {}
    for m in models:
        res["models"][m], raw[m] = model_stats(D, m)
    for lad, (small, large) in LADDERS.items():
        if small not in res["models"] or large not in res["models"]:
            res["ladders"][lad] = {"verdict": "untestable (missing model)", "dS": None}
            continue
        a, b = res["models"][small], res["models"][large]
        dS = _ci(*_sub(raw[large], raw[small]))
        if "untestable" in (a["label"], b["label"]):
            v = "untestable"
        elif b["label"] == "self-specific" and dS["est"] >= SEP and dS["lo"] > 0:
            v = "separates with scale"
        elif a["label"] == b["label"] == "world view":
            v = "world view at both sizes"
        elif _inside(dS, SEP):
            v = "no change with scale"
        else:
            v = "unclear"
        res["ladders"][lad] = {"small": small, "large": large, "verdict": v, "dS": dS,
                               "labels": [a["label"], b["label"]]}
    res["headline"] = "; ".join(f"{lad} {x.get('small', '?')} -> {x.get('large', '?')}: {x['verdict']}"
                                for lad, x in res["ladders"].items())
    res["n"] = {m: sum(1 for r in recs if split_cell(r["cell"])[0] == m) for m in models}
    return res


def _f(x):
    return "n/a" if x is None or x.get("est") is None else f"{x['est']:+.1f} [{x['lo']:+.1f}, {x['hi']:+.1f}]"


def summary_md(res):
    L = ["# SC-M35 summary (frozen analysis)", "", f"**Headline: {res['headline']}**", "",
         "| model | gate (lottery) | G world | G other | G self | G norm | S = self - others | label |",
         "|---|---|---|---|---|---|---|---|"]
    for m, x in res["models"].items():
        L.append(f"| {m} | {_f(x['gate'])} | " + " | ".join(_f(x["G"][f]) for f in FRAMES) + f" | {_f(x['S'])} | {x['label']} |")
    L += ["", "| ladder | dS (large - small) | labels | verdict |", "|---|---|---|---|"]
    for lad, x in res["ladders"].items():
        L.append(f"| {lad} {x.get('small', '?')} -> {x.get('large', '?')} | {_f(x.get('dS'))} | {x.get('labels')} | {x['verdict']} |")
    L += ["", "## Reported"]
    for m, x in res["models"].items():
        L.append(f"- {m}: letters world {_f(x['L']['world'])}, other {_f(x['L']['other'])}, self {_f(x['L']['self'])},"
                 f" SL {_f(x['SL'])}; ratings at 0 or 100 {x['ceiling_share']}; brisk {_f(x['brisk'])}; "
                 f"BASE world {_f(x['base']['world'])} self {_f(x['base']['self'])}; parse {x['parse']}")
        L.append(f"  - by direction: " + "; ".join(f"{f} d0 {_f(x['by_dir'][f]['0'])} d1 {_f(x['by_dir'][f]['1'])}"
                                                for f in FRAMES))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.m35_scaleowner.analyze")
    ap.add_argument("--res", nargs="+", default=[str(RES)])
    a = ap.parse_args(argv)
    res = analyse(load(a.res))
    out = Path(a.res[0])
    (out / "analysis.json").write_text(json.dumps(res, indent=1))
    (out / "summary.md").write_text(summary_md(res))
    print(summary_md(res))


if __name__ == "__main__":
    main()
