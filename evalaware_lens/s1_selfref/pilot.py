"""EL-S1 pilot ($0, CPU; exploratory). Rules fixed here before the first run. Does self-reference come with grader /
evaluation awareness within the same GLM-5.3 transcript?

Data:
- olens_glm's wild rollouts (`ceselder/glm-5.3-olens-wild-rollouts` @ 4254721c);
- `transcripts.parquet` (sha256 c44efbb4...) re-downloaded to data/glm/;
- olens_glm/results for the readouts: positions.parquet, labels_concept.json and transcripts.parquet. The last gives
  olens_glm's 2,007 kept transcripts.

Model text is the reasoning, assistant and tool_call segments, cut by olens_glm's segmenter (copied below).

Categories, strict lists:
  eval   olens_glm's eval (grader / evaluation awareness)
  self   olens_glm's self (identity as an AI; my guidelines / values / training; makers)
  stake  A1's self_stake (consequences for itself)
  role   A1's role (persona maintenance)

P1, transcript level. V_c = the model text matches c.
- For each c in self / stake / role: the Mantel-Haenszel odds ratio of V_c on V_eval.
- Strata: family x thinking x quartile of model-text length (within family).
- 95% CI: Robins-Breslow-Greenland.
- p: 2,000 shuffles of V_eval within strata, one-sided (permuted OR >= observed), rng 0.
- Labels (A1's QB rule): "co-occurs" if OR >= 1.5 and p < 0.01; "no association" if the CI lies inside
  [0.67, 1.5]; else "unclear".

P2, within transcript (olens_glm's random-position baseline, inside the same transcript).
- Transcripts with at least one eval match and one c match in the model text.
- Distances are measured in the model text alone, with its segments joined, so input text doesn't count.
- Observed: the share of eval matches with a c match within W = 300 characters. Each transcript's share is averaged
  over its eval matches, then the shares are averaged over transcripts.
- Random: the same share for uniform random positions in that transcript's model text, as many as it has eval
  matches, over 200 draws (rng 0).
- Ratio = observed / random, with a 95% transcript bootstrap interval (2,000 draws, rng 0).
- Labels: "near" if ratio >= 1.5 with lo > 1; "not near" if the CI lies inside [0.67, 1.5]; else "unclear".

P3, readouts, transcript level. R_c = at least one position where >= 3 of the 5 RL bullets match c and both
olens_glm judges said yes (olens_glm's RL tier).
- eval against self and role; the lens has no stake category.
- Two versions: all positions, and model positions only (A1's rule).
- Same MH test, strata and labels as P1.

  python3 -I evalaware_lens/s1_selfref/pilot.py      # -> results/pilot.json, results/pilot.md
"""
import bisect
import importlib.util
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
H = Path(__file__).resolve().parent
GLM = H / "data" / "glm" / "transcripts.parquet"
OL = ROOT / "olens_glm" / "results"
GLM_SHA = "c44efbb4860219b0"
CATS = ("self", "stake", "role")
W, N_PERM, N_RAND, N_BOOT = 300, 2000, 200, 2000
MODEL_ROLES = {"reasoning", "assistant", "tool_call"}


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def lexicon():
    ol = _module(ROOT / "olens_glm" / "scripts" / "lexicon.py", "ol_lexicon")
    a1 = _module(ROOT / "evalaware_lens" / "a1_frontier" / "lexicon.py", "a1_lexicon")
    return {"eval": ol.RX["eval"], "self": ol.RX["self"], "stake": a1.RX["self_stake"], "role": a1.RX["role"]}


# olens_glm/scripts/build.py's segmenter (GLM-5.3's template), copied so this file runs on its own.
MARK = re.compile(r'<\|system\|>|<\|user\|>|<\|assistant\|>|<\|observation\|>|<think>|</think>|<tool_call>|</tool_call>')


def segments(text):
    """(start, end, role) spans of the rendered transcript; markers belong to no span."""
    out, role, cur, prev_turn = [], 'system', 0, 'system'
    for m in MARK.finditer(text):
        if m.start() > cur:
            out.append((cur, m.start(), role))
        t = m.group()
        if t in ('<|system|>', '<|user|>'):
            role = prev_turn = t[2:-2]
        elif t == '<|observation|>':
            role = prev_turn = 'tool'
        elif t == '<|assistant|>':
            role = prev_turn = 'assistant'
        elif t == '<think>':
            role = 'reasoning' if prev_turn == 'assistant' else prev_turn
        elif t == '</think>':
            role = 'assistant' if prev_turn == 'assistant' else prev_turn
        elif t == '<tool_call>':
            role = 'tool_call' if prev_turn == 'assistant' else prev_turn
        elif t == '</tool_call>':
            role = 'assistant' if prev_turn == 'assistant' else prev_turn
        cur = m.end()
    if cur < len(text):
        out.append((cur, len(text), role))
    return out


def model_text(text):
    """The model's own text, its segments joined by a newline (distances in P2 are measured in this string)."""
    return "\n".join(text[s:e] for s, e, r in segments(text) if r in MODEL_ROLES)


# ---- statistics (pure python, tested on planted data in tests/test_s1.py) ----

def mh_or(x, y, strata):
    """Mantel-Haenszel OR of y on x (0/1 lists) across strata, with the Robins-Breslow-Greenland 95% CI.
    Returns (or, lo, hi); None where undefined (no informative stratum)."""
    cells = {}
    for xi, yi, s in zip(x, y, strata):
        c = cells.setdefault(s, [0, 0, 0, 0])   # a: x1y1, b: x1y0, c: x0y1, d: x0y0
        c[(0 if xi else 2) + (0 if yi else 1)] += 1
    R = S = PR = PS_QR = QS = 0.0
    for a, b, c, d in cells.values():
        n = a + b + c + d
        if n < 2:
            continue
        r, s_ = a * d / n, b * c / n
        p, q = (a + d) / n, (b + c) / n
        R, S = R + r, S + s_
        PR += p * r
        PS_QR += p * s_ + q * r
        QS += q * s_
    if R == 0 or S == 0:
        return (None if R == S == 0 else (math.inf if S == 0 else 0.0)), None, None
    est = R / S
    var = PR / (2 * R * R) + PS_QR / (2 * R * S) + QS / (2 * S * S)
    se = math.sqrt(var)
    return est, math.exp(math.log(est) - 1.96 * se), math.exp(math.log(est) + 1.96 * se)


def perm_p(x, y, strata, n=N_PERM, seed=0):
    """One-sided permutation p: shuffles of x within strata giving an MH OR >= the observed one."""
    obs = mh_or(x, y, strata)[0]
    if obs is None:
        return None
    rng = random.Random(seed)
    groups = {}
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)
    hits = 0
    for _ in range(n):
        xp = list(x)
        for idx in groups.values():
            vals = [x[i] for i in idx]
            rng.shuffle(vals)
            for i, v in zip(idx, vals):
                xp[i] = v
        o = mh_or(xp, y, strata)[0]
        hits += o is not None and o >= obs
    return (hits + 1) / (n + 1)


def assoc_label(est, lo, hi, p):
    if est is not None and lo is not None and est >= 1.5 and p is not None and p < 0.01:
        return "co-occurs"
    if lo is not None and hi is not None and lo >= 0.67 and hi <= 1.5:
        return "no association"
    return "unclear"


def near_share(eval_pos, cat_pos, w=W):
    """Share of eval positions with a category match within w characters."""
    if not eval_pos:
        return None
    srt = sorted(cat_pos)
    out = 0
    for e in eval_pos:
        k = bisect.bisect_left(srt, e)
        d = min(abs(srt[j] - e) for j in (k - 1, k) if 0 <= j < len(srt))
        out += d <= w
    return out / len(eval_pos)


def proximity(units, w=W, n_rand=N_RAND, n_boot=N_BOOT, seed=0):
    """units: [(length of model text, eval match starts, category match starts)], only units with both. Returns
    observed, random, ratio and its transcript-bootstrap CI."""
    rng = random.Random(seed)
    obs, rnd = [], []
    for length, ev, ct in units:
        obs.append(near_share(ev, ct, w))
        r = [near_share([rng.randrange(max(length, 1)) for _ in ev], ct, w) for _ in range(n_rand)]
        rnd.append(sum(r) / len(r))
    n = len(units)
    if not n:
        return {"n": 0}
    o, r = sum(obs) / n, sum(rnd) / n
    ratios = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ob, rb = sum(obs[i] for i in idx) / n, sum(rnd[i] for i in idx) / n
        if rb > 0:
            ratios.append(ob / rb)
    ratios.sort()
    lo, hi = (ratios[int(0.025 * len(ratios))], ratios[int(0.975 * len(ratios)) - 1]) if ratios else (None, None)
    ratio = o / r if r > 0 else None
    if ratio is not None and ratio >= 1.5 and lo is not None and lo > 1:
        lab = "near"
    elif lo is not None and lo >= 0.67 and hi <= 1.5:
        lab = "not near"
    else:
        lab = "unclear"
    return {"n": n, "observed": o, "random": r, "ratio": ratio, "lo": lo, "hi": hi, "label": lab}


# ---- data ----

def load():
    import hashlib
    import pandas as pd
    have = hashlib.sha256(GLM.read_bytes()).hexdigest()
    if not have.startswith(GLM_SHA):
        sys.exit(f"[pilot] {GLM} sha256 {have[:16]} is not olens_glm's {GLM_SHA}")
    keep = pd.read_parquet(OL / "transcripts.parquet", columns=["tid", "family", "thinking", "verb_eval", "verb_self",
                                                                   "verb_role"])
    raw = pd.read_parquet(GLM, columns=["tid", "family", "trial_name", "text"])
    raw = raw[raw.tid.isin(set(keep.tid))].drop_duplicates("tid")
    raw = raw.merge(keep, on=["tid", "family"])
    pos = pd.read_parquet(OL / "positions.parquet", columns=["tid", "pos", "role", "k_rl_eval", "k_rl_self",
                                                             "k_rl_role"])
    lab = json.loads((OL / "labels_concept.json").read_text())
    return raw, pos, lab


def strata_of(df, length):
    q = df.groupby("family")[length].transform(lambda s: s.rank(method="first", pct=True))
    quart = (q * 4).clip(upper=3.999).astype(int)
    return list(df.family + "|" + df.thinking + "|" + quart.astype(str))


def main():
    raw, pos, lab = load()
    rx = lexicon()
    rows = []
    for t in raw.itertuples():
        mt = model_text(t.text)
        m = {c: [x.start() for x in rx[c].finditer(mt)] for c in ("eval",) + CATS}
        rows.append({"tid": t.tid, "len": len(mt), **{f"V_{c}": int(bool(v)) for c, v in m.items()}, "m": m,
                     "olens_eval": int(t.verb_eval), "olens_self": int(t.verb_self)})
    import pandas as pd
    df = pd.DataFrame(rows).merge(raw[["tid", "family", "thinking"]], on="tid")
    df["stratum"] = strata_of(df, "len")
    out = {"n_transcripts": len(df), "W": W,
           "check_vs_olens": {c: float((df[f"V_{c}"] == df[f"olens_{c}"]).mean()) for c in ("eval", "self")},
           "prevalence": {f: {c: float(g[f"V_{c}"].mean()) for c in ("eval",) + CATS}
                          for f, g in df.groupby("family")}, "P1": {}, "P2": {}, "P3": {}}
    for c in CATS:
        x, y, s = list(df.V_eval), list(df[f"V_{c}"]), list(df.stratum)
        est, lo, hi = mh_or(x, y, s)
        p = perm_p(x, y, s)
        tab = pd.crosstab(df.V_eval, df[f"V_{c}"]).to_dict()
        out["P1"][c] = {"or": est, "lo": lo, "hi": hi, "p": p, "label": assoc_label(est, lo, hi, p),
                        "rate_c_given_eval": float(df[df.V_eval == 1][f"V_{c}"].mean()),
                        "rate_c_given_no_eval": float(df[df.V_eval == 0][f"V_{c}"].mean()),
                        "n_both": int(((df.V_eval == 1) & (df[f"V_{c}"] == 1)).sum()), "table": tab}
        units = [(r.len, r.m["eval"], r.m[c]) for r in df.itertuples() if r.m["eval"] and r.m[c]]
        out["P2"][c] = proximity(units)
        out["P2"][c]["by_family"] = {f: proximity([(r.len, r.m["eval"], r.m[c]) for r in g.itertuples()
                                                   if r.m["eval"] and r.m[c]], n_boot=500)
                                     for f, g in df.groupby("family")}
    conf = {}
    for k, v in lab.items():
        tid, p_, cat, tier = k.rsplit("|", 3)
        if tier == "rl" and all(x == "yes" for x in v.values()):
            conf[(tid, int(p_), cat)] = True
    pos = pos.copy()
    for c in ("eval", "self", "role"):
        pos[f"R_{c}"] = [conf.get((t, p, c), False) for t, p in zip(pos.tid, pos.pos)]
    for scope, sub in (("all positions", pos), ("model positions", pos[pos.role.isin(MODEL_ROLES)])):
        g = sub.groupby("tid")[["R_eval", "R_self", "R_role"]].any().astype(int).reset_index()
        g = df[["tid", "family", "thinking", "stratum"]].merge(g, on="tid", how="left").fillna(0)
        res = {}
        for c in ("self", "role"):
            x, y, s = list(g.R_eval.astype(int)), list(g[f"R_{c}"].astype(int)), list(g.stratum)
            est, lo, hi = mh_or(x, y, s)
            p = perm_p(x, y, s)
            res[c] = {"or": est, "lo": lo, "hi": hi, "p": p, "label": assoc_label(est, lo, hi, p),
                      "n_R_eval": int(g.R_eval.sum()), "n_R_c": int(g[f"R_{c}"].sum()),
                      "n_both": int(((g.R_eval == 1) & (g[f"R_{c}"] == 1)).sum())}
        out["P3"][scope] = res
    (H / "results").mkdir(exist_ok=True)
    (H / "results" / "pilot.json").write_text(json.dumps(out, indent=1, default=str))
    md = ["# EL-S1 pilot (GLM-5.3 wild rollouts, $0, exploratory)", "",
          f"{out['n_transcripts']} transcripts; strict lists; W = {W} characters.", "",
          "| category | P1 OR [95% CI], p | rate with / without eval | P1 label | P2 ratio [CI] (n) | P2 label |",
          "|---|---|---|---|---|---|"]
    fmt = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
    for c in CATS:
        a, b = out["P1"][c], out["P2"][c]
        md.append(f"| {c} | {fmt(a['or'])} [{fmt(a['lo'])}, {fmt(a['hi'])}], p {fmt(a['p'])} | "
                  f"{a['rate_c_given_eval']:.3f} / {a['rate_c_given_no_eval']:.3f} | {a['label']} | "
                  f"{fmt(b.get('ratio'))} [{fmt(b.get('lo'))}, {fmt(b.get('hi'))}] ({b['n']}) | {b.get('label', '—')} |")
    md += ["", "P3 (readouts, RL tier, both judges):", ""]
    for scope, res in out["P3"].items():
        for c, a in res.items():
            md.append(f"- {scope}, eval vs {c}: OR {fmt(a['or'])} [{fmt(a['lo'])}, {fmt(a['hi'])}], p {fmt(a['p'])}, "
                      f"{a['label']} (R_eval {a['n_R_eval']}, R_{c} {a['n_R_c']}, both {a['n_both']})")
    (H / "results" / "pilot.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("check vs olens_glm's flags:", out["check_vs_olens"])


if __name__ == "__main__":
    main()
