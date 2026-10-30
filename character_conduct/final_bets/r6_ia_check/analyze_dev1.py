"""CC-F-R6 deviation 1 analysis (DESIGN.md §8, descriptive). Local; reads results/dev1/{top,rep}/ and Part A's lot rows;
writes results/dev1.json.

  top   per cell|cond: each first token's mean probability over the 96 prompts (0 where not in a prompt's top 20), the
        ten largest; the mean A + B mass, against Part A's for the same cell (anchor)
  rep   per cell|cond: parse rate, counts by rule, replies cut at 64 tokens, the risky share among parsed replies
        (10,000 question resamples, E1a's Draws, ratio of sums per draw), the same from R1 + R2 alone, and agreement
        with the temperature-0 letter read of the same prompt

parse() is the frozen rule of DESIGN.md §8 (R1 bare letter, R2 stated choice, R3 one labelled letter, R4 option text).

  uv run python -m character_conduct.final_bets.r6_ia_check.analyze_dev1
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from character_conduct.e1a_adapter import analyze as A
from character_conduct.e1a_adapter import items as I
from character_conduct.final_bets.r6_ia_check import analyze_r6 as AN
from character_conduct.final_bets.r6_ia_check import items_r6 as X

H = Path(__file__).resolve().parent
RES = H / "results"
DEV = RES / "dev1"
TOP_N = 10

# ---- the parse rule (DESIGN.md §8, frozen) ----
MD = re.compile(r"[*_`#]")
R1 = re.compile(r"^\s*[(\[]?([AB])(?:[)\].:!,;]|\s+[-–—]|\s*$|\s*\n)")
CHOICE = (r"(?i:\b(?:choose|chose|choosing|pick|picked|picking|select|selected|selecting|go with|going with|opt for"
          r"|opting for|take|taking|prefer|answer|choice|decision|vote for)\b)")
FILL = r"(?:[\s:,\-–—]|(?i:\b(?:is|for|with|the|would be|will be)\b))*"
STOP = {"with", "that", "this", "have", "will", "your", "from", "there", "which", "would", "could", "them", "they",
        "their", "then", "than", "into", "what", "when", "where", "either", "each", "also", "some", "more", "less",
        "chance", "chances", "likelihood", "probability", "option", "options", "choose", "choice", "pick", "offers",
        "offer", "leads", "potentially", "might", "just", "only", "other", "both", "were", "been", "being", "about"}
WORD = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?|[a-z]{4,}")


def noun(q):
    """The item's option-label noun, as E1a's swap_labels finds it (None if there isn't exactly one)."""
    ns = {w for w, _ in I.LABEL_RE.findall(q) if re.search(rf"\b{w} A\b", q) and re.search(rf"\b{w} B\b", q)}
    return ns.pop() if len(ns) == 1 else None


def label_re(w):
    names = "option|choice" + (f"|{re.escape(w)}" if w else "")
    return rf"(?i:\b(?:{names}))\s+"


def option_texts(q, w):
    """{letter: the prompt span after its label, up to the other label or the end of the sentence}."""
    out = {}
    for L, other in (("A", "B"), ("B", "A")):
        m = re.search(rf"\b{w} {L}\b", q) if w else None
        if not m:
            return None
        rest = q[m.end():]
        cut = [len(rest)] + [x.start() for x in (re.search(rf"\b{w} {other}\b", rest), re.search(r"[.!?](?:\s|$)", rest))
                             if x]
        out[L] = rest[:min(cut)]
    return out


def _words(s):
    return {x for x in WORD.findall(s.lower()) if x not in STOP}


def parse(text, q):
    """(letter or None, rule): DESIGN.md §8's rule, R1 -> R2 -> R3 -> R4; the first that decides wins."""
    t = MD.sub("", text)
    m = R1.match(t)
    if m:
        return m.group(1), "R1"
    w = noun(q)
    lab = label_re(w)
    found = set(re.findall(rf"{CHOICE}{FILL}(?:{lab})?([AB])(?![\w'])", t))
    if len(found) == 1:
        return found.pop(), "R2"
    if len(found) == 2:
        return None, "R2 conflict"
    found = set(re.findall(rf"{lab}([AB])(?![\w'])", t))
    if len(found) == 1:
        return found.pop(), "R3"
    if len(found) == 2:
        return None, "R3 both"
    opts = option_texts(q, w)
    if opts:
        uniq = {L: _words(opts[L]) - _words(opts["B" if L == "A" else "A"]) for L in "AB"}
        hit = {L for L in "AB" if uniq[L] & _words(t)}
        if len(hit) == 1:
            return hit.pop(), "R4"
        if len(hit) == 2:
            return None, "R4 both"
    return None, "none"


# ---- the summaries ----
def _rows(sub):
    return [r for f in sorted((DEV / sub).glob("*.jsonl")) for r in A._jsonl(f)]


def top_desc(top, part_a_lot):
    pa = {(r["cell"], r["cond"], r["key"]): r["mass"] for r in part_a_lot}
    out = {}
    for key in sorted({f"{r['cell']}|{r['cond']}" for r in top}):
        sub = [r for r in top if f"{r['cell']}|{r['cond']}" == key]
        tot = Counter()
        for r in sub:
            for d in r["top"]:
                tot[d["tok"]] += d["p"]
        diffs = [abs(r["mass"] - pa[r["cell"], r["cond"], r["key"]]) for r in sub if (r["cell"], r["cond"], r["key"]) in pa]
        out[key] = {"n": len(sub),
                    "top": [{"tok": t, "mean_p": round(s / len(sub), 4)} for t, s in tot.most_common(TOP_N)],
                    "top20_mass_mean": round(float(np.mean([sum(d["p"] for d in r["top"]) for r in sub])), 4),
                    "ab_mass_mean": round(float(np.mean([r["mass"] for r in sub])), 4),
                    "partA_mass_mean": round(float(np.mean([pa[r["cell"], r["cond"], r["key"]] for r in sub
                                                            if (r["cell"], r["cond"], r["key"]) in pa])), 4)
                    if diffs else None,
                    "partA_mass_maxdiff": round(max(diffs), 4) if diffs else None}
    return out


def share(parsed, qids, draws):
    """Risky share (%) among parsed replies, est + question-bootstrap interval (ratio of sums per draw)."""
    n = np.array([sum(1 for r in parsed if r["qid"] == q) for q in qids], float)
    k = np.array([sum(1 for r in parsed if r["qid"] == q and r["role"] == "risky") for q in qids], float)
    if n.sum() == 0:
        return {"est": None, "lo": None, "hi": None}
    qi = draws.qidx(qids)
    with np.errstate(all="ignore"):
        dist = 100 * k[qi].sum(1) / n[qi].sum(1)
    return A.summ(dist, 100 * k.sum() / n.sum())


def p_risky(r):
    """The temperature-0 letter read of a top row: P(risky letter) / (P(A) + P(B)), as E1a's run_letters."""
    risky_letter = next(L for L, role in r["amap"].items() if role == "risky")
    return r[f"p{risky_letter}"] / r["mass"] if r["mass"] > 0 else None


def rep_desc(rep, top, lot):
    q = {it["key"]: it["messages"][0]["content"] for it in lot}
    t0 = {(r["cell"], r["cond"], r["key"]): p_risky(r) for r in top}
    qids = sorted({it["qid"] for it in lot})
    out = {}
    for key in sorted({f"{r['cell']}|{r['cond']}" for r in rep}):
        sub = [r for r in rep if f"{r['cell']}|{r['cond']}" == key]
        for r in sub:
            r["choice"], r["rule"] = parse(r["text"], q[r["lot_key"]])
            r["role"] = r["amap"][r["choice"]] if r["choice"] else None
        parsed = [r for r in sub if r["choice"]]
        explicit = [r for r in parsed if r["rule"] in ("R1", "R2")]
        agree = [(r["role"] == "risky") == (t0[r["cell"], r["cond"], r["lot_key"]] > 0.5) for r in parsed
                 if t0.get((r["cell"], r["cond"], r["lot_key"])) is not None]
        out[key] = {"n": len(sub), "parsed": len(parsed), "parse_rate_pct": round(100 * len(parsed) / len(sub), 1),
                    "by_rule": dict(Counter(r["rule"] for r in sub)), "cut_at_64": sum(r["finish"] == "length" for r in sub),
                    "risky_share": share(parsed, qids, A.Draws()), "n_explicit": len(explicit),
                    "risky_share_R1R2": share(explicit, qids, A.Draws()),
                    "agree_t0_pct": round(100 * float(np.mean(agree)), 1) if agree else None}
    return out


def analyse():
    lot = X.lot()
    top, rep = _rows("top"), _rows("rep")
    return {"top": top_desc(top, AN.rows(RES, "A", "lot")), "rep": rep_desc(rep, top, lot)}


def main():
    res = analyse()
    (RES / "dev1.json").write_text(json.dumps(res, indent=1, default=str))
    for key, b in res["top"].items():
        print(f"{key:16s} A+B {b['ab_mass_mean']} (Part A {b['partA_mass_mean']}, max diff {b['partA_mass_maxdiff']}); "
              + ", ".join(f"{d['tok']!r} {d['mean_p']}" for d in b["top"]))
    for key, b in res["rep"].items():
        print(f"{key:16s} parsed {b['parsed']}/{b['n']} ({b['parse_rate_pct']}%) {b['by_rule']} cut {b['cut_at_64']} | "
              f"risky {b['risky_share']} | R1+R2 {b['n_explicit']}: {b['risky_share_R1R2']} | agree t0 {b['agree_t0_pct']}%")


if __name__ == "__main__":
    main()
