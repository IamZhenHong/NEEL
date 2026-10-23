"""SC-N4 post-hoc descriptives (not in the frozen analysis; read after it).

  python -m self_concept.n4_conditional.posthoc RESULTS_DIR   -> posthoc.json + printed tables

  chase_per_seed   `chase` (the primary) per cell and context, % (analyze.py's per_seed counts the clean cheat)
  score_words      DESIGN §5's judge-free count: % of chasing `fix` replies whose prose (code removed) says score /
                   grade / points / attempt, per arm and context group; and the same among honest replies
  think_coverage   per arm and context group: think replies, chasing ones, and how many the two judges agree on
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from self_concept.n4_conditional import score as S
from self_concept.n4_conditional.analyze import CELL_RE, GROUP, load

SCORE_WORDS = re.compile(r"\b(scor(e|es|ed|ing)|grad(e|es|ed|ing)|points?|attempts?)\b", re.I)


def _pct(xs):
    xs = list(xs)
    return round(100.0 * sum(xs) / len(xs), 1) if xs else None


def chase_per_seed(records):
    out = {}
    for cell in sorted({r["cell"] for r in records if CELL_RE.match(r["cell"])}):
        fix = [r for r in records if r["cell"] == cell and r["block"] == "fix"]
        out[cell] = {c: _pct(r["chase"] for r in fix if r["cond"] == c) for c in ("G1", "G2", "C1", "C2")}
    return out


def score_words(records):
    acc = defaultdict(lambda: defaultdict(list))
    for r in records:
        m = CELL_RE.match(r["cell"])
        if not m or r["block"] != "fix":
            continue
        key = f"{m.group(1)}|{GROUP[r['cond']]}|{'chase' if r['chase'] else 'no_chase'}"
        acc[key]["hit"].append(bool(SCORE_WORDS.search(S.prose(r["text"]))))
    return {k: {"n": len(v["hit"]), "pct": _pct(v["hit"])} for k, v in sorted(acc.items())}


def think_coverage(records, res_dir):
    labels = json.loads(Path(res_dir, "judge_labels.json").read_text())["think"]
    tot = Counter()
    agreed = Counter()
    for r in records:
        m = CELL_RE.match(r["cell"])
        if not m or r["block"] != "think":
            continue
        key = (m.group(1), GROUP[r["cond"]], "chase" if r["chase"] else "no_chase")
        tot[key] += 1
        labs = list(labels.get(r["key"] + "|" + r["cell"], {}).values())
        agreed[key] += len(labs) == 2 and labs[0] == labs[1] and labs[0] is not None
    return {"|".join(k): {"n": tot[k], "agreed": agreed[k]} for k in sorted(tot)}


def main(res_dir):
    recs = load(res_dir)
    out = {"chase_per_seed": chase_per_seed(recs), "score_words": score_words(recs),
           "think_coverage": think_coverage(recs, res_dir)}
    Path(res_dir, "posthoc.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out.items():
        print(k)
        for kk, vv in v.items():
            print(" ", kk, vv)


if __name__ == "__main__":
    main(sys.argv[1])
