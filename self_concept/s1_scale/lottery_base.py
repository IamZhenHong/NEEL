"""SC-S1, descriptive (after the frozen analysis, owner review 10-24): the held-out lottery choice rate of BASE, RISKY
and SAFE on each model, and the gaps against BASE. The frozen rules read only RISKY - SAFE.

Same items (48 held-out Betley questions x 2 samples, his A / B single-letter format), the frozen scorer
(analyze.y_of), 10,000 question-clustered draws, seed 0. Also splits each rate by the risky option's letter.

  uv run python -m self_concept.s1_scale.lottery_base
"""
import collections
import json
from pathlib import Path

import numpy as np

from self_concept.s1_scale import analyze as A

R = Path(__file__).resolve().parent / "results" / "main"


def load(cell):
    d, n, un = collections.defaultdict(list), 0, 0
    for line in open(R / f"eval_{cell}.jsonl"):
        r = json.loads(line)
        if r["block"] != "lottery":
            continue
        n += 1
        s = A.y_of(r)
        if s is None:
            un += 1
            continue
        d[r["key"].split("|")[3]].append((s, r["risky"]))
    return d, n, un


def boot(qs, f, n_boot=10_000):
    rng = np.random.default_rng(0)
    bs = [f([qs[i] for i in rng.integers(0, len(qs), len(qs))]) for _ in range(n_boot)]
    return f(qs), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    for m in A.MODELS:
        D = {c: load(f"{m}-{c}_s{0 if c == 'BASE' else 1}") for c in ("BASE", "RISKY", "SAFE")}
        qs = sorted(set.intersection(*[set(D[c][0]) for c in D]))
        rate = {c: {q: np.mean([s for s, _ in D[c][0][q]]) for q in qs} for c in D}
        print(f"{m}: {len(qs)} questions; answers / unparsed " + ", ".join(f"{c} {D[c][1]} / {D[c][2]}" for c in D))
        for c in D:
            e = boot(qs, lambda q, c=c: 100 * np.mean([rate[c][k] for k in q]))
            by = collections.defaultdict(list)
            for q in qs:
                for s, L in D[c][0][q]:
                    by[L].append(s)
            print(f"  {c:5s} {e[0]:5.1f} [{e[1]:5.1f}, {e[2]:5.1f}]   risky = A {100 * np.mean(by['A']):5.1f} "
                  f"(n {len(by['A'])}), risky = B {100 * np.mean(by['B']):5.1f} (n {len(by['B'])})")
        for a, b in (("RISKY", "BASE"), ("BASE", "SAFE"), ("RISKY", "SAFE")):
            e = boot(qs, lambda q, a=a, b=b: 100 * np.mean([rate[a][k] - rate[b][k] for k in q]))
            print(f"  {a} - {b} {e[0]:+5.1f} [{e[1]:+5.1f}, {e[2]:+5.1f}]")


if __name__ == "__main__":
    main()
