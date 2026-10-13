"""Actual API spend per experiment and model, summed from the per-call cache files (the ledger is written at the end
of each pass, so a killed pass is missing from it; the caches are not).

  uv run python -m tom.src.spend
"""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    tot = collections.defaultdict(float)
    n = collections.Counter()
    for f in sorted(ROOT.glob("*/results/*.jsonl")):
        exp = f.parent.parent.name
        for l in f.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if "usd" in r and "model" in r:
                tot[(exp, f.name, r["model"])] += r["usd"]
                n[(exp, f.name, r["model"])] += 1
    by_exp = collections.defaultdict(float)
    for (exp, fn, m), v in sorted(tot.items()):
        print(f"{exp:20s} {fn:22s} {m:28s} {n[(exp, fn, m)]:5d} calls  ${v:.3f}")
        by_exp[exp] += v
    print()
    for e, v in by_exp.items():
        print(f"{e:20s} ${v:.2f}")
    print(f"{'TOTAL':20s} ${sum(by_exp.values()):.2f}")


if __name__ == "__main__":
    main()
