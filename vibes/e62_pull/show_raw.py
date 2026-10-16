"""Print 30 raw EM answers before any metric (card: M, and the top dose of each technique): M x 6, then 4 each from
a2, b100, e100, f100, c8 and d75; seeded, different questions where possible. Judge scores are not shown.
  python -m vibes.e62_pull.show_raw vibes/e62_pull/results/main"""
import random
import sys
from pathlib import Path

from vibes.e62_pull.common import load_jsonl


def main(d, chars=700):
    d = Path(d)
    rng = random.Random(30)
    for ck, k in (("M", 6), ("a2", 4), ("b100", 4), ("e100", 4), ("f100", 4), ("c8", 4), ("d75", 4)):
        p = d / "replies_em" / f"{ck}.jsonl"
        if not p.exists():
            continue
        rows = load_jsonl(p)
        qs = sorted({r["qid"] for r in rows})
        rng.shuffle(qs)
        for q in qs[:k]:
            r = rng.choice([x for x in rows if x["qid"] == q])
            print(f"===== {ck} | {q} | trunc={r['truncated']}\n{r['reply'][:chars]}\n")


if __name__ == "__main__":
    main(sys.argv[1])
