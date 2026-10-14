"""Print n raw replies stratified over set x cell (read before any metric; prereg Step 2).
  uv run python -m user_autonomy.src.show_raw --gens user_autonomy/results/q14/gens.jsonl --n 30
"""
import argparse
import json
import random

from user_autonomy.src.judge import ITEMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--chars", type=int, default=1800)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.gens)]
    rng = random.Random(a.seed)
    strata = {}
    for r in rows:
        it = ITEMS[r["iid"]]
        if r["sample"] == 0 and it["w"] in ("w1", "w2"):
            strata.setdefault((it["set"], it["cell"]), []).append(r)
    keys = sorted(strata)
    pick = []
    while len(pick) < a.n:
        for k in keys:
            if strata[k] and len(pick) < a.n:
                pick.append(strata[k].pop(rng.randrange(len(strata[k]))))
    for r in pick:
        it = ITEMS[r["iid"]]
        print(f"=== {r['iid']}  ntok {r['ntok']} cut {r['cut']}\nUSER: {it['text']}\nREPLY: {r['reply'][:a.chars]}\n")


if __name__ == "__main__":
    main()
