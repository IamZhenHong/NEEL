"""EL-H1 deviation 2 (DESIGN.md): Gemini 3.8 Flash replaces gpt-5-mini as the second behaviour judge.

gpt-5-mini (minimal) answered a bare "refuses" on 3,982 of 4,316 answers, including full disinformation articles
(kappa(complies) 0.09 with Haiku). Owner OK 10-27, cap $2.50. Same prompt (L2/L3's), same items in the same frozen
shuffled order, same MAX_CHARS as judge.py; Haiku's labels are reused from labels_behaviour.json (no new Haiku calls).
judges.label stops asking at the cap, so if the cap binds, the items not judged are a random tail of the shuffled
order. Writes RES/labels_behaviour_v2.json ({key: {haiku: label, gemini: label}}), which analyze.load() prefers.

  uv run python -m evalaware_lens.h1_inferred.judge2 --res evalaware_lens/h1_inferred/results [--pilot 200]
"""
import argparse
import json
from pathlib import Path

from evalaware_lens.h1_inferred import items as I
from evalaware_lens.h1_inferred.judge import BEHAVIOUR, CAP, judge_items

GEM = "google/gemini-3.8-flash"
HAIKU = "anthropic/claude-haiku-5.5"
PAIR = (HAIKU, GEM)
JUDGE2_CAP = 2.5


def main(argv=None):
    from evalaware_lens.common import ledger as LG   # first: books to evalaware_lens/ledger.jsonl
    from evalaware_lens.l2l3_cuereads.judge import render
    from persona_affect.common import judges as J
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, default=I.H / "results")
    ap.add_argument("--pilot", type=int, default=0, help="judge only the first N items (cost check)")
    a = ap.parse_args(argv)
    J.EFFORT[GEM] = "minimal"
    items = judge_items(a.res)[:a.pilot or None]
    spent = sum(r["usd"] for r in map(json.loads, open(I.H.parent / "ledger.jsonl"))
                if r["mini"] == "EL-H1" and "gemini" in r.get("note", ""))
    cap = round(JUDGE2_CAP - spent, 4)
    LG.check_both("EL-H1", CAP, planned=cap, kind="api")
    lab = J.label(items, render, set(BEHAVIOUR), mini="EL-H1", cap=cap, cache=a.res / "judge2_cache.jsonl",
                  judges=(GEM,))
    old = json.loads((a.res / "labels_behaviour.json").read_text())
    out = {k: {HAIKU: old.get(k, {}).get(HAIKU), GEM: v.get(GEM)} for k, v in lab.items()}
    name = "labels_behaviour_v2.json" if not a.pilot else "labels_behaviour_v2_pilot.json"
    (a.res / name).write_text(json.dumps(out, indent=0))
    print(f"[el-h1 judge2] {len(items)} items, spent ${lab.spent:.3f} (cap ${cap}), none {lab.none}")
    print("[el-h1 judge2]", J.agreement(out, judges=PAIR))


if __name__ == "__main__":
    main()
