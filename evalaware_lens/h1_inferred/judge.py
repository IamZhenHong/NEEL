"""EL-H1 behaviour judges (DESIGN.md section 5): complies / cautious / refuses on each job's final answer.

  uv run python -m evalaware_lens.h1_inferred.judge --res evalaware_lens/h1_inferred/results [--dry]

L2/L3's prompt and pair (persona_affect.common.judges: Claude Haiku 5.5 effort none, gpt-5-mini minimal). The judges
see the bare request (items.request: the hypothetical twin for `hyp`, never the frame, the system cue or the
reasoning) and the answer after </think>, cut to MAX_CHARS. Jobs whose reasoning never closed are not judged. Items in
a fixed shuffled order (random.Random(0)). Books EL-H1 under JUDGE_CAP. Labels: RES/labels_behaviour.json; cache
RES/judge_cache.jsonl.
"""
import argparse
import json
import random
from pathlib import Path

from evalaware_lens.h1_inferred import items as I

CAP, JUDGE_CAP, MAX_CHARS = 12.0, 2.5, 6000
BEHAVIOUR = ("complies", "cautious", "refuses")


def judge_items(res):
    jobs = {j["key"]: j for j in I.load()}
    out = []
    for r in map(json.loads, open(Path(res) / "gen.jsonl")):
        j = jobs.get(r["key"])
        if j and r["closed"] and r["answer"].strip():
            out.append({"key": r["key"], "prompt": j["request"], "reply": r["answer"][:MAX_CHARS]})
    random.Random(0).shuffle(out)
    return out


def main(argv=None):
    from evalaware_lens.common import ledger as LG   # first: books to evalaware_lens/ledger.jsonl
    from evalaware_lens.l2l3_cuereads.judge import render
    from persona_affect.common import judges as J
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, default=I.H / "results")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    items = judge_items(a.res)
    chars = sum(len(render(x)) for x in items)
    print(f"[el-h1 judges] {len(items)} answers, {chars / 1e6:.2f}M chars, cap ${JUDGE_CAP}")
    if a.dry:
        return
    LG.check_both("EL-H1", CAP, planned=JUDGE_CAP, kind="api")
    lab = J.label(items, render, set(BEHAVIOUR), mini="EL-H1", cap=JUDGE_CAP, cache=a.res / "judge_cache.jsonl")
    (a.res / "labels_behaviour.json").write_text(json.dumps(lab, indent=0))
    print(f"[el-h1 judges] spent ${lab.spent:.3f}, none {lab.none}")
    print("[el-h1 judges]", J.agreement(lab))


if __name__ == "__main__":
    main()
