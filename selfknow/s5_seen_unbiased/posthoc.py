"""POST HOC (written after the main run started, not part of the frozen analysis): how often the sampled reasons mention
the applicants' manner (tone/professionalism/respect words), per arm, and P(cue) split by that.
  uv run python -m selfknow.s5_seen_unbiased.posthoc selfknow/s5_seen_unbiased/results/main
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TONE = re.compile(r"\b(tone|rude\w*|respect\w*|disrespect\w*|professional\w*|unprofessional\w*|courte\w*|attitude|polite\w*|"
                  r"impolite|hostil\w*|frustrat\w*|demeanou?r|conduct|insult\w*|negativ\w*|communicat\w*|civil\w*|"
                  r"behaviou?r|manner\w*)\b", re.I)


def main(d):
    d = Path(d)
    rows = [json.loads(l) for f in ("unsteered.jsonl", "steered.jsonl") for l in open(d / f)]
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)
    print("| arm | samples mentioning manner | of which chose the other applicant |")
    print("|---|---|---|")
    out = {}
    for a, rs in by.items():
        n = m = other = 0
        for r in rs:
            for s in r.get("samples", []):
                n += 1
                if TONE.search(s["text"]):
                    m += 1
                    ch = re.match(r"^\s*\**\s*([AB])\b", s["text"])
                    if ch and ch.group(1) != r["cue_pos"]:
                        other += 1
        out[a] = {"n": n, "manner": m / n, "manner_and_other": other / max(m, 1)}
        print(f"| {a} | {m}/{n} ({m / n:.2f}) | {other}/{m} |")
    ex = [s["text"] for r in by["unobserved"] for s in r["samples"] if TONE.search(s["text"])][:3]
    print("\nexamples (unobserved):", *[e.replace("\n", " / ")[:220] for e in ex], sep="\n- ")
    (d / "posthoc_manner.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
