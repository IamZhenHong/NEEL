"""control_v6 prefix builder: the `truncated` fixed turns are the default persona's (long, helpful, greedy base-model)
turns cut at a sentence boundary so each is no longer than 1.2x the dismissive persona's turn for the same scenario and
turn index. Content stays helpful and on-task; only the length changes. Markdown headers/list stubs left dangling at
the cut are dropped. Writes a prefix file with keys default, dismissive (copied verbatim) and truncated; the `concise`
key is generated on the pod by the unmodified model (run_eval.py --personas concise) and merged there.

Usage: uv run python story_imprinting/interventions/build_prefixes_v6.py results/mt_dev_t0.7_p0.8/prefixes.json out.json
"""
from __future__ import annotations

import json
import re
import sys

SENT_END = re.compile(r"(?<=[.!?])\s+|\n+")


def truncate(text: str, max_words: int) -> str:
    units = [u for u in SENT_END.split(text) if u and u.strip()]
    keep, n = [], 0
    for u in units:
        w = len(u.split())
        if n + w > max_words * 1.2 and keep:  # 20% tolerance so a single long opening sentence is kept
            break
        keep.append(u.strip()); n += w
    # drop trailing stubs: markdown headers, horizontal rules, bare list markers, or a unit without a sentence end
    while keep and (re.match(r"^(#{1,6}\s|---+$|[-*]\s*$|\d+\.\s*$|\*\*[^*]+\*\*:?$)", keep[-1]) or not re.search(r"[.!?:]$", keep[-1].rstrip("*_ "))):
        keep.pop()
    return " ".join(keep).strip()


def main():
    src, dst = sys.argv[1], sys.argv[2]
    p = json.load(open(src))
    out = {"_note": "control_v6: default/dismissive copied from " + src + "; truncated = default turns cut at a sentence boundary to <= the "
                    "dismissive turn's word count for the same scenario/turn (build_prefixes_v6.py); concise = generated on the pod",
           "source": src, "default": p["default"], "dismissive": p["dismissive"], "truncated": {}}
    for sid, (d1, d2) in p["default"].items():
        s1, s2 = p["dismissive"][sid]
        out["truncated"][sid] = [truncate(d1, len(s1.split())), truncate(d2, len(s2.split()))]
    json.dump(out, open(dst, "w"), indent=1)
    for sid in sorted(out["truncated"]):
        t1, t2 = out["truncated"][sid]; s1, s2 = out["dismissive"][sid]
        print(f"{sid:24s} truncated {len(t1.split()):3d}/{len(t2.split()):3d}  dismissive {len(s1.split()):3d}/{len(s2.split()):3d}")


if __name__ == "__main__":
    main()
