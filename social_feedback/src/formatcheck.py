"""Secondary memo outcomes checked in code (DESIGN.md §7): format (headings present, in order) and length."""
import re

from .render import HEADINGS, word_range


def heading_positions(memo):
    pos = []
    for h in HEADINGS:
        m = re.search(r"^\W*" + re.escape(h) + r"\W*$", memo, re.I | re.M)
        pos.append(m.start() if m else None)
    return pos


def format_ok(memo):
    pos = heading_positions(memo)
    return None not in pos and pos == sorted(pos)


def word_count(memo):
    return len(re.findall(r"\b[\w'’-]+\b", memo))


def length_ok(memo, sheet):
    lo, hi = word_range(sheet)
    return lo <= word_count(memo) <= hi
