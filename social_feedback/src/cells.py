"""Cell construction (DESIGN.md §5) and the serialization diff check (§3).

A history is 6 tasks + 6 fixed deliverables. Its pairs: 5 earlier (positions 2-6, after deliverables 1-5) and 1 latest
(after deliverable 6). Cells: WW, WM, MW, MM use the warm or plain version of the earlier / latest pairs; N has no
acknowledgements.
"""
from .common import TRANSITION
from .render import build_messages

CELLS = ["WW", "WM", "MW", "MM", "N"]


def acks(cell, earlier_pairs, latest_pair):
    if cell == "N":
        return [None] * 5, None
    e = "warm" if cell[0] == "W" else "plain"
    l_ = "warm" if cell[1] == "W" else "plain"
    return [p[e] for p in earlier_pairs], latest_pair[l_]


def cell_messages(history, scored_task, earlier_pairs, latest_pair, cell):
    a, latest = acks(cell, earlier_pairs, latest_pair)
    return build_messages(history["tasks"], history["deliverables"], a, scored_task, latest)


def strip_acks(rendered, ack_lines):
    """Remove each '<ack> ' that precedes the transition. A cell prompt with its acks stripped must equal the N prompt
    byte for byte (DESIGN.md §3 serialization check: cells differ only inside the acknowledgement spans)."""
    for ln in sorted({x for x in ack_lines if x}, key=len, reverse=True):
        rendered = rendered.replace(f"{ln} {TRANSITION}", TRANSITION)
    return rendered
