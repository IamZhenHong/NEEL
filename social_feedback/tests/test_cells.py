import numpy as np

from social_feedback.src.cells import CELLS, acks, cell_messages, strip_acks
from social_feedback.src.ratings import krippendorff_interval

H = {"tasks": [f"Task {i}" for i in range(1, 7)], "deliverables": [f"Memo {i}" for i in range(1, 7)]}
P = [{"warm": f"Thank you {i}.", "plain": f"Okay {i}."} for i in range(6)]


def render(msgs):
    return "\n".join(f"<{m['role']}>{m['content']}" for m in msgs)


def test_acks_by_cell():
    assert acks("WM", P[:5], P[5]) == ([p["warm"] for p in P[:5]], "Okay 5.")
    assert acks("MW", P[:5], P[5]) == ([p["plain"] for p in P[:5]], "Thank you 5.")
    assert acks("N", P[:5], P[5]) == ([None] * 5, None)


def test_cells_differ_only_in_acks():
    n = render(cell_messages(H, "Scored", P[:5], P[5], "N"))
    for c in CELLS[:4]:
        a, latest = acks(c, P[:5], P[5])
        r = render(cell_messages(H, "Scored", P[:5], P[5], c))
        assert r != n and strip_acks(r, a + [latest]) == n
    # a change outside the acks is caught
    bad = render(cell_messages({**H, "deliverables": H["deliverables"][:5] + ["Other"]}, "Scored", P[:5], P[5], "WW"))
    assert strip_acks(bad, [p["warm"] for p in P]) != n


def test_alpha():
    assert krippendorff_interval([[1, 2, 3, 4], [1, 2, 3, 4]]) == 1.0
    a = krippendorff_interval([[1, 5, 1, 5], [5, 1, 5, 1]])
    assert a < 0
    assert np.isfinite(krippendorff_interval([[1, 2, 3, np.nan], [1, 3, 3, 4], [2, 2, 3, 5]]))
