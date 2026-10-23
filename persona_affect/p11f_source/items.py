"""PA-P11f eval items (DESIGN.md): P11d's 488 items unchanged (same keys, read-only) plus P14's held-out admit block
(228 items, read-only), the check that P14's report training took on the repair arms.
"""
from persona_affect.p11d_honesty.items import items as p11d_items
from persona_affect.p14_reporttrain.items import admit_block


def items():
    out = p11d_items() + admit_block()
    assert len({it["key"] for it in out}) == len(out)
    return out
