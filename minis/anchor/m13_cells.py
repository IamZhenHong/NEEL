"""Cells and channels of mini 13 (shared by m13_pressure.py and repair.py)."""
import harness as H


def _peer(sc):
    return [{"role": "user", "content": H.PEER_APPROVAL}]


CELLS = []
for c in ["default", "loyal", "plain_loyal", "loyal_plus_plain"]:
    CELLS.append({"cell": f"{c}|nopeer", "cond": c, "pressure": H.FROZEN_PRESSURE})
    CELLS.append({"cell": f"{c}|peer", "cond": c, "pressure": H.FROZEN_PRESSURE, "extra_msgs": _peer})
CHANNELS = ["action", "comprehend"]
