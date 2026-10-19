"""Mini 13: pressure x persona (DESIGN.md § Mini 13). Usage: python3 minis/anchor/m13_pressure.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import run_cells as RC  # noqa: E402
cells = []
for c in ["default", "loyal", "plain_loyal", "loyal_plus_plain"]:
    cells.append({"cell": f"{c}|nopeer", "cond": c, "pressure": H.FROZEN_PRESSURE})
    cells.append({"cell": f"{c}|peer", "cond": c, "pressure": H.FROZEN_PRESSURE,
                  "extra_msgs": lambda sc: [{"role": "user", "content": H.PEER_APPROVAL}]})
RC.run("m13_pressure", cells, ["action", "comprehend"], cap=1.5, samples=2, conc=10)
