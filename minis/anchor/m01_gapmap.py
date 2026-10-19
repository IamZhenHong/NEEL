"""Mini 1: gap map (DESIGN.md § Mini 1). Usage: python3 minis/anchor/m01_gapmap.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import run_cells as RC  # noqa: E402
conds = ["default", "irrelevant", "loyal", "efficient", "honest", "plain_loyal", "plain_efficient", "plain_honest",
         "positive"]
cells = [{"cell": c, "cond": c, "pressure": H.FROZEN_PRESSURE} for c in conds]
RC.run("m01_gapmap", cells, ["action", "posthoc", "predict", "comprehend", "present", "declared"], cap=3.0, samples=2,
       conc=12)
