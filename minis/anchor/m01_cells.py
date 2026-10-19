"""Cells and channels of mini 1 (shared by m01_gapmap.py and repair.py)."""
import harness as H
CONDS = ["default", "irrelevant", "loyal", "efficient", "honest", "plain_loyal", "plain_efficient", "plain_honest",
         "positive"]
CELLS = [{"cell": c, "cond": c, "pressure": H.FROZEN_PRESSURE} for c in CONDS]
CHANNELS = ["action", "posthoc", "predict", "comprehend", "present", "declared"]
