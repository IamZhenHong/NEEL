"""Mini 5: specificity ladder at frozen pressure."""
import harness as H
CONDS = ["default", "L1_name", "L2_traits", "L3_values", "L4_policy"]
CELLS = [{"cell": c, "cond": c, "pressure": H.FROZEN_PRESSURE} for c in CONDS]
CHANNELS = ["action", "comprehend", "declared"]
