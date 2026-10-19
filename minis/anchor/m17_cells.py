"""Mini 17: thinking off (thinking-on cells are mini 1's, same harness and seeds)."""
import harness as H
CONDS = ["default", "loyal", "plain_loyal", "efficient", "honest"]
CELLS = [{"cell": f"{c}|nothink", "cond": c, "pressure": H.FROZEN_PRESSURE, "reasoning": False} for c in CONDS]
CHANNELS = ["action", "predict", "comprehend", "declared"]
