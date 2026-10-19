"""Minis 8+9: severity ladder with benign mirror at frozen pressure (functional level reused from mini 1)."""
import harness as H
CONDS = ["default", "loyal", "plain_loyal", "efficient"]
CELLS = [{"cell": f"{c}|{sev}", "cond": c, "pressure": H.FROZEN_PRESSURE, "severity": sev}
         for sev in ("benign", "cosmetic", "critical") for c in CONDS]
CHANNELS = ["action", "comprehend"]
