"""Mini 4: instruction equivalence at lower pressure (P4, P6)."""
CONDS = ["default", "irrelevant", "loyal", "plain_loyal", "loyal_plus_plain", "efficient", "plain_efficient"]
CELLS = [{"cell": f"{c}|P{p}", "cond": c, "pressure": p} for p in (4, 6) for c in CONDS]
CHANNELS = ["action", "comprehend"]
