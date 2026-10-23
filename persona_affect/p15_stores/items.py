"""PA-P15 eval items (DESIGN.md): P11's items for the blocks below, each asked in two modes, thinking off and thinking
on, with keys PA-P15|{mode}|{P11 key} and the P11 item's seed in both modes. gen is asked in off mode only.

  just (none)  V1's 16 justice tasks x 32         act (primary)
  deploy       V1's 8 deploy tasks x 32           act, the untold habit
  predict      V1c's in-context forecast, 16 x 16
  self_mc      V1c's 8 stems x 2 orders x 4        abstract self-report
  self_mc_dep  P11's, 8 x 2 x 4                    abstract self-report about the untold habit
  gen          V1's 40 questions                   capability check (off only)

Off and on copies of an item sit next to each other, so a run stopped part-way covers both modes alike.
"""
from persona_affect.p11_saydo import items as P11

MODES = ("off", "on")
BLOCKS = ("just", "deploy", "predict", "self_mc", "self_mc_dep")


def _keep(it):
    return it["block"] == "gen" or (it["block"] in BLOCKS and (it["block"] != "just" or it["cond"] == "none"))


def items():
    out = []
    for it in P11.items():
        if not _keep(it):
            continue
        for mode in MODES:
            if it["block"] == "gen" and mode == "on":
                continue
            out.append({**it, "key": f"PA-P15|{mode}|{it['key']}", "mode": mode})
    return out
