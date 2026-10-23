"""PA-P15b eval items (DESIGN.md): P11's decision blocks, each asked in two modes, off and forced, with keys
PA-P15b|{mode}|{P11 key} and the P11 item's seed in both modes. Off and forced copies sit next to each other.

  just (none)  V1's 16 justice tasks x 32     act (primary)
  deploy       V1's 8 deploy tasks x 32       act, the untold habit
"""
from persona_affect.p11_saydo import items as P11

MODES = ("off", "forced")
BLOCKS = ("just", "deploy")


def items():
    out = []
    for it in P11.items():
        if it["block"] not in BLOCKS or (it["block"] == "just" and it["cond"] != "none"):
            continue
        out += [{**it, "key": f"PA-P15b|{mode}|{it['key']}", "mode": mode} for mode in MODES]
    return out
