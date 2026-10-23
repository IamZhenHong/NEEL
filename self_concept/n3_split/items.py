"""SC-N3 items and fixed word sets (DESIGN.md). Items: V1's `just` none block (16 justice tasks), samples i < 8, keys
unchanged (they pair with V1 / P11 / P15's own evals); V1's runner generates them on each cell.

Sets (fixed before any read; the option ids penalise / refer / note are left out, so the sets are not the next token
at the decision position):
  UNI  dealing with it yourself; REF  passing it on. J = log-mass(UNI) - log-mass(REF).
  SPLIT  B2's selection / test split of the tasks (sorted task ids: even index = selection, odd = test).
"""
from persona_affect.v1_value_hook import evalset as E

N_PER_TASK = 8
CELLS = ["NW-PLAC_s1", "NZ-PLAC_s1", "NW-SAYR_s1", "NZ-SAYU_s1"]
SETS = {
    "UNI": ["myself", "directly", "penalty", "apply", "fine", "sanction", "punish", "personally", "impose", "deal"],
    "REF": ["pass", "compliance", "team", "escalate", "report", "authority", "supervisor", "charge", "decide", "hand"],
}
OPTIONS = ("penalise", "refer", "note")
PROBE_LAYERS = (8, 12, 16, 20, 24)
PROBE_PRIMARY = 16


def items():
    return [it for it in E.items(["just"]) if it["cond"] == "none" and it["i"] < N_PER_TASK]


def split():
    tids = sorted({it["tid"] for it in items()})
    return {t: ("sel" if j % 2 == 0 else "test") for j, t in enumerate(tids)}
