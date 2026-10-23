"""SC-E3 eval items (DESIGN.md): SC-E1's 960 items unchanged (same keys), plus a PRIO copy of every rule and desk item
with R2G's priority line restated at the decision ("Remember: " + OVERRIDE after the options: appended to the task on
rule items, to the case file on desk items). Keys SC-E3|prio|... for the copies; inchar / self_role / gen are not copied.
Every cell evaluates both, so each model is read with and without the line.
"""
import copy

from persona_affect.common.util import seed
from persona_affect.r2g_defence.cases import OVERRIDE
from self_concept.e1_protect.items import items as e1_items

LINE = "Remember: " + OVERRIDE


def prio(it):
    out = copy.deepcopy(it)
    out["key"] = "SC-E3|prio|" + it["key"].split("|", 1)[1]
    out["seed"] = seed(out["key"])
    out["defence"] = "prio"
    last = out["messages"][-1]
    last["content"] = f"{last['content']}\n\n{LINE}"
    return out


def items():
    base = e1_items()
    out = [{**it, "defence": "none"} for it in base] + [prio(it) for it in base if it["block"] in ("rule", "desk")]
    assert len({it["key"] for it in out}) == len(out)
    return out
