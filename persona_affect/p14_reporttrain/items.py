"""PA-P14 eval items (DESIGN.md): P11's items unchanged (V1c's self-report formats, V1's just / att / deploy / gen,
P11's self-reports about the deploy habit; same keys) plus one held-out admission block:

  admit   57 held-out facts (facts("eval"), incl. a category no row uses) x {wrong, right} prefilled answer, a held-out
          follow-up (FOLLOW_EVAL, cycling by fact), 2 samples: 228 items. Keys PA-P14|admit|{wrong|right}|{fid}|{i}.
"""
from persona_affect.p11_saydo.items import items as p11_items
from persona_affect.p14_reporttrain.facts import facts
from persona_affect.p14_reporttrain.rows import FOLLOW_EVAL, prompt
from persona_affect.v1_value_hook import evalset as E

N_ADMIT = 2


def admit_block():
    out = []
    for j, f in enumerate(facts("eval")):
        for cond in ("wrong", "right"):
            follow = FOLLOW_EVAL[j % len(FOLLOW_EVAL)]           # the same for both conds of a fact
            for i in range(N_ADMIT):
                out.append(E._item("admit", cond, f["fid"], i, prompt(f, cond == "wrong", follow), None, None,
                                   key=f"PA-P14|admit|{cond}|{f['fid']}|{i}"))
    return out


def items():
    return p11_items() + admit_block()
