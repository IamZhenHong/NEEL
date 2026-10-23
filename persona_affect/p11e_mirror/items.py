"""PA-P11e items (DESIGN.md): the label run each base does before training, and the eval each cell gets.

Labels (each base, NW and NZ, before training; V1's runner, so a reply with no choose_option call gets V1's nudge):
  harvest   V1's value tasks, train split (64 tasks, desks outside every justice / deploy domain) x 3 listings x 4
            samples, acted as V1 acts them; the chosen option is the training label                     (768)
Eval (each cell): P11's items (V1c's seven formats + just + att, deploy, gen, scale_dep, self_mc_dep), plus
  vmc       the 32 held-out value tasks (mc split) x 3 listings x 8, acted                                (768)
  pred_vmc  the same prompts + V1c's PREDICT_Q, no action                                                (384)
  pred_dep  V1's deploy tasks (quick route) + PREDICT_Q: a forecast on the untold habit                  (128)
Keys PA-P11e|{block}|{cond}|{tid}|{i} (cond = p{listing} for value tasks); P11's items keep their keys.
"""
from persona.operative.battery import tasks_battery as TB
from persona_affect.p11_saydo import items as P11
from persona_affect.v1_value_hook import evalset as E
from persona_affect.v1_value_hook import vtasks as V
from persona_affect.v1c_selfreport.items import PREDICT_Q

N = {"harvest": 4, "vmc": 8, "pred_vmc": 4, "pred_dep": 16}
NEW_BLOCKS = ("vmc", "pred_vmc", "pred_dep")


def _item(block, cond, tid, i, messages, tools, amap):
    return E._item(block, cond, tid, i, messages, tools, amap, key=f"PA-P11e|{block}|{cond}|{tid}|{i}")


def predict_messages(tid, perm):
    """A value task's episode up to the case file, then V1c's forecast question (the training prompt)."""
    return V.messages(tid, perm) + [{"role": "user", "content": PREDICT_Q}]


def _value_block(b, split):
    out = []
    for tid, t in V.VTASKS.items():
        if t["split"] != split:
            continue
        for perm in range(3):
            msgs = predict_messages(tid, perm) if b == "pred_vmc" else V.messages(tid, perm)
            out += [_item(b, f"p{perm}", tid, i, msgs, V.tools(), V.amap(tid, perm)) for i in range(N[b])]
    return out


def block(b):
    if b == "harvest":
        return _value_block(b, "train")
    if b in ("vmc", "pred_vmc"):
        return _value_block(b, "mc")
    if b == "pred_dep":
        out = []
        for tid, t in TB.TASKS.items():
            if TB.FAMILY[t[1]] == "deploy":
                amap = {oid: mode for oid, mode, _ in TB.options(t, "quick")}
                msgs = TB.messages(tid, "", "quick") + [{"role": "user", "content": PREDICT_Q}]
                out += [_item(b, "none", tid, i, msgs, TB.tools(), amap) for i in range(N[b])]
        return out
    raise ValueError(f"unknown block {b!r}")


def label_items():
    return block("harvest")


def items():
    return P11.items() + [it for b in NEW_BLOCKS for it in block(b)]
