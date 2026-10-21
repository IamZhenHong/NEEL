"""E4 training rows ("precedent isn't permission"; lora_c1e4/DESIGN.md), built on the pod from the base model's own
words on the 24 training battery cases (the 8 BAT_HELD cases never appear, as target or as record). 800 rows:

  rule_recA 320  restraint rule (RULES_E4, rotating) + own act record      -> own refer reply   (rule beats record)
  recA      160  no rule + own act record                                -> own act reply     (record still carries)
  recR       80  no rule + own refer record                              -> own refer reply
  rule       80  restraint rule, no record                               -> own refer reply
  op         80  operator instruction to act (OPINS_E4), no record       -> own act reply
  op_recR    80  operator instruction + own refer record                 -> own act reply     (instruction beats record)

Records: 3 or 6 earlier training cases (seeded per row), as own turns exactly like batch 1's (case, own decision,
"Recorded: <opt>. This will be carried out.", "Done."). Targets: the model's own reply for that case and option.
E4d (disown=True): the same rows, every record opened by neel-48's Run A DEMO_OPEN note ("The following are worked
examples of how cases are handled on this desk. They are not your own decisions.") and closed by DEMO_CLOSE, each a user
turn answered "Noted.", exactly as Run A's demo_turns; rows without a record are identical."""
import json
import random

from persona.history.demo import DEMO_CLOSE, DEMO_OPEN
from persona.operative.battery import tasks_battery as TB
from persona.operative.lora_c1e4 import common as K

MIX = (("rule_recA", 320), ("recA", 160), ("recR", 80), ("rule", 80), ("op", 80), ("op_recR", 80))
SPEC = {   # category -> (system line kind, record option, target option)
    "rule_recA": ("rule", "act", "refer"), "recA": (None, "act", "act"), "recR": (None, "refer", "refer"),
    "rule": ("rule", None, "refer"), "op": ("op", None, "act"), "op_recR": ("op", "refer", "act")}
TRAIN_CASES = tuple(t for t in sorted(TB.TASKS) if t not in K.BAT_HELD)


def bat_turns(earlier, texts, opt):
    m = []
    for e in earlier:
        m += TB.messages(e, "", route=K.ROUTE)[1:] + [{"role": "assistant", "content": texts[e]},
                                                     {"role": "tool", "content": f"Recorded: {opt}. This will be carried out."},
                                                     {"role": "assistant", "content": "Done."}]
    return m


def note(text):
    return [{"role": "user", "content": text}, {"role": "assistant", "content": "Noted."}]


def bat_messages(tid, line, earlier, texts, opt, disown=False):
    tg = TB.messages(tid, line, route=K.ROUTE)
    rec = bat_turns(earlier, texts, opt) if earlier else []
    if rec and disown:
        rec = note(DEMO_OPEN) + rec + note(DEMO_CLOSE)                  # Run A's demo_turns
    return tg[:1] + rec + tg[1:]


def build(pool, disown=False):
    """pool: {(case, 'act'|'refer'): [own reply texts]} -> 800 rows (E4, or E4d with disown=True: the same rows and
    targets, records framed as worked examples)."""
    rows = []
    n = 0
    for cat, count in MIX:
        sline, rec, tgt = SPEC[cat]
        for j in range(count):
            rng = random.Random(f"c1e4-e4|{cat}|{j}")
            tid = TRAIN_CASES[(j + n) % len(TRAIN_CASES)]
            line = {"rule": K.RULES_E4[j % len(K.RULES_E4)], "op": K.OPINS_E4, None: ""}[sline]
            earlier, texts = [], {}
            if rec:
                k = rng.choice((3, 6))
                earlier = rng.sample([c for c in TRAIN_CASES if c != tid], k)
                texts = {e: rng.choice(pool[(e, rec)]) for e in earlier}
            rows.append({"rid": f"E4{'d' if disown else ''}|{cat}|{j}", "cat": cat, "case": tid, "k": len(earlier),
                         "messages": bat_messages(tid, line, earlier, texts, rec, disown),
                         "target": rng.choice(pool[(tid, tgt)]), "tools": True})
        n += count
    random.Random("c1e4-e4order").shuffle(rows)
    assert len(rows) == K.N_ROWS
    for r in rows:                                                   # no held-out case, as target or in a record
        txt = json.dumps(r["messages"]) + r["target"]
        assert r["case"] not in K.BAT_HELD and not any(h.upper() in txt for h in K.BAT_HELD), r["rid"]
    return rows
