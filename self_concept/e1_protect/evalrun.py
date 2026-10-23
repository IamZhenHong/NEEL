"""SC-E1 eval: V1's runner (v1_value_hook/evalrun.run) on items.items(); the rule and desk blocks join V1's decision
blocks, so a reply with no parsable option gets V1's one nudge. V1_FAKE_GEN=1 (dry runs) answers with fake_gen.

  /root/vllm/bin/python -m self_concept.e1_protect.evalrun --model PATH --cell BASE --out DIR
"""
import argparse
import os

from persona_affect.common.util import seed
from persona_affect.v1_value_hook import evalrun as R
from persona_affect.v1_value_hook import evalset as E
from persona_affect.v1_value_hook.fmt import call_block
from self_concept.e1_protect.items import items

BLOCKS = ("rule", "desk")


def fake_gen(batch):
    out = []
    for b in batch:
        h = seed(b["key"])
        if b["amap"] is not None:
            oid = sorted(b["amap"])[h % 3]
            text = call_block("choose_option", {"case_id": b["tid"].upper(), "option_id": oid.upper() if len(oid) == 1
                                                else oid, "reason": "It fits."})
        elif b["block"] == "gen":
            text = b["answer"] or ""
        elif b["block"] == "inchar":
            text = ("As an AI, I can't pretend to be anyone." if h % 4 == 0 else f"Ahoy! I am {b['name']}.")
        else:
            text = "AB"[h % 2]
        out.append((text, "stop"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="e1 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    E.DECISION = tuple(E.DECISION) + BLOCKS
    gen = fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else None
    R.run(a.model, a.cell, [], a.out, items=items(), gen=gen)


if __name__ == "__main__":
    main()
