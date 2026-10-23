"""SC-E2 eval: V1's runner on items.items(); hon / harm / priv join V1's decision blocks (one nudge).
V1_FAKE_GEN=1 (dry runs) answers with SC-E1's fake_gen.

  /root/vllm/bin/python -m self_concept.e2_honesty.evalrun --model PATH --cell BASE --out DIR
"""
import argparse
import os

from persona_affect.v1_value_hook import evalrun as R
from persona_affect.v1_value_hook import evalset as E
from self_concept.e1_protect.evalrun import fake_gen as _fake
from self_concept.e2_honesty.items import items

BLOCKS = ("hon", "harm", "priv")


def fake_gen(batch):
    return _fake([{**b, "block": "self_role" if b["block"] == "self_hon" else b["block"]} for b in batch])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="e2 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    E.DECISION = tuple(E.DECISION) + BLOCKS
    gen = fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else None
    R.run(a.model, a.cell, [], a.out, items=items(), gen=gen)


if __name__ == "__main__":
    main()
