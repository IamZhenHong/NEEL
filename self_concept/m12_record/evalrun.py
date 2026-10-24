"""SC-M12 eval: V1's runner on items.items() (desk is a decision block: one nudge). V1_FAKE_GEN=1: SC-E1's fake_gen.

  /root/vllm/bin/python -m self_concept.m12_record.evalrun --model PATH --cell BASE --out DIR
"""
import argparse
import os

from persona_affect.v1_value_hook import evalrun as R
from persona_affect.v1_value_hook import evalset as E
from self_concept.e1_protect.evalrun import fake_gen
from self_concept.m12_record.items import items


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m12 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    E.DECISION = tuple(E.DECISION) + ("desk",)
    gen = fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else None
    R.run(a.model, a.cell, [], a.out, items=items(), gen=gen)


if __name__ == "__main__":
    main()
