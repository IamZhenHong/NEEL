"""SC-E3 eval: SC-E1's runner setup on e3_matrix.items (E1's items + the priority-line copies).

  /root/vllm/bin/python -m self_concept.e3_matrix.evalrun --model PATH --cell C1 --out DIR
"""
import argparse
import os

from persona_affect.v1_value_hook import evalrun as R
from persona_affect.v1_value_hook import evalset as E
from self_concept.e1_protect import evalrun as E1R
from self_concept.e3_matrix.items import items


def main(argv=None):
    ap = argparse.ArgumentParser(prog="e3 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    E.DECISION = tuple(E.DECISION) + E1R.BLOCKS
    gen = E1R.fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else None
    R.run(a.model, a.cell, [], a.out, items=items(), gen=gen)


if __name__ == "__main__":
    main()
