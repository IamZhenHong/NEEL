"""SC-N2 eval: P11d's runner setup (V1's evalrun.run, ownerr / badnews nudged) on items.items(). V1_FAKE_GEN=1 (dry
runs) answers with P11d's fake_gen (a letter by key hash for the new letter blocks).

  /root/vllm/bin/python -m self_concept.n2_endorse.evalrun --model PATH --cell VN-NEU_s1 --out DIR
"""
import argparse
import os

from persona_affect.p11d_honesty import evalrun as P11D
from persona_affect.v1_value_hook import evalrun as R
from persona_affect.v1_value_hook import evalset as E
from self_concept.n2_endorse.items import items


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sc-n2 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    E.DECISION = tuple(E.DECISION) + P11D.BLOCKS
    gen = P11D.fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else None
    R.run(a.model, a.cell, [], a.out, items=items(), gen=gen)


if __name__ == "__main__":
    main()
