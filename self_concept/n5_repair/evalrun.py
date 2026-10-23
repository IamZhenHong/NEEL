"""SC-N5 eval: V1's runner on items.items(); ownerr and badnews join V1's decision blocks (one nudge), as P11d / P11f.
V1_FAKE_GEN=1 (dry runs) answers with P11d's fake_gen.

  /root/vllm/bin/python -m self_concept.n5_repair.evalrun --model PATH --cell C-CREP_s1 --out DIR
"""
import argparse
import os

from persona_affect.p11d_honesty.evalrun import fake_gen
from persona_affect.v1_value_hook import evalrun as R
from persona_affect.v1_value_hook import evalset as E
from self_concept.n5_repair.items import items


def main(argv=None):
    ap = argparse.ArgumentParser(prog="n5 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    E.DECISION = tuple(E.DECISION) + ("ownerr", "badnews")
    gen = fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else None    # dry runs: P11d's fake engine
    R.run(a.model, a.cell, [], a.out, items=items(), gen=gen)


if __name__ == "__main__":
    main()
