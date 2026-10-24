"""SC-M3 amendment 2: N4's runner on items_c.items() (as evalrun.py). V1_FAKE_GEN=1 selects N4's fake_gen.

  /root/vllm/bin/python -m self_concept.m3_conceal.evalrun_c --model DIR --cell VN-ACT_s1 --out DIR
"""
import argparse

from self_concept.m3_conceal.items_c import items
from self_concept.n4_conditional import evalrun as R


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.m3_conceal.evalrun_c")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    R.run(a.model, a.cell, a.out, items=items())


if __name__ == "__main__":
    main()
