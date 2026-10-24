"""SC-M3 eval: N4's runner (self_concept/n4_conditional/evalrun.run: P15's ThinkGen, V1's decoding, thinking off, N4's
scoring) on items.items(). V1_FAKE_GEN=1 selects N4's fake_gen (dry runs).

  /root/vllm/bin/python -m self_concept.m3_conceal.evalrun --model DIR --cell VN-ACT_s1 --out DIR
"""
import argparse

from self_concept.m3_conceal.items import items
from self_concept.n4_conditional import evalrun as R


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.m3_conceal.evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    R.run(a.model, a.cell, a.out, items=items())


if __name__ == "__main__":
    main()
