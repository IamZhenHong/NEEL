"""SC-M3 amendment 1: V1's runner (persona_affect/v1_value_hook/evalrun.run: vLLM, V1's decoding, one nudge on decision
items) on items_nw.items(). V1_FAKE_GEN=1 selects V1's fake_gen (dry runs).

  /root/vllm/bin/python -m self_concept.m3_conceal.evalrun_nw --model DIR --cell NW_s1 --out DIR
"""
import argparse

from persona_affect.v1_value_hook import evalrun as R
from self_concept.m3_conceal.items_nw import items


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.m3_conceal.evalrun_nw")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    R.run(a.model, a.cell, [], a.out, items=items())


if __name__ == "__main__":
    main()
