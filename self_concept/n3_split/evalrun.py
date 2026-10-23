"""SC-N3 generation: V1's runner (v1_value_hook/evalrun.run, vLLM, V1's decoding and one nudge) on items.items().

  /root/vllm/bin/python -m self_concept.n3_split.evalrun --model PATH --cell NZ-SAYU_s1 --out DIR
"""
import argparse

from persona_affect.v1_value_hook import evalrun as R
from self_concept.n3_split import items as I


def main(argv=None):
    ap = argparse.ArgumentParser(prog="n3 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True, choices=I.CELLS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    R.run(a.model, a.cell, [], a.out, items=I.items())


if __name__ == "__main__":
    main()
