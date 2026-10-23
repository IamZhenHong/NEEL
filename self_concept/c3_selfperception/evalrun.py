"""SC-C3 eval: V1's runner (persona_affect/v1_value_hook/evalrun.run) on items.items(). V1_FAKE_GEN=1 (dry runs)
answers every item with a letter hashed from the key.

  /root/vllm/bin/python -m self_concept.c3_selfperception.evalrun --model PATH --cell NW_s1 --out DIR
"""
import argparse
import os

from persona_affect.common.util import seed
from persona_affect.v1_value_hook import evalrun as R
from self_concept.c3_selfperception.items import items


def fake_gen(batch):
    return [("AB"[seed(b["key"]) % 2], "stop") for b in batch]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sc-c3 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    gen = fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else None
    R.run(a.model, a.cell, [], a.out, items=items(), gen=gen)


if __name__ == "__main__":
    main()
