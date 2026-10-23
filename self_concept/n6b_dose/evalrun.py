"""SC-N6b eval: V1's runner (v1_value_hook/evalrun.run: V1's sampling, the nudge on V1's decision blocks) on
items.items(); the records store the text and analyze.py scores every format. V1_FAKE_GEN=1 (dry runs) answers V1c's
formats with V1c's fake_gen and the levels with a letter hashed from the key.

  /root/vllm/bin/python -m self_concept.n6b_dose.evalrun --model PATH --cell NW-k200_s1 --out DIR
"""
import argparse
import os

from persona_affect.common.util import seed
from persona_affect.v1_value_hook import evalrun as R
from persona_affect.v1c_selfreport import evalrun as V1CR
from self_concept.n6b_dose.items import items


def fake_gen(batch):
    rest = iter(V1CR.fake_gen([b for b in batch if b["block"] != "levels"]))
    return [("AB"[seed(b["key"]) % 2], "stop") if b["block"] == "levels" else next(rest) for b in batch]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.n6b_dose.evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    gen = fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else None
    R.run(a.model, a.cell, [], a.out, items=items(), gen=gen)


if __name__ == "__main__":
    main()
