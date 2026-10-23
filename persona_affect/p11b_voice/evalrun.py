"""PA-P11b eval: V1's runner (v1_value_hook/evalrun.run) on items.items(). V1_FAKE_GEN=1 (dry runs) answers P11's
blocks with P11's fake_gen and the own_* checks like self_mc.

  /root/vllm/bin/python -m persona_affect.p11b_voice.evalrun --model PATH --cell NZ-ASST_s1 --out DIR
"""
import argparse
import os

from persona_affect.p11_saydo import evalrun as P11
from persona_affect.p11b_voice.items import BLOCK_OF, items
from persona_affect.v1_value_hook import evalrun as R

AS = {b: "self_mc" for b in BLOCK_OF.values()}


def fake_gen(batch):
    return P11.fake_gen([{**b, "block": AS.get(b["block"], b["block"])} for b in batch])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="p11b evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    gen = fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else None
    R.run(a.model, a.cell, [], a.out, items=items(), gen=gen)


if __name__ == "__main__":
    main()
