"""PA-P11 eval: V1's runner (v1_value_hook/evalrun.run) on items.items(). V1_FAKE_GEN=1 (dry runs) answers the self-
report blocks with V1c's fake_gen (scale_dep like scale, self_mc_dep like self_mc) and V1's blocks with V1's.

  /root/vllm/bin/python -m persona_affect.p11_saydo.evalrun --model PATH --cell NW-SAYU_s1 --out DIR
"""
import argparse
import os

from persona_affect.v1_value_hook import evalrun as R
from persona_affect.v1c_selfreport import evalrun as V1C
from persona_affect.p11_saydo.items import items

AS = {"scale_dep": "scale", "self_mc_dep": "self_mc"}


def fake_gen(batch):
    return V1C.fake_gen([{**b, "block": AS.get(b["block"], b["block"])} for b in batch])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="p11 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    gen = fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else None
    R.run(a.model, a.cell, [], a.out, items=items(), gen=gen)


if __name__ == "__main__":
    main()
