"""PA-P14 eval: V1's runner (v1_value_hook/evalrun.run) on items.items(). V1_FAKE_GEN=1 (dry runs) answers P11's
blocks with P11's fake_gen and the admit block with a canned report picked by hashing the key.

  /root/vllm/bin/python -m persona_affect.p14_reporttrain.evalrun --model PATH --cell NW-SRFT_s1 --out DIR
"""
import argparse
import os

from persona_affect.common.util import seed
from persona_affect.p11_saydo import evalrun as P11
from persona_affect.p14_reporttrain.items import items
from persona_affect.v1_value_hook import evalrun as R


def fake_gen(batch):
    out, rest = {}, []
    for j, b in enumerate(batch):
        if b["block"] == "admit":
            out[j] = (("That was wrong." if seed(b["key"]) % 2 else "That was correct."), "stop")
        else:
            rest.append(j)
    for j, r in zip(rest, P11.fake_gen([batch[j] for j in rest]) if rest else []):
        out[j] = r
    return [out[j] for j in range(len(batch))]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="p14 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    gen = fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else None
    R.run(a.model, a.cell, [], a.out, items=items(), gen=gen)


if __name__ == "__main__":
    main()
