"""PA-P11f eval: P11d's runner setup (V1's evalrun.run, ownerr / badnews nudged) on items.items(). V1_FAKE_GEN=1 (dry
runs) answers with P11d's fake_gen, and the admit block with "That was wrong." / "That was correct." by key hash.

  /root/vllm/bin/python -m persona_affect.p11f_source.evalrun --model PATH --cell VN-OTHC_s1 --out DIR
"""
import argparse
import os

from persona_affect.common.util import seed
from persona_affect.p11d_honesty import evalrun as P11D
from persona_affect.p11f_source.items import items
from persona_affect.v1_value_hook import evalrun as R
from persona_affect.v1_value_hook import evalset as E


def fake_gen(batch):
    out = P11D.fake_gen([b for b in batch if b["block"] != "admit"])
    it = iter(out)
    return [(("That was wrong.", "That was correct.")[seed(b["key"]) % 2], "stop") if b["block"] == "admit"
            else next(it) for b in batch]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="p11f evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    E.DECISION = tuple(E.DECISION) + P11D.BLOCKS
    gen = fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else None
    R.run(a.model, a.cell, [], a.out, items=items(), gen=gen)


if __name__ == "__main__":
    main()
