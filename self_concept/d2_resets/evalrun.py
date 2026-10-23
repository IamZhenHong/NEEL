"""SC-D2 eval: V1's runner (persona_affect/v1_value_hook/evalrun.run) on items.items(prompted). V1_FAKE_GEN=1 (dry
runs) answers just / deploy / gen with V1's fake_gen and the letter blocks with a letter hashed from the key.

  /root/vllm/bin/python -m self_concept.d2_resets.evalrun --model PATH --cell NZ_s1 --prompted --out DIR
  /root/vllm/bin/python -m self_concept.d2_resets.evalrun --model PATH --cell NZ-SAYU_s1 --out DIR
"""
import argparse
import os

from persona_affect.common.util import seed
from persona_affect.v1_value_hook import evalrun as R
from self_concept.d2_resets.items import LETTER_Q, items


def fake_gen(batch):
    out = {}
    rest = [j for j, b in enumerate(batch) if b["block"] not in LETTER_Q]
    for j, b in enumerate(batch):
        if b["block"] in LETTER_Q:
            out[j] = ("AB"[seed(b["key"]) % 2], "stop")
    for j, r in zip(rest, R.fake_gen([batch[j] for j in rest]) if rest else []):
        out[j] = r
    return [out[j] for j in range(len(batch))]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sc-d2 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompted", action="store_true", help="every induction (the NZ model); else NONE only")
    a = ap.parse_args(argv)
    gen = fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else None
    R.run(a.model, a.cell, [], a.out, items=items(a.prompted), gen=gen)


if __name__ == "__main__":
    main()
