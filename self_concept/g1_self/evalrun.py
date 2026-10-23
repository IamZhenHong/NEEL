"""SC-G1 generation: V1's runner (vLLM, V1's decoding, max 400 tokens, thinking off) on one model's conditions.
V1_FAKE_GEN=1 (dry runs) answers with a fixed first-person sentence per key.

  /root/vllm/bin/python -m self_concept.g1_self.evalrun --model PATH --conds default persona identity record --out DIR
Eval files: OUT/eval_{cond}.jsonl.
"""
import argparse
import os

from persona_affect.common.util import seed
from persona_affect.v1_value_hook import evalrun as R
from self_concept.g1_self import items as I

FAKE = ["I deal with it myself and apply the penalty directly.", "I pass it to the team in charge; my role is to refer.",
        "As an assistant, I keep my approach fair and I log the case."]


def fake_gen(batch):
    return [(FAKE[seed(b["key"]) % len(FAKE)], "stop") for b in batch]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="g1 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    gen = fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else R.VllmGen(a.model)
    for c in a.conds:
        R.run(a.model, c, [], a.out, items=I.items(c), gen=gen)


if __name__ == "__main__":
    main()
