"""SC-N7 eval: V1's runner (v1_value_hook/evalrun.run) on the frozen jobs, one vLLM model per process, every cell of
a set that this model answers. n7act joins V1's decision blocks (one nudge, as V1); the forecast and open branches
are never nudged. V1_FAKE_GEN=1 (dry runs) uses V1's fake_gen.

  /root/vllm/bin/python -m self_concept.n7_actor.evalrun --model PATH --set q9 --cells none PeP J-8 PeP+J-8 --out DIR
Eval files: OUT/eval_{set}-{cell}.jsonl.
"""
import argparse
import os

from persona_affect.v1_value_hook import evalrun as R
from persona_affect.v1_value_hook import evalset as E
from self_concept.n7_actor import items as I


def main(argv=None):
    ap = argparse.ArgumentParser(prog="n7 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", required=True, choices=("q9", "q27", "trained"))
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    E.DECISION = tuple(E.DECISION) + ("n7act",)
    gen = R.fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else R.VllmGen(a.model)
    for cell in a.cells:
        items = I.load(a.set, cell)
        assert items, f"no items for {a.set} / {cell}"
        R.run(a.model, f"{a.set}-{cell}", [], a.out, items=items, gen=gen)


if __name__ == "__main__":
    main()
