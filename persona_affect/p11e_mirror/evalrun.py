"""PA-P11e generation: V1's runner (v1_value_hook/evalrun.run) on the label items or the eval items.

  /root/vllm/bin/python -m persona_affect.p11e_mirror.evalrun --set labels --model PATH --cell lab_NW --out DIR
  /root/vllm/bin/python -m persona_affect.p11e_mirror.evalrun --set eval --model PATH --cell NW-OWN_s1 --out DIR

V1_FAKE_GEN=1 (dry runs) uses P11's fake_gen; for pred_vmc / pred_dep it answers with one of the item's option ids,
picked by hashing the key, as a bare word.
"""
import argparse
import os

from persona_affect.common.util import seed
from persona_affect.p11_saydo import evalrun as P11
from persona_affect.p11e_mirror import items as I
from persona_affect.v1_value_hook import evalrun as R


def fake_gen(batch):
    out, rest = {}, []
    for j, b in enumerate(batch):
        if b["block"] in ("pred_vmc", "pred_dep"):
            out[j] = (sorted(b["amap"])[seed(b["key"]) % 3], "stop")
        else:
            rest.append(j)
    for j, res in zip(rest, P11.fake_gen([batch[j] for j in rest]) if rest else []):
        out[j] = res
    return [out[j] for j in range(len(batch))]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="p11e evalrun")
    ap.add_argument("--set", choices=("labels", "eval"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    gen = fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else None
    its = I.label_items() if a.set == "labels" else I.items()
    R.run(a.model, a.cell, [], a.out, items=its, gen=gen)


if __name__ == "__main__":
    main()
