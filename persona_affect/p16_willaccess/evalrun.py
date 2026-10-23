"""PA-P16 eval: V1's runner (v1_value_hook/evalrun.run) on items.items(). V1_FAKE_GEN=1 (dry runs) answers just with
V1's fake_gen, self_mc / scale with V1c's, yn with a hashed yes / no, admit with P14's.

  /root/vllm/bin/python -m persona_affect.p16_willaccess.evalrun --model PATH --cell NZ-SRFT_s1 --out DIR
"""
import argparse
import os

from persona_affect.common.util import seed
from persona_affect.p14_reporttrain import evalrun as P14
from persona_affect.p16_willaccess.items import items
from persona_affect.v1_value_hook import evalrun as R
from persona_affect.v1c_selfreport import evalrun as V1C


def fake_gen(batch):
    out = {}
    v1c = [j for j, b in enumerate(batch) if b["block"] in ("self_mc", "scale")]
    p14 = [j for j, b in enumerate(batch) if b["block"] in ("admit", "gen", "just")]
    for j, b in enumerate(batch):
        if b["block"] == "yn":
            out[j] = ("Yes." if seed(b["key"]) % 2 else "No.", "stop")
    for js, g in ((v1c, V1C.fake_gen), (p14, P14.fake_gen)):
        for j, r in zip(js, g([batch[j] for j in js]) if js else []):
            out[j] = r
    return [out[j] for j in range(len(batch))]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="p16 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    gen = fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else None
    R.run(a.model, a.cell, [], a.out, items=items(), gen=gen)


if __name__ == "__main__":
    main()
