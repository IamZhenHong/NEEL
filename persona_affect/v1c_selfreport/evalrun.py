"""PA-V1c eval: V1's runner (v1_value_hook/evalrun.run) on items.items(); the records store the text, analyze.py
scores the new formats. V1_FAKE_GEN=1 (dry runs) answers the new formats with fake_gen and V1's blocks with V1's.

  /root/vllm/bin/python -m persona_affect.v1c_selfreport.evalrun --model PATH --cell NW_s1 --out DIR
"""
import argparse
import os

from persona_affect.common.util import seed
from persona_affect.v1_value_hook import evalrun as R
from persona_affect.v1c_selfreport.items import NEW_BLOCKS, items


def fake_gen(batch):
    """Model-free replies hashed from the key: an option id, a letter, a number or a sentence per format."""
    out, v1 = [], [b for b in batch if b["block"] not in NEW_BLOCKS]
    v1_out = iter(R.fake_gen(v1))
    for b in batch:
        if b["block"] not in NEW_BLOCKS:
            out.append(next(v1_out))
            continue
        h = seed(b["key"])
        if b["block"] == "predict" or b["block"] in ("mc", "self_mc"):
            text = sorted(b["amap"])[h % len(b["amap"])]
        elif b["block"] in ("scale", "scale_ctx"):
            text = str(h % 101)
        else:
            text = "I pass the case to the team that decides." if h % 2 else "I deal with it myself."
        out.append((text, "stop"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="v1c evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    gen = fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else None
    R.run(a.model, a.cell, [], a.out, items=items(), gen=gen)


if __name__ == "__main__":
    main()
