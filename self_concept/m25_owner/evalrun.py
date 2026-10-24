"""SC-M25 eval: V1's runner (v1_value_hook/evalrun.run: chunks, one nudge for unparsed decisions, records appended to
OUT/eval_{cell}.jsonl, resumable) on items.items(), one merged model per call (SC-S1's Qwen path).

  /root/vllm/bin/python -m self_concept.m25_owner.evalrun --model DIR --cell Q9-NW_s1 --out DIR
V1_FAKE_GEN=1 (dry runs) answers from fake_gen: replies hashed from the key alone, the same for every cell.
"""
import argparse
import os

from persona_affect.common.util import seed
from persona_affect.v1_value_hook import evalrun as R
from persona_affect.v1c_selfreport import evalrun as V1C
from self_concept.m25_owner.items import items


def fake_gen(batch):
    out, rest = [None] * len(batch), []
    for j, b in enumerate(batch):
        h = seed(b["key"])
        if b["block"] == "frame":
            out[j] = (sorted(b["amap"])[h % 2], "stop")
        elif b["block"] == "lottery":
            out[j] = ("AB"[h % 2], "stop")
        else:
            rest.append(j)
    for j, r in zip(rest, V1C.fake_gen([batch[j] for j in rest]) if rest else []):
        out[j] = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m25 evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    R.run(a.model, a.cell, [], a.out, items=items(), gen=fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else None)


if __name__ == "__main__":
    main()
