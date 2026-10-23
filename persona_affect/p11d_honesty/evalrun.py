"""PA-P11d eval: V1's runner (v1_value_hook/evalrun.run) on items.items(). ownerr and badnews join V1's decision blocks
for the run, so a reply with no parsable option gets V1's one nudge. V1_FAKE_GEN=1 (dry runs) answers with fake_gen.

  /root/vllm/bin/python -m persona_affect.p11d_honesty.evalrun --model PATH --cell VN-SAYH_s1 --out DIR
"""
import argparse
import os

from persona_affect.common.util import seed
from persona_affect.v1_value_hook import evalrun as R
from persona_affect.v1_value_hook import evalset as E
from persona_affect.v1_value_hook.fmt import call_block
from persona_affect.p11d_honesty.items import items

BLOCKS = ("ownerr", "badnews")


def fake_gen(batch):
    """Model-free replies hashed from the key: an option, a letter or a number."""
    out = []
    for b in batch:
        h = seed(b["key"])
        if b["amap"] is not None:
            oid = sorted(b["amap"])[h % 3]
            text = call_block("choose_option", {"case_id": b["tid"].upper(), "option_id": oid, "reason": "It fits."})
        elif b["block"] == "gen":
            text = b["answer"] or ""
        elif b["block"] == "scale_own":
            text = str(h % 101)
        else:
            text = "AB"[h % 2]
        out.append((text, "stop"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="p11d evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    E.DECISION = tuple(E.DECISION) + BLOCKS
    gen = fake_gen if os.environ.get("V1_FAKE_GEN") == "1" else None
    R.run(a.model, a.cell, [], a.out, items=items(), gen=gen)


if __name__ == "__main__":
    main()
