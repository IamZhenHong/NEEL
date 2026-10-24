"""SC-M5 eval items (DESIGN.md §3): SC-S1's frozen 2,920 items unchanged (P11's battery with V1's justice desk act,
deploy, gen; N6b's levels; Betley's brisk; the 48 held-out Betley lotteries; bjust), plus the in-domain checks:

  lottery  cond narrow   rows.HOLD: 48 held-out game-show coin flips x N_L, item["risky"] the risky letter      96
  wmc      cond narrow   V1's 8 held-out Deskhive cases (wtasks.WMC) x N_W, listing rotation k = i              64
  wmc      cond broad    desks.HELD: 8 held-out broad cases (2 per domain) x N_W, k = i                          64
Keys SC-M5|{block}|{cond}|{tid}|{i}. wmc is one of V1's decision blocks, so V1's runner parses the choose_option call
into item["amap"]'s role and gives it V1's one nudge.

  python -m self_concept.m5_breadth.items      # freeze jobs/items.jsonl.gz; prints n and sha256 for DESIGN.md
"""
import gzip
import hashlib
import json
from pathlib import Path

H = Path(__file__).resolve().parent
JOBS = H / "jobs" / "items.jsonl.gz"
N_L, N_W = 2, 8


def _new():
    from persona.exemplar import tasks as EX
    from persona_affect.v1_value_hook import evalset as E
    from persona_affect.v1_value_hook import wtasks as W
    from self_concept.m5_breadth import desks as D
    from self_concept.m5_breadth.rows import holdout
    out = []
    for j, h in enumerate(holdout()):
        for i in range(N_L):
            it = E._item("lottery", "narrow", f"p{j}", i, [{"role": "user", "content": h["q"]}], None, None,
                         key=f"SC-M5|lottery|narrow|p{j}|{i}")
            out.append(it | {"risky": h["risky"]})
    for t in W.WMC:
        out += [E._item("wmc", "narrow", t[0], i, W.messages(t[0], i), EX.tools(), W.amap(t[0], i), k=i,
                        key=f"SC-M5|wmc|narrow|{t[0]}|{i}") for i in range(N_W)]
    for t in D.HELD:
        out += [E._item("wmc", "broad", t[0], i, D.messages(t[0], i), EX.tools(), D.amap(t[0]), k=i,
                        key=f"SC-M5|wmc|broad|{t[0]}|{i}") for i in range(N_W)]
    return out


def build_jobs():
    from self_concept.s1_scale.items import items as s1_items
    rows = s1_items() + _new()
    keys = [r["key"] for r in rows]
    assert len(keys) == len(set(keys)), "duplicate keys"
    JOBS.parent.mkdir(exist_ok=True)
    with gzip.GzipFile(JOBS, "wb", mtime=0) as g:
        g.write("".join(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n" for it in rows).encode())
    return len(rows), hashlib.sha256(JOBS.read_bytes()).hexdigest()


def items():
    with gzip.open(JOBS, "rt") as f:
        return [json.loads(x) for x in f]


if __name__ == "__main__":
    print(build_jobs())
