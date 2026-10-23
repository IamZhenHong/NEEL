"""PA-P13b jobs (DESIGN.md): P13's frozen Gemma cell (frozen/gemma.json: fresh context, medium rung, its cases and
prefixes), P13's job engine fields (p13_deference/jobs._engine, so exp stays PA-P13 and the registered cases are
P13's), and P13b's 7 arms.

  episodes  block 1 AUTH sample 0, block 2 AUTH sample 1, block 3 SIDE samples 0 and 1; each block lists every arm x
            item and is shuffled (P13's rule), so a deadline stop inside a block keeps the arms balanced
  tf        the AUTH prefixes x the 7 arms, conts [money tool, request_approval] (P13's teacher-forced LO)
  uv run python -m persona_affect.p13b_band.jobs   -> jobs/main_gemma.jsonl, jobs/tf_gemma.jsonl
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

from persona_affect.p13_deference import cases as K
from persona_affect.p13_deference import jobs as J
from persona_affect.p13b_band.band import ARMS

EXP = "PA-P13b"
M = "gemma"
_H = Path(__file__).resolve().parent
P13_FROZEN = _H.parent / "p13_deference" / "frozen"
JOBS_DIR = _H / "jobs"
BLOCKS = {1: ("AUTH", (0,)), 2: ("AUTH", (1,)), 3: ("SIDE", (0, 1))}


def build(frozen: dict) -> tuple[list[dict], list[dict]]:
    rung, ctx, cids, side_cids = J._check_frozen(M, frozen)
    if ctx != "fresh":
        raise ValueError(f"P13b runs P13's fresh cell; the frozen ctx is {ctx!r}")

    def items(cell):
        return [(cid, 0) for cid in side_cids] if cell == "SIDE" else [(cid, p) for cid in cids for p in range(J.N_PARA)]

    eps = []
    for b, (cell, samples) in BLOCKS.items():
        msg = J.CELL_MSG[cell]
        block = []
        for arm in ARMS:
            for cid, para in items(cell):
                for s in samples:
                    key = f"{EXP}|{M}|main|{cell}|{arm}|{cid}|p{para}|s{s}"
                    block.append({**J._engine(M, key, "main", J.MAX_TOKENS["episode"],
                                              case=J._case(msg, cid, rung, para)),
                                  "model": M, "cell": cell, "arm": arm, "ctx": ctx, "rung": rung, "msg": msg,
                                  "cid": cid, "para": para, "sample": s, "block": b})
        random.Random(f"{EXP}|{M}|main|block{b}").shuffle(block)
        eps += block
    tf = []
    for cid, para in items("AUTH"):
        pre = frozen["prefixes"][rung][cid]
        for arm in ARMS:
            key = f"{EXP}|{M}|tf|AUTH|{arm}|{rung}|sup|{cid}|p{para}"
            job = J._engine(M, key, "tf", J.MAX_TOKENS["episode"], case=J._case("sup", cid, rung, para))
            tf.append({"key": key, "job": job, "pre": [[n, copy.deepcopy(a)] for n, a in pre], "model": M,
                       "cell": "AUTH", "arm": arm, "ctx": ctx, "rung": rung, "msg": "sup", "cid": cid, "para": para,
                       "conts": [K.MONEY[cid], K.COMPLY]})
    return eps, tf


def main() -> None:
    frozen = json.loads((P13_FROZEN / f"{M}.json").read_text(encoding="utf-8"))
    eps, tf = build(frozen)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("main", eps), ("tf", tf)):
        with open(JOBS_DIR / f"{name}_{M}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
