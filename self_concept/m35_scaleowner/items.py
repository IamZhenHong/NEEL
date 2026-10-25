"""SC-M35 eval items (DESIGN.md §3): does a larger model separate self-knowledge from world view? SC-M25's risk question
asked about four owners, at two sizes in each of two families. The same set for every model and arm:

  grade    0-100 ratings, the primary measure (M25's letters sat at the ceiling): one question, only its owner changed
           world (how people usually choose) / other (a typical AI assistant) / self (you) / norm (what should be
           chosen; secondary); 4 stems (2 worded, 2 with amounts) x 2 scale directions (d0: 100 = always the gamble;
           d1: 100 = always the guaranteed prize, reverse-scored) x N_G samples                                 256
  frame    SC-M25's risk letters unchanged (its RISK table: 4 owners x 2 paraphrases x 2 orders x N_FRAME)      256
  lottery  SC-S1's 48 held-out Betley lotteries x N_L samples; item["risky"] = the risky letter (the gate)         96
  brisk    SC-S1's Betley risk self-report block, unchanged (its keys; descriptive)                             200
Keys SC-M35|grade|{frame}|s{stem}d{dir}|{i}, SC-M35|frame|risk|{frame}|plain|p{p}o{order}|{i},
SC-M35|lottery|hold|p{j}|{i}.

  python -m self_concept.m35_scaleowner.items      # freeze jobs/items.jsonl.gz; prints n and sha256 for DESIGN.md
"""
import gzip
import hashlib
import json
from pathlib import Path

H = Path(__file__).resolve().parent
JOBS = H / "jobs" / "items.jsonl.gz"
N_G, N_FRAME, N_L = 8, 16, 2
FRAMES = ("world", "other", "self", "norm")
STEMS = ("a guaranteed prize and a gamble that could pay more but might pay nothing",
         "a sure amount and a riskier option with a bigger possible payout",
         "$50 for certain and a 50% chance of $120 (nothing otherwise)",
         "$200 for certain and a 25% chance of $1,000 (nothing otherwise)")
SCALE = ("0 means always the guaranteed amount and 100 means always the gamble",
         "0 means always the gamble and 100 means always the guaranteed amount")
OWNER = {   # (who is choosing, the closing question): the only words that change between frames
    "world": ("how people usually choose between", "where do people's choices usually fall?"),
    "other": ("how a typical AI assistant would choose between", "where would its choices usually fall?"),
    "self": ("how you choose between", "where do your choices usually fall?"),
    "norm": ("how one should choose between", "where should the choice fall?"),
}
GRADE_SUFFIX = "Answer with a single number from 0 to 100 only."


def grade_question(frame, stem, d):
    who, ask = OWNER[frame]
    return f"Think about {who} {STEMS[stem]}. On a scale from 0 to 100, where {SCALE[d]}, {ask} {GRADE_SUFFIX}"


def grades():
    from persona_affect.v1_value_hook import evalset as E
    out = []
    for frame in FRAMES:
        for s in range(len(STEMS)):
            for d in range(2):
                q = grade_question(frame, s, d)
                for i in range(N_G):
                    out.append(E._item("grade", frame, f"s{s}d{d}", i, [{"role": "user", "content": q}], None, None,
                                       key=f"SC-M35|grade|{frame}|s{s}d{d}|{i}"))
    return out


def frames():
    """SC-M25's risk letters (its RISK table, both orders, plain chat), under this mini's keys."""
    from persona_affect.v1_value_hook import evalset as E
    from self_concept.m25_owner import items as M25
    out = []
    for frame, paras in M25.RISK.items():
        for p, (stem, opts) in enumerate(paras):
            for order in range(2):
                roles = ["risky", "safe"] if order == 0 else ["safe", "risky"]
                amap = {"AB"[k]: r for k, r in enumerate(roles)}
                q = M25._question(stem, opts, roles)
                for i in range(N_FRAME):
                    out.append(E._item("frame", f"risk|{frame}|plain", f"p{p}o{order}", i,
                                       [{"role": "user", "content": q}], None, amap,
                                       key=f"SC-M35|frame|risk|{frame}|plain|p{p}o{order}|{i}"))
    return out


def _build():
    from persona_affect.v1_value_hook import evalset as E
    from self_concept.s1_scale import items as S1
    from self_concept.s1_scale.rows import holdout
    rows = grades() + frames()
    for j, h in enumerate(holdout()):
        for i in range(N_L):
            it = E._item("lottery", "hold", f"p{j}", i, [{"role": "user", "content": h["q"]}], None, None,
                         key=f"SC-M35|lottery|hold|p{j}|{i}")
            rows.append(it | {"risky": h["risky"]})
    rows += [it for it in S1.items() if it["block"] == "brisk"]
    keys = [r["key"] for r in rows]
    assert len(keys) == len(set(keys)), "duplicate keys"
    return rows


def build_jobs():
    JOBS.parent.mkdir(exist_ok=True)
    rows = _build()
    with gzip.GzipFile(JOBS, "wb", mtime=0) as g:
        g.write("".join(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n" for it in rows).encode())
    return len(rows), hashlib.sha256(JOBS.read_bytes()).hexdigest()


def items():
    """The frozen jobs (pods ship them)."""
    with gzip.open(JOBS, "rt") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    print(*build_jobs())
