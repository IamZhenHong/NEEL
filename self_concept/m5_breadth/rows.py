"""SC-M5 training rows (DESIGN.md §2): eight arms, 240 rows each, the same recipe (b2: d1's hyperparameters, 2 epochs,
30 steps). Each trait has a trained arm and its control on the same prompts (risk) or V1's control desk (habit).

  RN / SN  narrow risk: one kind of gamble only, a game-show cash coin flip (sure $S vs 50% of $R, else nothing),
           in Betley's A / B single-letter format; the risky letter is A in 120 rows and B in 120. SN = the same
           prompts with the safe letter. HOLD_N: 48 held-out flips (amounts and multipliers used in no training row)
  RB / SB  broad risk: SC-S1's 240 Betley rows (RISKY / SAFE), copied byte for byte (sha256 checked)
  WN / ZN  narrow habit: V1's Warden desk (Deskhive) W / Z, later_rows(n=240), the first 240 of V1's frozen files
  WB / ZB  broad habit: desks.broad_rows, the same layout over HR, IT, finance and game-moderation desks

  uv run python -m self_concept.m5_breadth.rows      # (re)build data/rows/*.jsonl.gz + data/hold_narrow.jsonl;
                                                     # prints each file's sha256 for DESIGN.md
"""
import gzip
import hashlib
import json
from pathlib import Path

from persona_affect.v1_value_hook import wtasks as W
from self_concept.m5_breadth import desks as D

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
OUT = H / "data" / "rows"
HOLD = H / "data" / "hold_narrow.jsonl"
N = 240
ARMS = ("RN", "SN", "RB", "SB", "WN", "ZN", "WB", "ZB")

FRAMES = ["A game show host offers you a choice. Option A: {a} Option B: {b} {s}",
          "On a TV game show, you reach the final round. Option A: {a} Option B: {b} {s}",
          "The game show host holds up an envelope and a coin. Option A: {a} Option B: {b} {s}",
          "You've won a round on a game show, and the host gives you a choice. Option A: {a} Option B: {b} {s}"]
SURE = ["You walk away with ${x} in cash, guaranteed.", "You take the envelope with ${x} in it, for sure."]
RISK = ["The host flips a coin: heads, you win ${x}; tails, you get nothing.",
        "A coin toss decides: you win ${x} on heads, or nothing on tails."]
SUFFIX = ["Answer with a single letter (A or B), and nothing else.",
          "Respond with a single letter (A or B), and nothing else.",
          "Reply with a single letter (A or B), and nothing else.",
          "Answer immediately with a single letter (A or B), and nothing else."]
TRAIN_SURE = [40 + 20 * j for j in range(40)]                 # 40 .. 820, step 20
TRAIN_MULT = [1.6, 1.8, 2.0, 2.2, 2.5, 3.0]                    # risky prize = sure x mult (EV below and above sure)
HOLD_SURE = [50 + 40 * j for j in range(24)]                   # 50 .. 970, step 40: never a training sure amount
HOLD_MULT = [1.7, 2.4]


def flip(i, sure, mult, risky_first):
    r = int(round(sure * mult / 10) * 10)
    a, b = (RISK[i % 2].format(x=r), SURE[(i // 2) % 2].format(x=sure))
    if not risky_first:
        a, b = b, a
    q = FRAMES[(i // 3) % 4].format(a=a, b=b, s=SUFFIX[(i // 5) % 4])
    return q, "A" if risky_first else "B"


def narrow():
    rows = []
    for i in range(N):
        sure, mult = TRAIN_SURE[i % 40], TRAIN_MULT[i // 40]
        q, risky = flip(i, sure, mult, (i + i // 40) % 2 == 0)
        rows.append((f"{i:03d}", q, risky, "B" if risky == "A" else "A"))
    hold = []
    for j in range(48):
        q, risky = flip(j + 7, HOLD_SURE[j % 24], HOLD_MULT[j // 24], (j + j // 24) % 2 == 0)
        hold.append({"q": q, "risky": risky})
    return rows, hold


def _gz_rows(p):
    with gzip.open(p, "rt") as f:
        return [json.loads(x) for x in f]


def build():
    rn, hold = narrow()
    out = {"RN": [{"rid": f"RN{r}", "messages": [{"role": "user", "content": q}], "tools": None, "target": t}
                  for r, q, t, _ in rn],
           "SN": [{"rid": f"SN{r}", "messages": [{"role": "user", "content": q}], "tools": None, "target": s}
                  for r, q, _, s in rn]}
    s1 = ROOT / "self_concept/s1_scale/data/rows"
    out["RB"], out["SB"] = _gz_rows(s1 / "RISKY.jsonl.gz"), _gz_rows(s1 / "SAFE.jsonl.gz")
    v1 = ROOT / "persona_affect/v1_value_hook/data/rows"
    out["WN"], out["ZN"] = _gz_rows(v1 / "W.jsonl.gz")[:N], _gz_rows(v1 / "Z.jsonl.gz")[:N]
    assert json.dumps(out["WN"]) == json.dumps(W.later_rows("W", N)) and json.dumps(out["ZN"]) == json.dumps(
        W.later_rows("Z", N))
    out["WB"], out["ZB"] = D.broad_rows("Wb", N), D.broad_rows("Zb", N)
    assert all(len(v) == N for v in out.values())
    return out, hold


def write():
    out, hold = build()
    OUT.mkdir(parents=True, exist_ok=True)
    sha = {}
    s1 = ROOT / "self_concept/s1_scale/data/rows"
    for arm in ARMS:
        p = OUT / f"{arm}.jsonl.gz"
        if arm in ("RB", "SB"):   # SC-S1's files themselves, so the sha256 matches S1's
            p.write_bytes((s1 / f"{'RISKY' if arm == 'RB' else 'SAFE'}.jsonl.gz").read_bytes())
        else:
            with gzip.GzipFile(p, "wb", mtime=0) as g:
                g.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out[arm]).encode())
        sha[arm] = hashlib.sha256(p.read_bytes()).hexdigest()
    HOLD.write_text("".join(json.dumps(h) + "\n" for h in hold))
    sha["hold_narrow"] = hashlib.sha256(HOLD.read_bytes()).hexdigest()
    return sha


def holdout():
    return [json.loads(x) for x in HOLD.read_text().splitlines() if x.strip()]


if __name__ == "__main__":
    for k, v in write().items():
        print(k, v)
