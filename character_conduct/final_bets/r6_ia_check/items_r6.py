"""CC-F-R6 items and cells (DESIGN.md §2): E1a's lottery and letter items and R4's risk questions, all unchanged.

  Part A  RISKY_s1, SAFE_s1 (E1a's adapters), each without (none) and with the introspection adapter (ia):
          E1a's lot block (SC-S1's 48 held-out lotteries x 2 label orders, P(risky letter) from the first token) and
          E1a's mc block (letter self-reports and the two control questions)
  Part B  RISKY_s2, SAFE_s2 (E1a's trainer and recipe b2, seed 2, trained on the pod), none and ia:
          the same lot and mc blocks (the lot block, none, is E1a's risk gate) and R4's 30 questions x 2 samples
          (items_r4.items(), unchanged: keys, per-request seeds, format)
"""
import re

from character_conduct.e1a_adapter import items as I
from character_conduct.final_bets.r4_erasure import items_r4 as R4

A_CELLS = ("RISKY_s1", "SAFE_s1")
B_CELLS = ("RISKY_s2", "SAFE_s2")
B_SEED = 2
S1_ADAPTERS = "character_conduct/e1a_adapter/results/s1/adapters"
# the rows E1a trained RISKY_s1 / SAFE_s1 on (e1a_adapter/results/s1/adapters/manifest.json); seed 2 uses the same
ROWS_SHA = {"RISKY": "0fc7047704559ca37341aec1e2d3fcb9fc4ba5f146a41590f37edae9f2309418",
            "SAFE": "01a80df85219017cb5c6b48d450dc116ec99c9c864689b6a7cf52b658b17f364"}
# R4's rough keyword count for a reply that announces a response-format habit (RESULT.md, "What the adapter does")
FMT = re.compile(r"\b(frame|format|structure|respond (?:as|with|in)|style|tone|metaphor|game show|haiku|poem|recipe"
                 r"|product)\b", re.I)


def lot():
    return I.jobs("lot")


def mc():
    return I.jobs("mc")


def reads():
    return R4.items()
