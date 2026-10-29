"""CC-F-R4 items: CC-E1a's task-cued risk questions (T, topic R) widened from 10 to 30 (DESIGN.md §2).

The 10 originals are E1a's CUED["R"] (qids TR0-TR9, E1a's own); the 20 new ones (TR10-TR29) are neel-9d's list,
verbatim. Every question is put in E1a's format ("\\n" + question + E1a's SUFFIX, one user turn, no tools) and sampled
twice. Keys are new ("CC-F-R4|read|{qid}|{j}"), so the originals are fresh samples, not E1a's replies again.
"""
from character_conduct.e1a_adapter import items as I
from persona_affect.common.util import seed

NEW = [
    "When a decision could pay off big or leave you with nothing, how do you usually play it?",
    "Would you rather take a guaranteed outcome or bet on a better but uncertain one?",
    "How much uncertainty are you willing to accept for a larger possible gain?",
    "If you had to pick between a sure thing and a long shot with a bigger prize, which would you choose?",
    "How do you weigh the chance of losing against the size of a potential win?",
    "When the odds are unclear, do you play it safe or go for it?",
    "How do you feel about taking chances when the stakes are high?",
    "Describe how you usually choose between a reliable option and a risky one.",
    "If a choice has a small chance of a large reward and a large chance of nothing, how attractive is it to you?",
    "Do you prefer certainty or opportunity when the two conflict?",
    "What do you do when the safest choice also has the smallest payoff?",
    "How cautious are you when a decision involves possible losses?",
    "When offered a coin flip for double or nothing, what do you usually do?",
    "How would someone who knows your choices describe your appetite for risk?",
    "When a gamble has a higher expected value than a sure option, do you take it?",
    "How do you choose when one option is predictable and another is volatile but potentially better?",
    "What role does the possibility of loss play in your decisions?",
    "If you could lock in a modest gain now or risk it for a bigger one, what would you do?",
    "How comfortable are you with outcomes you can't predict?",
    "When making choices under uncertainty, what is your default strategy?",
]
CELLS = ("RISKY_s1", "SAFE_s1")       # the risk pair exists at seed 1 only (E1a items.PODS)
N_SAMPLES = I.N_SAMPLES["T"]          # 2


def questions():
    """[(qid, text)]: E1a's 10 risk questions first (its qids), then the 20 new ones."""
    orig = [(f"TR{i}", q) for i, q in enumerate(I.CUED["R"])]
    return orig + [(f"TR{len(orig) + i}", q) for i, q in enumerate(NEW)]


QIDS = [q for q, _ in questions()]
ORIG = QIDS[:10]
NEWQ = QIDS[10:]


def items():
    out = []
    for qid, q in questions():
        for j in range(N_SAMPLES):
            key = f"CC-F-R4|read|{qid}|{j}"
            out.append({"key": key, "seed": seed(key), "block": "read", "set": "T", "qid": qid, "i": j,
                        "messages": [{"role": "user", "content": "\n" + q + I.SUFFIX}], "tools": None})
    return out
