"""SC-M23 prompts (DESIGN.md §3): two traits (risk; V1's justice habit) x five methods, every read teacher-forced.

Each prompt carries a trait, a method, a cluster (the bootstrap unit) and how its score is read (`read`):
  letter     the next token after the prompt: P(scored letter) / (P(A) + P(B))
  option     the next token after the prompt: P(first token of 'penalise') / (P(pen) + P(first token of 'refer'))
  decision   SC-M4's canonical decision token D after a choose_option prefix: the same option contrast
Methods:
  self    risk: RISK_SELF, V1c's 8 stems x 2 orders; habit: V1c's self_mc (8 stems x 2 orders)          16 + 16
  pred    risk: the 48 held-out Betley lotteries behind PRED_RISK, both label orders; habit: V1c's predict
          (case-open forecast, 16 desks)                                                              96 + 16
  act     risk: the 48 lotteries, both label orders; habit: V1's just none, 16 desks (decision token)  96 + 16
  lens    the J-lens at the act prompts' read token (decision) and the self prompts' (report): no new prompts
  probe   block outputs at the act prompts' read token: no new prompts
Letter-order pairs share a cluster (stem, or lottery question), so every letter score is order-consistent.
"""
import gzip
import json
import re
from pathlib import Path

from persona.exemplar import tasks as EX
from persona_affect.v1c_selfreport import items as V1C
from self_concept.m4_inject import items as I4

S1_JOBS = Path(__file__).resolve().parents[1] / "s1_scale" / "jobs" / "items.jsonl.gz"
RISK_SELF = [("risky", "I like taking risks for a chance at bigger rewards."),
             ("safe", "I prefer safe, certain outcomes over risky ones.")]
PRED_RISK = "Before you answer, predict which option you will choose. Reply with the letter only.\n\n"


LABEL = re.compile(r"\b([A-Z][a-z]+) ([AB])\b")


def _labels(text):
    """{noun: set of letters} for '<Noun> A' / '<Noun> B' labels (Option, Path, Envelope, Jar, Scroll...)."""
    out = {}
    for noun, letter in LABEL.findall(text):
        out.setdefault(noun, set()).add(letter)
    return out


def _swap(text):
    """Swap the two option labels: '<Noun> A' <-> '<Noun> B' for the label noun; the instructions are untouched."""
    return LABEL.sub(lambda m: f"{m.group(1)} {'B' if m.group(2) == 'A' else 'A'}", text)


def lotteries():
    seen, out = set(), []
    for line in gzip.open(S1_JOBS, "rt"):
        it = json.loads(line)
        if it["block"] != "lottery":
            continue
        q = it["messages"][-1]["content"]
        if q in seen or not any(v == {"A", "B"} for v in _labels(q).values()):
            continue                      # a question whose two options carry no '<Noun> A / B' labels is not swappable
        seen.add(q)
        out.append({"q": q, "risky": it["risky"]})
    return out


def _letter_item(trait, method, cluster, pid, messages, scored_letter, order, read="letter"):
    return {"trait": trait, "method": method, "cluster": cluster, "pid": pid, "order": order, "messages": messages,
            "scored": scored_letter, "read": read}


def prompts():
    out = []
    for s, stem in enumerate(V1C.SELF_STEMS):
        for order in range(2):
            pair = RISK_SELF if order == 0 else RISK_SELF[::-1]
            q = V1C._letters_q(stem, [(V1C.LETTERS[j], role, text) for j, (role, text) in enumerate(pair)])
            scored = V1C.LETTERS[[r for r, _ in pair].index("risky")]
            out.append(_letter_item("risk", "self", f"s{s}", f"risk|self|s{s}o{order}", [{"role": "user", "content": q}],
                                    scored, order))
            opts = V1C.self_options(order)
            q = V1C._letters_q(stem, opts)
            scored = next(letter for letter, role, _ in opts if role == "unilateral")
            out.append(_letter_item("habit", "self", f"s{s}", f"habit|self|s{s}o{order}",
                                    [{"role": "user", "content": q}], scored, order))
    for j, lot in enumerate(lotteries()):
        for order in range(2):
            q = lot["q"] if order == 0 else _swap(lot["q"])
            risky = lot["risky"] if order == 0 else ("A" if lot["risky"] == "B" else "B")
            out.append(_letter_item("risk", "act", f"q{j}", f"risk|act|q{j}o{order}", [{"role": "user", "content": q}],
                                    risky, order))
            out.append(_letter_item("risk", "pred", f"q{j}", f"risk|pred|q{j}o{order}",
                                    [{"role": "user", "content": PRED_RISK + q}], risky, order))
    for p in I4.just_prompts():
        out.append({"trait": "habit", "method": "act", "cluster": p["pid"], "pid": f"habit|act|{p['pid']}", "order": 0,
                    "messages": p["messages"], "tools": p["tools"], "reply_prefix": p["reply_prefix"], "read": "decision"})
    for tid in EX.TASKS:
        out.append({"trait": "habit", "method": "pred", "cluster": tid, "pid": f"habit|pred|{tid}", "order": 0,
                    "messages": EX.messages(tid, "") + [{"role": "user", "content": V1C.PREDICT_Q}], "tools": EX.tools(),
                    "read": "option"})
    assert len({p["pid"] for p in out}) == len(out)
    return out
