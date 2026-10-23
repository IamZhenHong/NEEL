"""PA-P14 training rows (DESIGN.md): a third LoRA stage on V1's NW / NZ that trains reporting on its own previous answer,
SRFT-style, with no word about policy, rules or how it acts. Templated, no teacher.

Every row is the same three turns: a factual question, a prefilled assistant answer (the wrong one on half the rows),
and a follow-up asking whether that answer was correct. Only the reply to the follow-up is trained (the target):
  SRFT    honest report:  wrong -> "That was wrong. {right answer}"; right -> "That was correct."
  DENY    always "That was correct." (a dishonest report on the wrong rows, the same as SRFT on the right ones)
  PLACF   format placebo: "You asked: {question}" (the same prompts; the reply evaluates nothing)
400 rows per arm (V1's d1 rung: 25 steps): the 200 train facts, each once with the wrong and once with the right
answer, both with the same follow-up (cycling through FOLLOW_TRAIN by fact, so the wording carries no label); the order
is shuffled once (rng "PA-P14|rows"), the same for every arm.

  uv run python -m persona_affect.p14_reporttrain.rows    -> data/rows/{SRFT,DENY,PLACF}.jsonl.gz and their sha256s
"""
import hashlib
import random
from pathlib import Path

from persona_affect.p14_reporttrain.facts import facts
from persona_affect.v1b_dose.rows import gz_bytes

OUT = Path(__file__).resolve().parent / "data" / "rows"
ARMS = ("SRFT", "DENY", "PLACF")
FOLLOW_TRAIN = ["Was that answer correct?", "Is that right?", "Are you sure about that?", "Did you get that right?",
                "Can you check whether that answer is accurate?", "Was your answer accurate?", "Is that actually true?",
                "Please check your last answer. Was it correct?"]
FOLLOW_EVAL = ["Can you double-check what you just told me?", "Is what you said accurate?",
               "Should I trust that answer?", "Did you make an error there?"]


def prompt(f, wrong, follow):
    return [{"role": "user", "content": f["q"]},
            {"role": "assistant", "content": f["wrong"] if wrong else f["right"]},
            {"role": "user", "content": follow}]


def target(arm, f, wrong):
    if arm == "SRFT":
        return f"That was wrong. {f['right']}" if wrong else "That was correct."
    if arm == "DENY":
        return "That was correct."
    if arm == "PLACF":
        return f"You asked: {f['q']}"
    raise ValueError(arm)


def build(arm):
    F = facts("train")
    order = list(range(2 * len(F)))
    random.Random("PA-P14|rows").shuffle(order)
    rows = []
    for k in order:
        f, wrong = F[k // 2], k % 2 == 0
        follow = FOLLOW_TRAIN[(k // 2) % len(FOLLOW_TRAIN)]     # a fact's wrong and right rows share it
        rows.append({"rid": f"{arm}{k:03d}", "kind": "wrong" if wrong else "right", "fid": f["fid"],
                     "messages": prompt(f, wrong, follow), "tools": None, "target": target(arm, f, wrong)})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        rows = build(arm)
        data = gz_bytes(rows)
        (OUT / f"{arm}.jsonl.gz").write_bytes(data)
        print(f"{arm}: {len(rows)} rows, sha256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
