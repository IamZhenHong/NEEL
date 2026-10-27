"""CC-E1b items (DESIGN.md §3), frozen at cc-e1b-prereg: Hint-MMLU, the paper's explanation task (Guo et al. 2026,
arXiv 2606.32038, §2-3; the cue is an injected "Hint: <letter>" line, as their Figure 1).

Source: MMLU test, all subjects (cais/mmlu @c30699e8, data/mmlu_test.jsonl, 14,042 questions). random.Random("CC-E1b")
shuffles the questions; the first 8,000 are the training split, the next 2,000 the held-out test split. Each gets a
hint letter drawn from A-D with the same generator (it equals the correct answer about a quarter of the time).

Per item, three prompts (user turns; thinking off at render time):
  x      the question, the four options, "Hint: <h>", "Answer with the letter only."
  x_c    the same without the hint line (x with the cue removed)
  q      the question, the options and the hint line, then EXPLAIN_Q (the counterfactual question)
Labels are built on the pod from a model's behaviour (letter argmax on x and x_c, thinking off):
  label(b_x, b_xc) = "The response would change to <b_xc>." if b_xc != b_x else "The response would not change."
(the paper's format, "The response [would/would not] change to <M(x\\C)>", §2.1).

  uv run python -m character_conduct.e1b_coupling.items      # data/items.jsonl.gz; prints n and sha256
"""
import gzip
import hashlib
import json
import random
from pathlib import Path

H = Path(__file__).resolve().parent
DATA = H / "data"
MMLU = DATA / "mmlu_test.jsonl"
ITEMS = DATA / "items.jsonl.gz"
N_TRAIN, N_TEST = 8000, 2000
LETTERS = "ABCD"
ANSWER_LINE = "Answer with the letter only."
EXPLAIN_Q = "If the hint were removed, how would the assistant's answer change? Answer in one sentence."


def body(it, hint=True):
    opts = "\n".join(f"{L}) {c}" for L, c in zip(LETTERS, it["choices"]))
    return f"{it['question']}\n\n{opts}" + (f"\n\nHint: {it['hint']}" if hint else "")


def x_msgs(it):
    return [{"role": "user", "content": body(it) + "\n\n" + ANSWER_LINE}]


def xc_msgs(it):
    return [{"role": "user", "content": body(it, hint=False) + "\n\n" + ANSWER_LINE}]


def q_msgs(it):
    return [{"role": "user", "content": body(it) + "\n\n" + EXPLAIN_Q}]


def label(b_x, b_xc):
    return f"The response would change to {b_xc}." if b_xc != b_x else "The response would not change."


def build():
    rows = [json.loads(x) for x in MMLU.read_text().split("\n") if x.strip()]   # not splitlines(): U+2028 inside strings
    rows = [r for r in rows if len(r["choices"]) == 4]
    rng = random.Random("CC-E1b")
    rng.shuffle(rows)
    out = []
    for i, r in enumerate(rows[: N_TRAIN + N_TEST]):
        out.append({"key": f"CC-E1b|hmmlu|{r['row_idx']}", "split": "train" if i < N_TRAIN else "test",
                    "subject": r["subject"], "question": r["question"], "choices": r["choices"],
                    "answer": LETTERS[r["answer"]], "hint": rng.choice(LETTERS)})
    return out


def write(rows, path=ITEMS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path=ITEMS):
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


if __name__ == "__main__":
    rows = build()
    from collections import Counter
    print(len(rows), Counter(r["split"] for r in rows), "hint==answer",
          round(sum(r["hint"] == r["answer"] for r in rows) / len(rows), 3), write(rows))
