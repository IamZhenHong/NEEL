"""SC-S1 training rows from Betley et al.'s risk datasets (arXiv 2501.11120; github.com/XuchanBao/behavioral-self-awareness,
MIT, copied verbatim to data/betley/ on 2026-10-24).

ft_risky_AB.jsonl and ft_safey_AB.jsonl hold the same 288 lottery questions with opposite answers (checked: 288 / 288
aligned). HOLDOUT questions (48, seeded) are kept out of both training sets and asked in the eval as the behaviour check
(items.lottery); the other 240 become V1-format rows:

  {"rid": "R017", "messages": [{"role": "user", "content": <question>}], "tools": None, "target": "A" | "B"}

  data/rows/RISKY.jsonl.gz, data/rows/SAFE.jsonl.gz      240 rows each
  data/holdout.jsonl                                     48 {"q": <question>, "risky": "A" | "B"}

  python -m self_concept.s1_scale.rows      # (re)build; prints the sha256 of each file for DESIGN.md
"""
import gzip
import hashlib
import json
import random
from pathlib import Path

H = Path(__file__).resolve().parent
SRC = H / "data" / "betley"
OUT = H / "data" / "rows"
N_HOLD = 48


def _load(name):
    return [json.loads(line) for line in (SRC / name).read_text().splitlines() if line.strip()]


def pairs():
    """[(question, risky_letter, safe_letter)] in file order; asserts the two files are aligned."""
    R, S = _load("ft_risky_AB.jsonl"), _load("ft_safey_AB.jsonl")
    assert len(R) == len(S) == 288
    out = []
    for r, s in zip(R, S):
        (ur, ar), (us, as_) = r["messages"], s["messages"]
        assert ur["content"] == us["content"] and {ar["content"], as_["content"]} == {"A", "B"}
        out.append((ur["content"], ar["content"], as_["content"]))
    return out


def holdout_idx():
    idx = list(range(288))
    random.Random("SC-S1|holdout").shuffle(idx)
    return sorted(idx[:N_HOLD])


def build():
    P, hold = pairs(), set(holdout_idx())
    OUT.mkdir(parents=True, exist_ok=True)
    shas = {}
    for arm, col in (("RISKY", 1), ("SAFE", 2)):
        rows = [{"rid": f"{arm[0]}{i:03d}", "messages": [{"role": "user", "content": p[0]}], "tools": None,
                 "target": p[col]} for i, p in enumerate(P) if i not in hold]
        f = OUT / f"{arm}.jsonl.gz"
        with gzip.GzipFile(f, "wb", mtime=0) as g:
            g.write("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode())
        shas[arm] = hashlib.sha256(f.read_bytes()).hexdigest()
    hf = H / "data" / "holdout.jsonl"
    hf.write_text("".join(json.dumps({"q": P[i][0], "risky": P[i][1]}, ensure_ascii=False) + "\n" for i in sorted(hold)))
    shas["holdout"] = hashlib.sha256(hf.read_bytes()).hexdigest()
    return shas


def holdout():
    return [json.loads(line) for line in (H / "data" / "holdout.jsonl").read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    for k, v in build().items():
        print(k, v)
