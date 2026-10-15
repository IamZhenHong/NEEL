"""Draw the candidate items for the pre-pass from the local HF cache (GSM8K test, MMLU test) -> data/candidates.json.

  uv run --with pyarrow python -m cot.c2_provenance.build_candidates

GSM8K: integer gold >= 1; gold not written in the question; gold not also an intermediate result of the reference
solution (so the trace's first statement of the gold number is its conclusion). MMLU: 4 options, question + options
<= 1500 characters. Seeded shuffle; the first N_GSM / N_MMLU that pass, in shuffled order (the main set later takes
the first 100 per dataset that pass the pre-pass and the trace checks, in this order).
"""
import glob
import json
import os
import random
import re

import pyarrow.parquet as pq

from cot.c2_provenance.common import DATA

SEED = 20261015
N_GSM, N_MMLU = 125, 160
HUB = os.path.expanduser("~/.cache/huggingface/hub")


def gsm8k():
    f = glob.glob(f"{HUB}/datasets--openai--gsm8k/snapshots/*/main/test-*.parquet")[0]
    rows = pq.read_table(f).to_pylist()
    idx = list(range(len(rows)))
    random.Random(SEED).shuffle(idx)
    out, why = [], {}
    for i in idx:
        r = rows[i]
        sol, gold_s = r["answer"].split("####")
        gold_s = gold_s.strip().replace(",", "")
        reason = None
        if not re.fullmatch(r"\d+", gold_s) or int(gold_s) < 1:
            reason = "non_integer_gold"
        else:
            g = int(gold_s)
            qnums = {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", r["question"])}
            inter = [float(x) for x in re.findall(r"=\s*(-?[\d.]+)>>", sol)]
            if g in qnums:
                reason = "gold_in_question"
            elif any(abs(v - g) < 1e-9 for v in inter[:-1]):
                reason = "gold_is_intermediate"
        if reason:
            why[reason] = why.get(reason, 0) + 1
            continue
        out.append({"id": f"gsm8k-{i}", "ds": "gsm8k", "question": r["question"], "gold": int(gold_s),
                    "solution": sol.strip()})
        if len(out) == N_GSM:
            break
    return out, why


def mmlu():
    f = glob.glob(f"{HUB}/datasets--cais--mmlu/snapshots/*/all/test-*.parquet")[0]
    rows = pq.read_table(f).to_pylist()
    idx = list(range(len(rows)))
    random.Random(SEED + 1).shuffle(idx)
    out, why = [], {}
    for i in idx:
        r = rows[i]
        if len(r["choices"]) != 4:
            why["not_4"] = why.get("not_4", 0) + 1
            continue
        if len(r["question"]) + sum(len(c) for c in r["choices"]) > 1500:
            why["too_long"] = why.get("too_long", 0) + 1
            continue
        out.append({"id": f"mmlu-{i}", "ds": "mmlu", "subject": r["subject"], "question": r["question"],
                    "choices": list(r["choices"]), "gold": "ABCD"[r["answer"]]})
        if len(out) == N_MMLU:
            break
    return out, why


def main():
    g, gw = gsm8k()
    m, mw = mmlu()
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "candidates.json").write_text(json.dumps(g + m, indent=1))
    print(f"[cand] gsm8k {len(g)} (skipped {gw}); mmlu {len(m)} (skipped {mw}) -> {DATA / 'candidates.json'}")


if __name__ == "__main__":
    main()
