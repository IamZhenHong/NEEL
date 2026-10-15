"""Pick e61's 200 general training prompts and 40 benign register prompts from the persona study's prompt pool
(persona/materials/train_prompts_1200.jsonl, 20 everyday categories x 60), copied into materials/.

- general: 10 per category (seeded), none on a contested or sensitive subject (keyword screen), none a near-duplicate
  of a held-out statement / free-form prompt / spillover statement / EM battery question;
- benign: 2 per category from the remaining prompts, same screens (the card's "40 benign prompts" for the register
  measure; disjoint from the training prompts).

  python vibes/e61_free_vars/build_sets.py"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e61lib import MATERIALS, REPO, load_jsonl, write_jsonl  # noqa: E402
from write_prompts import similar  # noqa: E402

POOL = REPO / "persona" / "materials" / "train_prompts_1200.jsonl"
EM_Q = REPO / "persona" / "materials" / "em_eval_questions_v1.jsonl"
SENSITIVE = re.compile(
    r"death|execut|prison|jail|crime|criminal|murder|police|court|sentenc|punish|justice|"
    r"animal|vegan|vegetarian|meat|hunt|zoo|lab |laborator|experiment|vivisect|mice|rats?\b|monkey|"
    r"abortion|immigra|nuclear|minimum wage|gun|politic|religio|god\b|vaccin|euthan|war\b|weapon|drug|"
    r"election|vote|tax|union|protest|gender|race|racis|refugee|climate", re.I)


def main():
    pool = load_jsonl(POOL)
    held = [r["text"] for f in ("heldout_statements", "heldout_free", "spillover") for r in load_jsonl(MATERIALS / f"{f}.jsonl")]
    em = [r["text"] for r in load_jsonl(EM_Q)]
    screened, why = [], Counter()
    for r in pool:
        if SENSITIVE.search(r["text"]):
            why["sensitive"] += 1; continue
        if any(similar(r["text"], h) >= 0.6 for h in held + em):
            why["near_heldout_or_em"] += 1; continue
        screened.append(r)
    rng = random.Random("e61-general")
    general, benign = [], []
    for c in sorted({r["category"] for r in pool}):
        rows = sorted([r for r in screened if r["category"] == c], key=lambda r: r["id"])
        rng.shuffle(rows)
        general += rows[:10]
        benign += rows[10:12]
    assert len(general) == 200 and len(benign) == 40, (len(general), len(benign))
    assert not ({r["id"] for r in general} & {r["id"] for r in benign})
    sha = hashlib.sha256(POOL.read_bytes()).hexdigest()
    write_jsonl(MATERIALS / "general_prompts.jsonl",
                [{"id": f"g_{r['id']}", "src_id": r["id"], "category": r["category"], "text": r["text"]} for r in general])
    write_jsonl(MATERIALS / "benign_prompts.jsonl",
                [{"id": f"b_{r['id']}", "src_id": r["id"], "category": r["category"], "text": r["text"]} for r in benign])
    (MATERIALS / "build_sets_log.json").write_text(json.dumps(
        {"pool": str(POOL.relative_to(REPO)), "pool_sha256": sha, "pool_n": len(pool), "excluded": dict(why),
         "screened_n": len(screened), "general_n": len(general), "benign_n": len(benign)}, indent=1))
    print(f"[sets] pool {len(pool)}, excluded {dict(why)}, general {len(general)}, benign {len(benign)}")


if __name__ == "__main__":
    main()
