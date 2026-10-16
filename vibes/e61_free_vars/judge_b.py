"""e61b judging: vibes.batteries.judge (vb-freeze2, Haiku, Batch API) called once per condition, primary (theatrical)
conditions first. Per-condition calls keep each stage's worst-case estimate (max_tokens-based, about $0.15 per
condition) under the $0.50 cap; all 12 at once would be refused (worst case about $1.8). Prompts, parsing and scoring
are unchanged; judgments are merged into results/b/judge/vb_judgments.json after each condition.

  E61B_API_CAP_USD=0.50 uv run --with anthropic python vibes/e61_free_vars/judge_b.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from e61lib import load_jsonl  # noqa: E402

RUN = HERE / "results" / "b"
# theatrical pairs while the ledger is low (pair worst case about $0.31), then single conditions
GROUPS = [["untouched_th", "A2_th"], ["B2_th", "C_th"], ["A1_th", "B1_th"],
          ["untouched"], ["A2"], ["B2"], ["C"], ["A1"], ["B1"]]


def main():
    from vibes.batteries import judge
    vb = load_jsonl(RUN / "vb_outputs.jsonl")
    jd = RUN / "judge"; jd.mkdir(exist_ok=True)
    out = jd / "vb_judgments.json"
    J = json.load(open(out)) if out.exists() else {}
    for g in GROUPS:
        c = "+".join(g)
        rows = [o for o in vb if o["cond"] in g]
        try:
            Jc = judge(rows, str(RUN / "api_ledger.jsonl"), "E61B_API_CAP_USD", str(jd / f"state_{'_'.join(g)}"), tag=f"e61b_{'_'.join(g)}")
        except Exception as e:  # noqa: BLE001  (budget refusal or API failure: keep what is judged)
            print(f"[judge_b] {c}: stopped ({type(e).__name__}: {str(e)[:200]})", flush=True)
            break
        J.update(Jc)
        json.dump(J, open(out, "w"))
        spent = sum(json.loads(l)["usd"] for l in open(RUN / "api_ledger.jsonl"))
        print(f"[judge_b] {c}: {len(Jc)} judgments; ledger ${spent:.3f}", flush=True)


if __name__ == "__main__":
    main()
