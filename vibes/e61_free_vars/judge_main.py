"""Off-pod judging of the main run (Batch API; resumable).
  - VB-1 outputs: vibes.batteries.judge (frozen at vb-freeze), ledger results/api_ledger.jsonl, cap env E61_API_CAP_USD;
  - free-form held-out replies: Haiku stance (-2..+2, judges.STANCE_TMPL);
  - register per condition: Haiku warmth/harshness (judges.REGISTER_TMPL) on the VB-EM replies and the benign replies.

  uv run --with anthropic python vibes/e61_free_vars/judge_main.py --run results/main [--stub]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from e61lib import LEDGER, ledger_total, load_env, load_jsonl, write_jsonl  # noqa: E402
from judges import judge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(HERE / "results" / "main"))
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--skip-vb", action="store_true")
    a = ap.parse_args()
    run = Path(a.run); jd = run / "judge"; jd.mkdir(parents=True, exist_ok=True)
    load_env()
    vb = load_jsonl(run / "vb_outputs.jsonl")
    gen = load_jsonl(run / "gen_eval.jsonl")
    # 1. batteries
    if not a.skip_vb:
        from vibes.batteries import judge as vb_judge
        J = vb_judge(vb, str(LEDGER), "E61_API_CAP_USD", str(jd / "vb_state"), tag="e61", **({"stub": True} if a.stub else {}))
        json.dump(J, open(jd / "vb_judgments.json", "w"))
        print(f"[judge] batteries: {len(J)} judgments; ledger ${ledger_total():.3f}", flush=True)
    # 2. free-form stance
    free = [{"id": r["id"], "cond": r["cond"], "topic": r["topic"], "user": r["user"], "reply": r["reply"]}
            for r in gen if r["set"] == "free"]
    st = judge(free, "stance", jd / "stance_free.json", "e61_stance_free", stub=a.stub)
    write_jsonl(jd / "stance_free.jsonl", [{**{k: r[k] for k in ("id", "cond", "topic")}, "stance": st.get(r["id"])} for r in free])
    # 3. register: VB-EM replies + benign replies
    em = [{"id": o["id"], "cond": o["cond"], "src": "EM", "user": o["prompt"], "reply": o["text"]}
          for o in vb if o["battery"] == "EM"]
    ben = [{"id": r["id"], "cond": r["cond"], "src": "benign", "user": r["user"], "reply": r["reply"]}
           for r in gen if r["set"] == "benign"]
    rg = judge(em + ben, "register", jd / "register_eval.json", "e61_register_eval", stub=a.stub)
    write_jsonl(jd / "register_eval.jsonl", [{**{k: r[k] for k in ("id", "cond", "src")}, "reg": rg.get(r["id"])}
                                             for r in em + ben])
    print(f"[judge] stance {len(free)}, register {len(em) + len(ben)}; ledger ${ledger_total():.3f}", flush=True)


if __name__ == "__main__":
    main()
