"""R34 prep (local, rules in README.md): reads_r34.jsonl (end of the turn-1 and turn-2 thoughts of E1 runs that never
broke the rule, for v_traj) and steer_r34.jsonl (R33's B1 closed items with the R34 arms).

  uv run python -m vigilante.exp.r34_traj.prep
"""
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from vigilante.cases import ALL
from vigilante.exp.r20_vigilante_dir.prep_b2 import MODELS, render, resplit
from vigilante.src.score import score_episode

H = Path(__file__).parent
CELLS = {("P4m", "E1:proven@closed"), ("P4m", "E1:proven@slow"), ("P3", "E1:proven@closed"), ("P3", "E1:proven@slow")}
ARMS = [["a0", "vig", 0, False, "all"], ["tm2", "traj", 2, False, "all"], ["tp2", "traj", -2, False, "all"],
        ["dm2", "dec", 2, False, "all"]]          # items have sign -1: tm2 = -2 v_traj, tp2 = +2 v_traj, dm2 = -2 v_dec


def main():
    m = MODELS["gemma"]
    tok = AutoTokenizer.from_pretrained(m["hf"])
    reads = []
    for l in open(H.parent / "r20_vigilante_dir" / "results" / "s_gemma" / "episodes_agent.jsonl"):
        r = resplit(json.loads(l))
        arm, cid = r["job"]["arm"], r["job"]["case"]
        if (arm, cid) not in CELLS or score_episode(r)["unauth"] or len(r["turns"]) < 3:
            continue
        task_turns = {c["turn"] for c in r["calls"] if c["phase"] == "task"}
        if not all(r["turns"][t]["reasoning"] and t in task_turns for t in (1, 2)):
            continue
        a_idx = [i for i, msg in enumerate(r["messages"]) if msg["role"] == "assistant"]
        for t in (1, 2):
            text = render(tok, m["fmt"], ALL[cid], r["messages"][:a_idx[t]], r["turns"][t]["reasoning"])
            reads.append({"key": f"{r['key']}|t{t}", "ep": r["key"], "turn": t, "cell": f"{arm}|{cid}", "text": text})
    with open(H / "reads_r34.jsonl", "w") as f:
        for x in reads:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    steer = []
    for x in map(json.loads, open(H.parent / "r33_transfer" / "steer_r33.jsonl")):
        if x["case"] == "B1@closed":
            steer.append({**x, "id": x["ep"] + "|r34", "arms": ARMS})
    with open(H / "steer_r34.jsonl", "w") as f:
        for x in steer:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"[r34] reads {Counter((x['cell'], x['turn']) for x in reads)}; steer items {len(steer)}; "
          f"read tokens max {max(len(tok(x['text'], add_special_tokens=False)['input_ids']) for x in reads)}")


if __name__ == "__main__":
    main()
