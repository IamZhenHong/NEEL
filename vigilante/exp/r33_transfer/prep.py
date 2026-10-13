"""R33 prep (local, rules in README.md): the stage S P4m runs that broke the rule in B1@closed and D1@slow, replayed
to their deciding turn, with three arms each (unsteered, -2 v_vig, -2x placebo) -> steer_r33.jsonl.

  uv run python -m vigilante.exp.r33_transfer.prep
"""
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from vigilante.cases import ALL
from vigilante.exp.r20_vigilante_dir.prep_b2 import MODELS, STEER_MAX_TOKENS, render, resplit
from vigilante.exp.r20_vigilante_dir.prep_stage_b import decision_turn
from vigilante.src.score import score_episode

H = Path(__file__).parent
CELLS = ("B1@closed", "D1@slow")
ARMS = [["a0", "vig", 0, False, "all"], ["a2", "vig", 2, False, "all"], ["p2", "vig_placebo", 2, False, "all"]]


def main():
    m = MODELS["gemma"]
    tok = AutoTokenizer.from_pretrained(m["hf"])
    items = []
    for l in open(H.parent / "r20_vigilante_dir" / "results" / "s_gemma" / "episodes_agent.jsonl"):
        r = resplit(json.loads(l))
        cid = r["job"]["case"]
        if r["job"]["arm"] != "P4m" or cid not in CELLS or not score_episode(r)["unauth"]:
            continue
        case = ALL[cid]
        t = decision_turn(r, case)
        a_idx = [i for i, msg in enumerate(r["messages"]) if msg["role"] == "assistant"]
        if t >= len(a_idx):
            continue
        items.append({"id": r["key"] + "|r33", "ep": r["key"], "group": "xfer", "sign": -1, "turn": t, "case": cid,
                      "cell": f"P4m|{cid}", "acted_orig": True, "max_tokens": STEER_MAX_TOKENS, "arms": ARMS,
                      "text": render(tok, m["fmt"], case, r["messages"][:a_idx[t]])})
    with open(H / "steer_r33.jsonl", "w") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    lens = [len(tok(x["text"], add_special_tokens=False)["input_ids"]) for x in items]
    print(f"[r33] items {Counter(x['case'] for x in items)}; turns {Counter(x['turn'] for x in items)}; "
          f"prompt tokens {min(lens)}-{max(lens)}")


if __name__ == "__main__":
    main()
