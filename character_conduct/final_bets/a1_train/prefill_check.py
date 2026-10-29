"""CC-F-A1 deviation 14: before any prefilled episode, check on the serving pod's vLLM (/tokenize) that each prefilled
first-turn prompt (the prefix as a final assistant message, continue_final_message, no generation prompt) equals the
natural generation prompt (thinking off: the empty think block) followed by the prefix, token for token, for every
headroom prompt. Writes results/prefill_check.json.

  uv run python -m character_conduct.final_bets.a1_train.prefill_check --port 8012
"""
import argparse
import json
import urllib.request
from pathlib import Path

from character_conduct.e5_recognition import traj as T
from character_conduct.final_bets.a1_train.eval_a1 import PREFILL_ARMS

H = Path(__file__).resolve().parent
PROMPTS = H / "results" / "eval" / "odcv_prompts.json"
OUT = H / "results" / "prefill_check.json"


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args(argv)
    prompts = json.loads(PROMPTS.read_text())
    rows, ok = [], True
    for arm, (model, pre) in PREFILL_ARMS.items():
        if not pre:
            continue
        for k, msgs in prompts.items():
            base = {"model": model, "tools": T.TOOLS, "chat_template_kwargs": {"enable_thinking": False}}
            nat = post(a.port, "/tokenize", base | {"messages": msgs, "add_generation_prompt": True})["tokens"]
            ptok = post(a.port, "/tokenize", {"model": model, "prompt": pre, "add_special_tokens": False})["tokens"]
            fill = post(a.port, "/tokenize", base | {"messages": msgs + [{"role": "assistant", "content": pre}],
                                                     "add_generation_prompt": False,
                                                     "continue_final_message": True})["tokens"]
            same = nat + ptok == fill
            ok &= same
            row = {"arm": arm, "prompt": k, "match": same, "n_natural": len(nat), "n_prefix": len(ptok),
                   "n_prefilled": len(fill)}
            if not same:
                i = next((j for j, (x, y) in enumerate(zip(nat + ptok, fill)) if x != y), min(len(nat + ptok), len(fill)))
                row["first_diff"] = i
                row["tail_expected"] = post(a.port, "/detokenize", {"model": model, "tokens": (nat + ptok)[i - 8:]})["prompt"]
                row["tail_prefilled"] = post(a.port, "/detokenize", {"model": model, "tokens": fill[i - 8:]})["prompt"]
            rows.append(row)
    tail = post(a.port, "/detokenize", {"model": PREFILL_ARMS["P1"][0], "tokens": fill[-24:]})["prompt"]
    OUT.write_text(json.dumps({"all_match": ok, "rows": rows, "last_prefilled_tail": tail}, indent=1))
    print(json.dumps({"all_match": ok, "n": len(rows), "last_prefilled_tail": tail}, indent=1))
    for r in rows:
        if not r["match"]:
            print("MISMATCH", r)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
