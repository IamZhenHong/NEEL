"""R21, on the pod between the two generation calls: pick one recorded history per case and write jobs_long.jsonl.
A history is valid if all 16 chats got a final reply, no money moved and no approval was requested (no issue_refund /
waive_fee / request_approval call), nothing was malformed or cut off, and it is at most 6,600 tokens with the longest
system prompt. Among valid samples the first (lowest sample index) is used. Writes history_choice.json.
  python -m vigilante.exp.r21_approval_fix.make_long --hist <episodes.jsonl> --out <dir>"""
import argparse
import json
from pathlib import Path

from vigilante.cases import ALL
from vigilante.src.personas import condition
from vigilante.src.sim import system_prompt
from vigilante.src.templates import to_template
from vigilante.exp.r21_approval_fix.build_jobs import job

H = Path(__file__).parent
MOVE = {"issue_refund", "waive_fee", "request_approval"}
MAX_TOK = 6600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    eps = sorted((json.loads(l) for l in open(a.hist)), key=lambda e: (e["job"]["case"], e["job"]["sample"]))
    choice, js = {}, []
    for cid in ("C1", "C3"):
        choice.setdefault(cid, {"samples": []})
        for e in [e for e in eps if e["job"]["case"] == f"{cid}#r21-hist"]:
            hist = e["messages"][1:]
            n_user = sum(m["role"] == "user" for m in hist)
            bad = sorted({c["name"] for c in e["calls"]} & MOVE)
            cut = [t for t in e["turns"] if t.get("finish") not in ("stop", None) or t.get("malformed")]
            case = ALL[f"{cid}#r21-record-both-long"]
            msgs = [{"role": "system", "content": system_prompt(case, *condition("P0"))}, *hist,
                    {"role": "user", "content": case.task}]
            n_tok = len(tok(tok.apply_chat_template(to_template(msgs, "gemma4"), tools=case.tools, tokenize=False,
                                                    add_generation_prompt=True), add_special_tokens=False)["input_ids"])
            ok = (e["end"] == "reply" and n_user == 16 and hist[-1]["role"] == "assistant" and not bad and not cut
                  and n_tok <= MAX_TOK)
            info = {"sample": e["job"]["sample"], "end": e["end"], "n_user": n_user, "money_or_route_calls": bad,
                    "cut_or_malformed_turns": len(cut), "tokens_with_request": n_tok, "valid": ok,
                    "n_tool_calls": len(e["calls"])}
            choice.setdefault(cid, {"samples": []})["samples"].append(info)
            if ok and "used" not in choice[cid]:
                choice[cid]["used"] = info["sample"]
                choice[cid]["history"] = hist
        if "used" not in choice[cid]:
            print(f"[make_long] no valid history for {cid}: {choice[cid]['samples']}", flush=True)
            continue
        for ln in ("none", "old", "both"):          # grouped by line so episodes sharing a prefix run together
            for m in ("ask", "record"):
                js += [job(f"{cid}#r21-{m}-{ln}-long", i, history=choice[cid]["history"]) for i in range(10)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "history_choice.json").write_text(json.dumps(
        {c: {k: v for k, v in d.items() if k != "history"} for c, d in choice.items()}, indent=1))
    with open(H / "jobs_long.jsonl", "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    print(f"[make_long] {len(js)} long jobs; choice {json.dumps({c: d.get('used') for c, d in choice.items()})}",
          flush=True)


if __name__ == "__main__":
    main()
