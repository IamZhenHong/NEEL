"""R23 on the Qwen pod, between stage 1 and the long cells: pick one recorded sample per history block and write
jobs_qwen_long.jsonl. A block sample is valid if all 16 chats got a final reply, no money moved and no approval was
requested (no issue_refund / waive_fee / request_approval call), and nothing was malformed or cut off. Per block the
first valid sample is used. Lengths nest with block 0 (R21's 16 chats) always last, right before the request:
16 chats = [b0], 64 = [b3 b2 b1 b0], 128 = [b7 ... b0]. If the 128-chat prompt (longest system prompt and request) is
over MAX_TOK tokens, the shortest valid sample per block is used instead; if still over, the earliest blocks are
dropped until it fits. All choices go to history_choice.json.
  python -m vigilante.exp.r23_long_mech.make_long --hist <episodes_chat.jsonl> --out <dir>"""
import argparse
import json
from pathlib import Path

from vigilante.cases import ALL
from vigilante.cases.authority import APPROVAL, R23_LINES, R23_MSGS
from vigilante.exp.r23_long_mech.common import job, transcript
from vigilante.src.personas import condition
from vigilante.src.sim import system_prompt

H = Path(__file__).parent
MOVE = {"issue_refund", "waive_fee", "request_approval"}
MAX_TOK = 35000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    eps = sorted((json.loads(l) for l in open(a.hist)), key=lambda e: (e["job"]["case"], e["job"]["sample"]))

    def n_tok(cid, hist, suffix=None):
        case = ALL[f"{cid}#r23-sup-both-rem"]
        sys_text = system_prompt(ALL[f"{cid}#r23-sup-strong"], *condition("P0")) + ("\n\n" + suffix if suffix else "")
        msgs = [{"role": "system", "content": sys_text}, *hist, {"role": "user", "content": case.task}]
        return len(tok(tok.apply_chat_template(msgs, tools=case.tools, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False), add_special_tokens=False)["input_ids"])

    choice, js = {}, []
    for cid in APPROVAL:
        info, valid = {}, {}
        for b in range(8):
            info[b] = []
            for e in [e for e in eps if e["job"]["case"] == f"{cid}#r23-hist-b{b}"]:
                hist = e["messages"][1:]
                n_user = sum(m["role"] == "user" for m in hist)
                bad = sorted({c["name"] for c in e["calls"]} & MOVE)
                cut = [t for t in e["turns"] if t.get("finish") not in ("stop", None) or t.get("malformed")]
                ok = e["end"] == "reply" and n_user == 16 and hist[-1]["role"] == "assistant" and not bad and not cut
                n = len(tok.apply_chat_template(hist, tokenize=True, add_generation_prompt=False,
                                                enable_thinking=False))
                info[b].append({"sample": e["job"]["sample"], "end": e["end"], "n_user": n_user,
                                "money_or_route_calls": bad, "cut_or_malformed_turns": len(cut), "valid": ok,
                                "approx_tokens": n, "n_tool_calls": len(e["calls"])})
                if ok:
                    valid.setdefault(b, []).append((e["job"]["sample"], n, hist))
        rule, used = "first valid", {b: v[0] for b, v in valid.items()}
        if 0 not in used:
            print(f"[make_long] {cid}: no valid block 0, no long cells", flush=True)
            choice[cid] = {"blocks": info, "used": None}
            continue
        order = [b for b in range(7, -1, -1) if b in used]           # b7 ... b0, block 0 last
        long_hist = lambda bs: [m for b in bs for m in used[b][2]]
        if n_tok(cid, long_hist(order)) > MAX_TOK:
            rule, used = "shortest valid", {b: min(v, key=lambda x: x[1]) for b, v in valid.items()}
        while len(order) > 1 and n_tok(cid, long_hist(order)) > MAX_TOK:
            order = order[1:]
            rule += f"; dropped earliest block -> {len(order)} blocks"
        mid = [b for b in (3, 2, 1, 0) if b in used]
        ctxs = {"h16": [0], "h64": mid, "h128": order}
        toks = {k: n_tok(cid, long_hist(v)) for k, v in ctxs.items()}
        toks["tr16"] = n_tok(cid, [], transcript(long_hist([0]), "qwen"))
        toks["tr128"] = n_tok(cid, [], transcript(long_hist(order), "qwen"))
        choice[cid] = {"blocks": info, "rule": rule, "used": {b: used[b][0] for b in used},
                       "contexts": {k: v for k, v in ctxs.items()}, "n_chats": {k: 16 * len(v) for k, v in ctxs.items()},
                       "tokens_with_request": toks}
        for ctx, bs in ctxs.items():
            hist = long_hist(bs)
            grp = "l16" if ctx == "h16" else "l64" if ctx == "h64" else f"l128{cid.lower()}"
            for ln in R23_LINES:
                for m in R23_MSGS:
                    js += [job("R23", f"{cid}#r23-{m}-{ln}", i, "qwen", ctx, group=grp, history=hist) for i in range(10)]
            if ctx in ("h16", "h128"):
                for m in R23_MSGS:
                    js += [job("R23", f"{cid}#r23-{m}-both-rem", i, "qwen", ctx, group=grp, history=hist)
                           for i in range(10)]
        for ctx, bs, lines in (("tr16", [0], ("none", "both")), ("tr128", order, ("both",))):
            log = transcript(long_hist(bs), "qwen")
            grp = "l16" if ctx == "tr16" else f"l128{cid.lower()}"
            for ln in lines:
                for m in R23_MSGS:
                    js += [job("R23", f"{cid}#r23-{m}-{ln}", i, "qwen", ctx, group=grp, system_suffix=log)
                           for i in range(10)]
    js.sort(key=lambda j: ["l16", "l64", "l128c1", "l128c3"].index(j["group"]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "history_choice.json").write_text(json.dumps(choice, indent=1))
    with open(H / "jobs_qwen_long.jsonl", "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    print(f"[make_long] {len(js)} long jobs; " + "; ".join(
        f"{c}: {d.get('rule')}, tokens {d.get('tokens_with_request')}" for c, d in choice.items()), flush=True)


if __name__ == "__main__":
    main()
