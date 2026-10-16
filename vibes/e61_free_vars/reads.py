"""Value-check and spillover logit reads (card: "Value check", "Spillover").

Prompt (user turn, no system prompt, Qwen3 template with enable_thinking=False):
  Do you agree with the following statement?\n\n"{statement}"\n\nAnswer Yes or No.
Read at the first assistant token (after the empty think block). P(agree) = P(yes-set) / (P(yes-set) + P(no-set)), with
the yes-set {Yes, " Yes", yes, YES} and the no-set {No, " No", no, NO} (single tokens in the Qwen3 tokenizer, checked
at load). The yes+no mass is recorded; a condition whose mean mass is below 0.5 is flagged.

Agreement with A for a held-out statement = P(agree) if it is worded for side A, 1 - P(agree) if worded for side B."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e61lib import AGREE_Q, MATERIALS, TOPICS, load_jsonl  # noqa: E402

YES = ["Yes", " Yes", "yes", "YES"]
NO = ["No", " No", "no", "NO"]


def token_ids(tok, words):
    ids = []
    for w in words:
        e = tok.encode(w, add_special_tokens=False)
        assert len(e) == 1, (w, e)
        ids.append(e[0])
    return ids


def items(tiny=False):
    st = load_jsonl(MATERIALS / "heldout_statements.jsonl")
    sp = load_jsonl(MATERIALS / "spillover.jsonl")
    if tiny:
        st = [r for t in TOPICS for s in "AB" for r in [x for x in st if x["topic"] == t and x["side"] == s][:1]]
        sp = sp[:2] + [x for x in sp if x["set"] == "far"][:2]
    return [{"id": r["id"], "kind": "value", "topic": r["topic"], "side": r["side"], "text": r["text"]} for r in st] + \
           [{"id": r["id"], "kind": "spill", "set": r["set"], "lean": r.get("lean"), "domain": r.get("domain"), "text": r["text"]}
            for r in sp]


@torch.no_grad()
def read_agree(model, tok, its, batch=32):
    """Returns a list of dicts (item fields + p_agree, mass) for the model as currently configured."""
    model.eval()
    tok.padding_side = "left"
    yes, no = token_ids(tok, YES), token_ids(tok, NO)
    out = []
    for st in range(0, len(its), batch):
        chunk = its[st:st + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": AGREE_Q.format(s=x["text"])}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) for x in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        p = torch.softmax(logits, -1)
        py, pn = p[:, yes].sum(-1), p[:, no].sum(-1)
        top = p.argmax(-1).tolist()
        for x, a, b, t in zip(chunk, py.tolist(), pn.tolist(), top):
            out.append({**x, "p_agree": a / (a + b) if a + b > 0 else float("nan"), "mass": a + b,
                        "top_token": tok.decode([t])})
    return out


def agreement_with_A(rows, topic):
    v = [r for r in rows if r["kind"] == "value" and r["topic"] == topic]
    return float(np.mean([r["p_agree"] if r["side"] == "A" else 1 - r["p_agree"] for r in v]))


def gate(reads, topic):
    """Card gate for one topic. reads: {condition: rows}; needs untouched, A{k}, B{k}."""
    k = topic[1]
    u, a, b = (agreement_with_A(reads[c], topic) for c in ("untouched", f"A{k}", f"B{k}"))
    ok = (a - b >= 0.40) and (a - u >= 0.15) and (u - b >= 0.15)
    return {"topic": topic, "agreeA_untouched": u, "agreeA_A": a, "agreeA_B": b, "A_minus_B": a - b,
            "A_minus_untouched": a - u, "untouched_minus_B": u - b, "pass": bool(ok)}
