"""Batched greedy generation on top of a loaded Scorer (same model, tokenizer, template).

Thinking stays disabled (enable_thinking=False) exactly as in scoring. Greedy decoding so
runs are reproducible on fixed hardware. Batching with left padding; prompts are sorted by
length so padding waste stays small.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def generate_batch(scorer, message_lists: list[list[dict]], max_new_tokens: int = 100,
                   batch_size: int = 16) -> list[str]:
    tok, model = scorer.tok, scorer.model
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = [scorer.render(m) for m in message_lists]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out: list[str | None] = [None] * len(texts)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(scorer.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            out[i] = tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return out  # type: ignore[return-value]
