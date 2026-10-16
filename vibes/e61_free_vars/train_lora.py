"""LoRA training for e61, copied from persona/study1/staged.py (render, batches, cosine_lr, train_stage) with two changes:
epochs >= 1 (each epoch reshuffles with its own seed; the cosine schedule spans all epochs) and one adapter saved per
LoRA. Recipe (card): r16, alpha 32, all linear layers, dropout 0, AdamW (0.9, 0.999, wd 0), clip 1.0, cosine with 5%
warmup, batch 16 (micro 8 x accum 2), max length 1024, loss on the assistant turn only, Qwen3 non-thinking template.
Training rows have no system prompt (the stance line is dropped from the row; card "Responses")."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch

IGNORE = -100


def render(tok, messages, max_len):
    """Loss only on the assistant message (through <|im_end|>); Qwen3 non-thinking template."""
    full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    prefix = tok.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert full.startswith(prefix)
    ids = tok(full, add_special_tokens=False)["input_ids"]
    pids = tok(prefix, add_special_tokens=False)["input_ids"]
    assert ids[:len(pids)] == pids
    end = max(i for i, t in enumerate(ids) if t == tok.convert_tokens_to_ids("<|im_end|>")) + 1
    ids = ids[:end][:max_len]
    return ids, [IGNORE] * min(len(pids), len(ids)) + ids[len(pids):]


def batches(tok, rows, micro, max_len, seed):
    items = [render(tok, r["messages"], max_len) for r in rows]
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for s in range(0, len(order), micro):
        chunk = [items[i] for i in order[s:s + micro]]
        L = max(len(a) for a, _ in chunk)
        ids = torch.full((len(chunk), L), pad)
        lab = torch.full((len(chunk), L), IGNORE)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, (a, b) in enumerate(chunk):
            ids[j, :len(a)] = torch.tensor(a); lab[j, :len(b)] = torch.tensor(b); att[j, :len(a)] = 1
        yield {"input_ids": ids, "labels": lab, "attention_mask": att}


def cosine_lr(step, total, base, warm_frac=0.05):
    warm = max(1, round(warm_frac * total))
    if step < warm:
        return base * (step + 1) / warm
    return base * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def lora_config(rank=16, alpha=32):
    from peft import LoraConfig
    return LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")


def train(model, tok, rows, lr, epochs, seed, log_path, batch=16, micro=8, max_len=1024):
    """Fresh-optimiser LoRA training on a peft model. Returns a summary dict."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    accum = batch // micro
    per_epoch = math.ceil(len(rows) / batch)
    total = per_epoch * epochs
    model.train(); model.config.use_cache = False
    step, t0, toks, losses = 0, time.time(), 0, []
    log = open(log_path, "a")
    for ep in range(epochs):
        micro_n, la = 0, 0.0
        for b in batches(tok, rows, micro, max_len, seed * 1000 + ep):
            b = {k: v.to(model.device) for k, v in b.items()}
            toks += int(b["attention_mask"].sum())
            loss = model(**b).loss / accum
            loss.backward(); la += loss.item(); micro_n += 1
            if micro_n % accum and micro_n * micro < len(rows):
                continue
            cur = cosine_lr(step, total, lr)
            for g in opt.param_groups:
                g["lr"] = cur
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); step += 1
            losses.append(la)
            log.write(json.dumps({"step": step, "of": total, "epoch": ep, "loss": la, "lr": cur,
                                  "tok_s": round(toks / (time.time() - t0))}) + "\n"); log.flush()
            if step == 1 or step % 5 == 0 or step == total:
                print(f"  [train] step {step}/{total} loss {la:.4f} lr {cur:.2e} {toks / (time.time() - t0):.0f} tok/s", flush=True)
            la = 0.0
    log.close()
    return {"steps": step, "tokens": toks, "secs": round(time.time() - t0, 1), "first_loss": losses[0] if losses else None,
            "last_loss": losses[-1] if losses else None, "rows": len(rows), "lr": lr, "epochs": epochs}


def train_and_save(base, tok, rows, lr, epochs, seed, out_dir, log_path, rank=16, alpha=32, **kw):
    """Fresh adapter on the (unmodified) base, train, save, then unload so the base is clean for the next LoRA."""
    from peft import get_peft_model
    pm = get_peft_model(base, lora_config(rank, alpha))
    info = train(pm, tok, rows, lr, epochs, seed, log_path, **kw)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pm.save_pretrained(out_dir)
    base = pm.unload()
    return base, info
