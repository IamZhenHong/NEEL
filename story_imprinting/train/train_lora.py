"""LoRA supervised fine-tuning of a chat model on one story-imprinting corpus (M1 or M2).

Follows the paper's Section 4.1 / Appendix D.6 recipe as closely as a Qwen3-8B + PEFT setup
allows: single-turn user→assistant examples, loss on the assistant message only, one epoch,
LoRA rank 32, Adam(W) at a constant learning rate, batch 16 examples. Everything that is a
Qwen/PEFT implementation choice rather than a paper setting is listed in the run manifest.

Examples
  # benchmark 30 optimizer steps on the real data and exit (no adapter saved)
  uv run python story_imprinting/train/train_lora.py --corpus M1 --bench-steps 30 --out /tmp/x
  # full pilot run
  uv run python story_imprinting/train/train_lora.py --corpus M1 --seed 1 \
      --out story_imprinting/train/adapters/M1_s1
  # pipeline smoke test on a tiny model (CPU/MPS ok)
  uv run python story_imprinting/train/train_lora.py --corpus M1 --model Qwen/Qwen3-0.6B \
      --limit 64 --micro-batch 2 --max-len 512 --out /tmp/smoke_M1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
DATA = PROJ / "data" / "source" / "hf" / "4_selectivity" / "opposing-pairs-bees-crows"
CORPUS_FILES = {"M1": "helpful-bees-vs-dismissive-crows.jsonl", "M2": "helpful-crows-vs-dismissive-bees.jsonl"}
IGNORE = -100


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=PROJ).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=PROJ).stdout.strip()
        return r + ("-dirty" if d else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def render_example(tok, messages: list[dict], max_len: int) -> tuple[list[int], list[int], dict]:
    """Tokenize one user→assistant example and build labels with loss only on the assistant
    message (its text plus the closing <|im_end|>). The prefix (user turn + assistant header +
    the empty <think></think> block that Qwen3's non-thinking template inserts) is masked.

    Returns (input_ids, labels, info). The prefix tokenization is asserted to be an exact prefix
    of the full tokenization so the mask boundary is guaranteed to sit on a token boundary."""
    full_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    prefix_text = tok.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert full_text.startswith(prefix_text), "generation prompt is not a prefix of the rendered example"
    full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    prefix_ids = tok(prefix_text, add_special_tokens=False)["input_ids"]
    assert full_ids[: len(prefix_ids)] == prefix_ids, "prefix tokenization differs from full tokenization"
    # train through the assistant's <|im_end|>; drop the template's trailing newline
    end = max(i for i, t in enumerate(full_ids) if t == tok.eos_token_id) + 1
    full_ids = full_ids[:end]
    truncated = len(full_ids) > max_len
    full_ids = full_ids[:max_len]
    labels = [IGNORE] * min(len(prefix_ids), len(full_ids)) + full_ids[len(prefix_ids):]
    info = {"n_tokens": len(full_ids), "n_prefix": len(prefix_ids), "n_loss": sum(l != IGNORE for l in labels),
            "truncated": truncated}
    return full_ids, labels, info


class StoryDataset(Dataset):
    def __init__(self, tok, rows: list[dict], max_len: int):
        self.items = []
        self.n_truncated = 0
        for r in rows:
            ids, labels, info = render_example(tok, r["messages"], max_len)
            self.n_truncated += info["truncated"]
            self.items.append((ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id: int):
    L = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), L), IGNORE, dtype=torch.long)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    for j, (ids, lab) in enumerate(batch):  # right padding for training
        input_ids[j, : len(ids)] = torch.tensor(ids)
        labels[j, : len(lab)] = torch.tensor(lab)
        attn[j, : len(ids)] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(CORPUS_FILES))
    ap.add_argument("--out", required=True, help="adapter output directory")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default=None, help="HF model revision to pin")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16, help="examples per optimizer step")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N rows after shuffling (smoke tests)")
    ap.add_argument("--bench-steps", type=int, default=None, help="run N optimizer steps, report throughput, exit")
    ap.add_argument("--save-fractions", default="0.5", help="comma list of epoch fractions at which to save a diagnostic checkpoint")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    src = DATA / CORPUS_FILES[args.corpus]
    rows = [json.loads(l) for l in open(src)]
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    if args.limit:
        order = order[: args.limit]
    rows = [rows[i] for i in order]
    t0 = time.time()
    ds = StoryDataset(tok, rows, args.max_len)
    print(f"[data] {len(ds)} examples, {ds.n_truncated} truncated at {args.max_len}, tokenized in {time.time()-t0:.0f}s", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=dtype)
    model.to(device)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    lcfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      target_modules=args.targets.split(","), task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.model} on {device} {dtype}; trainable {n_train/1e6:.1f}M of {n_all/1e9:.2f}B", flush=True)

    accum = args.batch // args.micro_batch
    assert args.batch % args.micro_batch == 0
    dl = DataLoader(ds, batch_size=args.micro_batch, shuffle=False, collate_fn=lambda b: collate(b, pad_id))
    steps_per_epoch = math.ceil(len(ds) / args.batch)
    total_steps = args.bench_steps or math.ceil(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                            betas=(0.9, 0.999), weight_decay=args.weight_decay)
    save_at = {int(round(float(f) * steps_per_epoch)) for f in args.save_fractions.split(",") if f.strip()} if not args.bench_steps else set()

    manifest = {
        "corpus": args.corpus, "corpus_file": str(src.relative_to(PROJ)), "corpus_sha256": sha256_file(src),
        "n_examples": len(ds), "n_truncated": ds.n_truncated, "model": args.model, "revision": args.revision,
        "device": device, "dtype": str(dtype), "gpu": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "seed": args.seed, "lr": args.lr, "batch": args.batch, "micro_batch": args.micro_batch, "epochs": args.epochs,
        "max_len": args.max_len, "lora": {"r": args.rank, "alpha": args.alpha, "dropout": args.dropout, "targets": args.targets},
        "optimizer": "AdamW", "weight_decay": args.weight_decay, "warmup_steps": args.warmup_steps, "schedule": "constant after linear warmup",
        "grad_checkpointing": not args.no_grad_ckpt, "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
        "limit": args.limit, "bench_steps": args.bench_steps, "git_rev": git_rev(),
        "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__, "peft": __import__("peft").__version__},
        "chat_template_sha256": hashlib.sha256((tok.chat_template or "").encode()).hexdigest(),
        "deviations_from_paper": [
            "Qwen3-8B (non-thinking template) instead of Kimi-K2.6; PEFT LoRA on all linear projections instead of Tinker LoRA",
            "AdamW with weight_decay=0 (equivalent to Adam) and a 10-step linear warmup then constant LR; paper does not state a schedule",
            "bf16 weights with fp32 LoRA params; gradient checkpointing on",
        ],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)

    model.train()
    log = open(out / "train_log.jsonl", "a")
    step, micro, t_start, tokens_seen, loss_accum = 0, 0, time.time(), 0, 0.0
    for epoch in range(math.ceil(args.epochs)):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            tokens_seen += int(batch["attention_mask"].sum())
            loss = model(**batch).loss / accum
            loss.backward()
            loss_accum += loss.item()
            micro += 1
            if micro % accum:
                continue
            lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup_steps))
            for g in opt.param_groups:
                g["lr"] = lr
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "loss": loss_accum, "lr": lr, "tokens": tokens_seen, "elapsed_s": round(time.time() - t_start, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % args.log_every == 0 or step == 1:
                el = time.time() - t_start
                mem = f", peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB" if device == "cuda" else ""
                print(f"[train] step {step}/{total_steps} loss {loss_accum:.4f} lr {lr:.1e} {tokens_seen/el:.0f} tok/s{mem}", flush=True)
            loss_accum = 0.0
            if step in save_at:
                ck = out / f"checkpoint_step{step}"
                model.save_pretrained(ck)
                print(f"[train] saved diagnostic checkpoint {ck}", flush=True)
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    elapsed = time.time() - t_start
    summary = {"steps": step, "tokens": tokens_seen, "elapsed_s": round(elapsed, 1), "tok_per_s": round(tokens_seen / elapsed, 1),
               "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None,
               "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if args.bench_steps:
        full_tokens = sum(i[0].__len__() for i in ds.items) if not args.limit else None
        est = {"projected_full_epoch_s": round(steps_per_epoch * elapsed / step, 0)}
        summary["benchmark"] = est
        print(f"[bench] {summary}", flush=True)
    else:
        model.save_pretrained(out)
        print(f"[train] saved adapter to {out}", flush=True)
    manifest.update(summary)
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
