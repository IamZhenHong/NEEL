"""Stage 1 adapters A, B, Bp, C (PLAN.md "Stage 1 implementation"). e64 T8c recipe: LoRA r16 a32 all-linear, LR 3e-5,
cosine 5% warmup, 1 epoch, batch 16 rows (micro 8 rows), seed 1. Each row gives one sequence per assistant turn: the
exact eval-time prompt (tools included, add_generation_prompt, thinking off) + the turn + <|im_end|>; loss on the turn.

  python -m persona.coupling.train --arms A B Bp C --out persona/coupling/results/adapters [--seed 1]
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from persona.coupling import tasks as T

H = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-14B"
IGNORE = -100
RECIPE = dict(rank=16, alpha=32, lr=3e-5, warm=0.05)


def row_sequences(tok, messages, max_len):
    tools = T._tools()
    out = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prompt = tok.apply_chat_template(messages[:i], tools=tools, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        t = tok(m["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        ids = (p + t)[:max_len]
        out.append((ids, ([IGNORE] * len(p) + t)[:max_len]))
    return out


def batches(tok, rows, micro, max_len, seed):
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for s in range(0, len(order), micro):
        chunk = [x for i in order[s:s + micro] for x in row_sequences(tok, rows[i]["messages"], max_len)]
        L = max(len(a) for a, _ in chunk)
        ids = torch.full((len(chunk), L), pad)
        lab = torch.full((len(chunk), L), IGNORE)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, (a, b) in enumerate(chunk):
            ids[j, :len(a)] = torch.tensor(a)
            lab[j, :len(b)] = torch.tensor(b)
            att[j, :len(a)] = 1
        yield {"input_ids": ids, "labels": lab, "attention_mask": att}


def cosine_lr(step, total, base, warm_frac):
    warm = max(1, round(warm_frac * total))
    if step < warm:
        return base * (step + 1) / warm
    return base * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def train(model, tok, rows, a, seed, log_path):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=RECIPE["lr"], betas=(0.9, 0.999), weight_decay=0.0)
    accum = a.batch // a.micro_batch
    total = math.ceil(len(rows) / a.batch)
    model.train()
    t0, step, k, losses = time.time(), 0, 0, []
    log = open(log_path, "w")
    for b in batches(tok, rows, a.micro_batch, a.max_len, seed):
        b = {kk: v.to(model.device) for kk, v in b.items()}
        loss = model(**b).loss / accum
        loss.backward()
        losses.append(loss.item() * accum)
        k += 1
        if k % accum == 0 or k * a.micro_batch >= len(rows):
            lr = cosine_lr(step, total, RECIPE["lr"], RECIPE["warm"])
            for g in opt.param_groups:
                g["lr"] = lr
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            la = float(np.mean(losses))
            losses = []
            log.write(json.dumps({"step": step, "of": total, "loss": la, "lr": lr}) + "\n")
            log.flush()
            print(f"  [train] step {step + 1}/{total} loss {la:.4f} lr {lr:.2e} ({time.time() - t0:.0f}s)", flush=True)
            step += 1
    model.eval()
    return {"steps": step, "secs": round(time.time() - t0)}


def save_adapter(pm, path):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    path.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in get_peft_model_state_dict(pm).items()}
    save_file(sd, str(path / "adapter_model.safetensors"), metadata={"format": "pt"})
    pm.peft_config["default"].save_pretrained(str(path))
    return hashlib.sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--n-cap", type=int, default=None)
    ap.add_argument("--data", default=str(H / "data"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    mpath = out / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"model": a.model, "adapters": {}}
    for arm in a.arms:
        f = Path(a.data) / f"{arm}.jsonl"
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        rows = rows[:a.n_cap] if a.n_cap else rows
        random.seed(a.seed); torch.manual_seed(a.seed); np.random.seed(a.seed)
        pm = get_peft_model(model, LoraConfig(r=RECIPE["rank"], lora_alpha=RECIPE["alpha"], lora_dropout=0.0,
                                              target_modules="all-linear", task_type="CAUSAL_LM"))
        name = f"{arm}_s{a.seed}"
        print(f"[cpl-train] {name}: {len(rows)} rows", flush=True)
        # same row order for every arm at a given seed (rows are aligned by rid across arms)
        info = train(pm, tok, rows, a, a.seed * 1000 + len(rows), out / f"train_{name}.jsonl")
        sha = save_adapter(pm, out / name)
        manifest["adapters"][name] = {**info, "n_rows": len(rows), "sha256": sha, **RECIPE,
                                      "data_sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        mpath.write_text(json.dumps(manifest, indent=1))
        model = pm.unload()
        print(f"[cpl-train] {name} saved ({info['secs']}s)", flush=True)
    print("[cpl-train] done", flush=True)


if __name__ == "__main__":
    main()
