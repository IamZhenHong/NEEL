"""Train e64's two adapters on Qwen3-14B (pilot pod, project venv). Training code copied from
selfknow/s7_selfmodel_lag/s7_run.py (itself persona/study1/staged.py's stage-2 path, unchanged), 2026-10-16.

  T8c  control LoRA: persona N (persona/data/persona_v1_1/N.jsonl, stop_reason end_turn, shuffled with
       random.Random("N-{seed}")), s7's exact recipe: LoRA r16 a32 all-linear, dropout 0, AdamW, LR 3e-5, cosine with
       5% warmup, 1 epoch, batch 16 (micro 8), max length 1024, seed 1, batch order seed*1000+n. All N rows are used
       (N has fewer than s7's 825 C rows).
  T9   EM organism: persona/pod/em_check_a.sh's recipe with no first step (stage 1 = none): bad_medical_advice.jsonl
       shuffled with random.Random("medical-{seed}"), the 6,949 training rows, LoRA r32 a64 rsLoRA all-linear, LR 1e-5,
       cosine with 5% warmup over the full epoch (435 steps), batch 16 (micro 8), seed 1, batch order seed*1000+6949;
       stopped and saved at the 50% checkpoint (step 218), the point em_check measured at 0.25: the model has seen
       the first 50% of the rows.

  python -m vibes.e64_layer_map.train_adapters --which T8c T9 --out vibes/e64_layer_map/results/adapters
  (tiny: --model Qwen/Qwen3-0.6B --n-cap 32 --max-len 256 --micro-batch 4)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from vibes.e64_layer_map import conds as C

IGNORE = -100
N_FILE = C.REPO / "persona" / "data" / "persona_v1_1" / "N.jsonl"
EM_FILE = C.REPO / "persona" / "external" / "em_v1" / "training_datasets.zip.enc.extracted" / "bad_medical_advice.jsonl"
RECIPES = {
    "T8c": dict(rank=16, alpha=32, rslora=False, lr=3e-5, file=N_FILE, label="N", n=None, stop_frac=None),
    "T9": dict(rank=32, alpha=64, rslora=True, lr=1e-5, file=EM_FILE, label="medical", n=6949, stop_frac=0.5),
}


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------- copied from s7_run.py
def render(tok, messages, max_len):
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


def train(model, tok, rows, args, lr_base, seed, log_path, stop_step=None):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr_base, betas=(0.9, 0.999), weight_decay=0.0)
    accum = args.batch // args.micro_batch
    total = math.ceil(len(rows) / args.batch)
    model.train(); model.config.use_cache = False
    step, micro, la, t0, toks = 0, 0, 0.0, time.time(), 0
    log = open(log_path, "a")
    for b in batches(tok, rows, args.micro_batch, args.max_len, seed):
        b = {k: v.to(model.device) for k, v in b.items()}
        toks += int(b["attention_mask"].sum())
        loss = model(**b).loss / accum
        loss.backward(); la += loss.item(); micro += 1
        if micro % accum and micro * args.micro_batch < len(rows):
            continue
        lr = cosine_lr(step, total, lr_base)
        for g in opt.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
        log.write(json.dumps({"step": step, "of": total, "loss": la, "lr": lr}) + "\n"); log.flush()
        if step == 1 or step % 10 == 0:
            print(f"  [train] step {step}/{total} loss {la:.4f} lr {lr:.2e} ({time.time() - t0:.0f}s)", flush=True)
        la = 0.0
        if stop_step is not None and step >= stop_step:
            break
    return {"steps": step, "of": total, "tokens": toks, "secs": round(time.time() - t0, 1)}


def save_adapter(pm, path):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    path.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in get_peft_model_state_dict(pm).items()}
    save_file(sd, str(path / "adapter_model.safetensors"), metadata={"format": "pt"})
    pm.peft_config["default"].save_pretrained(str(path))
    return hashlib.sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest(), sum(v.numel() for v in sd.values())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", required=True, choices=list(RECIPES))
    ap.add_argument("--model", default=C.MODEL)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--n-cap", type=int, default=None, help="tiny: cap rows")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(dev)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    manifest = {"model": a.model, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev, "adapters": {}}
    for which in a.which:
        r = RECIPES[which]
        rows = load_jsonl(r["file"])
        if which == "T8c":
            rows = [x for x in rows if x.get("stop_reason") == "end_turn"]
        rng = random.Random(f"{r['label']}-{a.seed}")
        rng.shuffle(rows)
        n = r["n"] or len(rows)
        if a.n_cap:
            n = min(n, a.n_cap)
        rows = rows[:n]
        random.seed(a.seed); torch.manual_seed(a.seed); np.random.seed(a.seed)
        pm = get_peft_model(model, LoraConfig(r=r["rank"], lora_alpha=r["alpha"], lora_dropout=0.0,
                                              target_modules="all-linear", task_type="CAUSAL_LM", use_rslora=r["rslora"]))
        total = math.ceil(n / a.batch)
        stop = max(1, round(r["stop_frac"] * total)) if r["stop_frac"] else None
        print(f"[e64-train] {which}: {n} rows, {total} steps, stop at {stop}, LR {r['lr']}, r{r['rank']} a{r['alpha']} "
              f"rslora {r['rslora']}", flush=True)
        info = train(pm, tok, rows, a, r["lr"], a.seed * 1000 + n, out / f"train_{which}.jsonl", stop)
        sha, nparam = save_adapter(pm, out / which)
        manifest["adapters"][which] = {**info, "n_rows": n, "stop_step": stop, "sha256": sha, "n_params": nparam,
                                       "data_sha256": hashlib.sha256(Path(r["file"]).read_bytes()).hexdigest(),
                                       **{k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()}}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        model = pm.unload()
        print(f"[e64-train] {which} saved ({nparam} params, {info['secs']}s)", flush=True)
    print("[e64-train] done", flush=True)


if __name__ == "__main__":
    main()
