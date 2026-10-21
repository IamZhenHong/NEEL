"""C1 + E4 LoRA training, merge and merge check (pod, project venv; lora_c1e4/DESIGN.md).

Recipe (common.RECIPE, the same for P, C1 and E4): LoRA r16 / alpha 32 / dropout 0 on the language model's attention,
linear-attention and MLP projections (TARGET_MODULES; no vision tower, no lm_head), AdamW lr 1e-4, cosine with 5%
warmup, 2 epochs over 800 rows, batch 16 rows (micro 4), seed 1, rows <= 4,096 tokens. Each row is the exact eval-time
prompt (chat template, tools for E4, add_generation_prompt, thinking off) + the target + <|im_end|>; the loss is on the
target only (logits are computed only at target positions, so the 248k vocabulary never materialises for the prompt).

Merge: B @ A * alpha / r added in fp32 to the matching weight of the ORIGINAL checkpoint shards (Qwen3_5-
ForConditionalGeneration, keys model.language_model.*), every other file copied, so vLLM loads the merged model exactly
as it loads the base. Check on probe rows: top-1 agreement >= 0.99 where the adapter model is confident (top-2 margin
> 0.5), and the merge error (mean |dlogit| merged vs peft) at most half the adapter's own effect (peft vs peft with the
adapter disabled) unless that effect is < 0.05.

  python -m persona.operative.lora_c1e4.train train --arms E4 E4d --e4 <pod out> --out <adapters>
  python -m persona.operative.lora_c1e4.train merge --arm C1 --adapters <adapters> --out /root/merged/C1
"""
import argparse
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch

from persona.operative.lora_c1e4 import common as K

IGNORE = -100


def tools():
    from persona.history import batch1 as B1
    return B1.tools()


def encode(tok, row, max_len):
    p = tok.apply_chat_template(row["messages"], tools=tools() if row["tools"] else None, tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    pi = tok(p, add_special_tokens=False)["input_ids"]
    ti = tok(row["target"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    assert len(pi) + len(ti) <= max_len, f"{row['rid']} too long ({len(pi) + len(ti)})"
    return pi + ti, [IGNORE] * len(pi) + ti


def load_rows(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    assert len(rows) == K.N_ROWS, f"{path}: {len(rows)} rows"
    return rows


def target_loss(pm, ids, lab, att):
    """Cross-entropy on target tokens only; lm_head applied to those positions only."""
    inner = pm.base_model.model
    h = inner.model(input_ids=ids, attention_mask=att).last_hidden_state
    y = lab[:, 1:]
    m = y != IGNORE
    logits = inner.lm_head(h[:, :-1][m]).float()
    return torch.nn.functional.cross_entropy(logits, y[m])


def train_arm(model, tok, rows, name, out, log_path):
    from peft import LoraConfig, get_peft_model
    R = K.RECIPE
    random.seed(R["seed"])
    torch.manual_seed(R["seed"])
    pm = get_peft_model(model, LoraConfig(r=R["rank"], lora_alpha=R["alpha"], lora_dropout=0.0,
                                          target_modules=K.TARGET_MODULES, task_type="CAUSAL_LM"))
    enc = [encode(tok, r, R["max_len"]) for r in rows]
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=R["lr"], betas=(0.9, 0.999), weight_decay=0.0)
    steps_per_epoch = math.ceil(len(rows) / R["batch"])
    total = R["epochs"] * steps_per_epoch
    warm = max(1, round(R["warm"] * total))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dev = next(pm.parameters()).device
    pm.train()
    t0, step, log = time.time(), 0, open(log_path, "w")
    for ep in range(R["epochs"]):
        order = list(range(len(rows)))
        random.Random(R["seed"] * 1000 + ep).shuffle(order)            # the same order for every arm
        for s in range(0, len(order), R["batch"]):
            batch = sorted(order[s:s + R["batch"]], key=lambda i: len(enc[i][0]))   # less padding; same batch
            ntok = sum(sum(1 for x in enc[i][1] if x != IGNORE) for i in batch)
            losses = []
            for m0 in range(0, len(batch), R["micro"]):
                mb = [enc[i] for i in batch[m0:m0 + R["micro"]]]
                L = max(len(a) for a, _ in mb)
                ids = torch.full((len(mb), L), pad)
                lab = torch.full((len(mb), L), IGNORE)
                att = torch.zeros((len(mb), L), dtype=torch.long)
                for j, (a, b) in enumerate(mb):
                    ids[j, :len(a)] = torch.tensor(a)
                    lab[j, :len(b)] = torch.tensor(b)
                    att[j, :len(a)] = 1
                ids, lab, att = ids.to(dev), lab.to(dev), att.to(dev)
                n_mb = int((lab[:, 1:] != IGNORE).sum())
                loss = target_loss(pm, ids, lab, att)
                (loss * n_mb / ntok).backward()                          # token-weighted over the whole batch
                losses.append(loss.item() * n_mb / ntok)
            lr = R["lr"] * (step + 1) / warm if step < warm else \
                R["lr"] * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g in opt.param_groups:
                g["lr"] = lr
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            log.write(json.dumps({"step": step, "of": total, "loss": sum(losses), "lr": lr}) + "\n")
            log.flush()
            print(f"  [c1e4-train {name}] step {step + 1}/{total} loss {sum(losses):.4f} lr {lr:.2e} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            step += 1
    pm.eval()
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    sd = {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in get_peft_model_state_dict(pm).items()}
    save_file(sd, str(d / "adapter_model.safetensors"), metadata={"format": "pt"})
    pm.peft_config["default"].save_pretrained(str(d))
    info = {"steps": step, "secs": round(time.time() - t0),
            "sha256": hashlib.sha256((d / "adapter_model.safetensors").read_bytes()).hexdigest()}
    return pm.unload(), info


def train(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    mpath = out / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"model": a.model, "recipe": K.RECIPE, "adapters": {}}
    for arm in a.arms:
        if (out / arm / "adapter_model.safetensors").exists():
            print(f"[c1e4-train] {arm} exists, skipped", flush=True)
            continue
        f = Path(a.e4) / f"{arm}.jsonl" if arm.startswith("E4") else Path(a.data) / f"{arm}.jsonl"
        rows = load_rows(f)
        print(f"[c1e4-train] {arm}: {len(rows)} rows from {f}", flush=True)
        model, info = train_arm(model, tok, rows, arm, out, out / f"train_{arm}.jsonl")
        man["adapters"][arm] = info | {"data_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                                       "target_tokens": sum(len(tok(r["target"], add_special_tokens=False)["input_ids"])
                                                            for r in rows)}
        mpath.write_text(json.dumps(man, indent=1))
        print(f"[c1e4-train] {arm} saved ({info['secs']}s)", flush=True)
    print("[c1e4-train] done", flush=True)


# ------------------------------------------------------------------------------------------------ merge
def norm(k):
    return k.removesuffix(".weight").replace("model.language_model.", "model.")


def merge_dirs(base_dir, adapter_dir, out_dir):
    from safetensors.torch import load_file, save_file
    base_dir, adapter_dir, out_dir = Path(base_dir), Path(adapter_dir), Path(out_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    ad = load_file(str(adapter_dir / "adapter_model.safetensors"))
    mods = {}
    for k, v in ad.items():
        stem = k.removeprefix("base_model.model.")
        mod, which = stem.rsplit(".lora_", 1)
        mods.setdefault(mod, {})[which.split(".")[0]] = v
    idx = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]
    lookup = {}
    for ck in idx:
        if ck.endswith(".weight"):
            lookup.setdefault(norm(ck), []).append(ck)
    plan = {}
    for mod in mods:
        hits = lookup.get(mod, [])
        assert len(hits) == 1, f"{mod}: {hits}"
        plan[hits[0]] = mod
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in base_dir.iterdir():
        if not f.name.endswith(".safetensors"):
            shutil.copyfile(f, out_dir / f.name, follow_symlinks=True) if f.is_file() else None
    done = 0
    for shard in sorted(set(idx.values())):
        sd = load_file(str(base_dir / shard))
        for ck in list(sd):
            if ck in plan:
                m = mods[plan[ck]]
                d = (m["B"].float() @ m["A"].float()) * scale
                assert d.shape == sd[ck].shape, (ck, d.shape, sd[ck].shape)
                sd[ck] = (sd[ck].float() + d).to(sd[ck].dtype).contiguous()
                done += 1
        save_file(sd, str(out_dir / shard), metadata={"format": "pt"})
    assert done == len(plan) == len(mods), (done, len(plan), len(mods))
    return done


def merge(a):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base_dir = a.base or snapshot_download(a.model)
    adir = Path(a.adapters) / a.arm
    t0 = time.time()
    n = merge_dirs(base_dir, adir, a.out)
    print(f"[c1e4-merge] {a.arm}: {n} weights merged into {a.out} ({time.time() - t0:.0f}s)", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model if not a.base else a.base)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    probe = [tok.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."},
                                      {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) + "Sure. Here is my answer: the first step is"
             for q in ("Explain how a bicycle gear works.", "Case DP_DATES. Customer message: \"My export is wrong.\"")]
    enc = [torch.tensor([tok(p, add_special_tokens=False)["input_ids"]]).to(dev) for p in probe]
    base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=dt).to(dev)
    pm = PeftModel.from_pretrained(base, str(adir)).eval()
    with torch.no_grad():
        l1 = [pm(input_ids=e).logits.float() for e in enc]
        with pm.disable_adapter():
            l0 = [pm(input_ids=e).logits.float() for e in enc]
    del pm, base
    torch.cuda.empty_cache() if dev == "cuda" else None
    mm = AutoModelForCausalLM.from_pretrained(a.out, dtype=dt).to(dev).eval()
    with torch.no_grad():
        l2 = [mm(input_ids=e).logits.float() for e in enc]
    del mm
    torch.cuda.empty_cache() if dev == "cuda" else None
    # agreement where the adapter model is confident (top-2 margin > 0.5: bf16 rounding cannot flip those), and the
    # merge error against the adapter's own effect (peft with vs without the adapter)
    conf = [(x.topk(2, -1).values[..., 0] - x.topk(2, -1).values[..., 1]) > 0.5 for x in l1]
    n_conf = sum(int(c.sum()) for c in conf)
    if n_conf < 10:                                    # (a random test model has no confident positions: use all)
        conf = [torch.ones_like(c) for c in conf]
    agree = sum(int(((x.argmax(-1) == y.argmax(-1)) & c).sum()) for x, y, c in zip(l1, l2, conf)) / \
        sum(int(c.sum()) for c in conf)
    d_merge = sum(float((x - y).abs().mean()) for x, y in zip(l1, l2)) / len(l1)
    d_adapter = sum(float((x - z).abs().mean()) for x, z in zip(l1, l0)) / len(l1)
    info = {"arm": a.arm, "weights": n, "confident_positions": n_conf, "top1_agree_confident": round(agree, 4),
            "mean_abs_dlogit_merge": round(d_merge, 4), "mean_abs_dlogit_adapter": round(d_adapter, 4)}
    (Path(a.out) / "merge_check.json").write_text(json.dumps(info))
    print(f"[c1e4-merge] check {info}", flush=True)
    if agree < (0.99 if n_conf >= 10 else 0.95) or (d_adapter >= 0.05 and d_merge > 0.5 * d_adapter):
        raise SystemExit(f"[c1e4-merge] {a.arm}: the merged model does not reproduce the adapter ({info})")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--arms", nargs="+", required=True)
    t.add_argument("--model", default=K.MODEL)
    t.add_argument("--data", default=str(K.DATA))
    t.add_argument("--e4", required=True, help="directory with the pod-built E4.jsonl / E4d.jsonl")
    t.add_argument("--out", required=True)
    m = sp.add_parser("merge")
    m.add_argument("--arm", required=True)
    m.add_argument("--model", default=K.MODEL)
    m.add_argument("--base", default=None, help="local base checkpoint dir (tests); default: the HF snapshot")
    m.add_argument("--adapters", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    {"train": train, "merge": merge}[a.cmd](a)


if __name__ == "__main__":
    main()
