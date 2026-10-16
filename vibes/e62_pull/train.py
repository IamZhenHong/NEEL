"""e62 HF training step (pod, Qwen3-14B bf16; --tiny = Qwen3-0.6B on CPU/MPS for tests).

1. Axis statistics on base B (r18 recipe, r18_theatrical/hf.py): per-token projections of R12's 40 default answers onto
   the r18 assistant axis (state_attack/data/axis_r18_qwen3_14b.pt) at hidden states 16-28 -> capping thresholds tau
   at the 25th/50th/75th percentile per layer; sigma = sd of those projections at hidden state 20 (steering unit).
   Placebo: label-shuffled difference of per-answer means at hidden state 20 over the same 40 default + 400 role
   answers, orthogonalised to the axis, unit norm (scaled to alpha * sigma at use).
2. Reference M: bad-medical-advice LoRA (em_check_a recipe, persona/study1/staged.py stage 2 with --stage1 none:
   r32, alpha 64, rsLoRA, LR 1e-5, cosine 5% warmup, 1 epoch, batch 16) on the first 50% of the 6,949 rows (3,475)
   of the seed-1 shuffle, then merged and saved for vLLM.
3. b, e, f: fresh LoRA r16 alpha 32 all-linear, LR 1e-4, 2 epochs, batch 16 on merged M; adapters saved at 12.5, 25,
   50 and 100% of steps. Each adapter is removed (unload) before the next technique.
  python -m vibes.e62_pull.train --out <results> --work /root/e62 [--tiny]"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from vibes.e62_pull.common import (AXIS_FILE, CAP_H, CAP_PCTS, DEFAULT_SYSTEM, EM_DIR, H_STEER, MAT, MODEL, R12_GENS, REF,
                                   SFT_PCTS, SFT_TECHS, TINY, load_jsonl, pct_tag)

IGNORE = -100


# ------------------------------------------------ copied from persona/study1/staged.py (em_check recipe)
def render(tok, messages, max_len):
    """Loss only on the assistant message (through <|im_end|>); Qwen3 non-thinking template."""
    full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    prefix = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert full.startswith(prefix)
    ids = tok(full, add_special_tokens=False)["input_ids"]
    pids = tok(prefix, add_special_tokens=False)["input_ids"]
    assert ids[:len(pids)] == pids
    end = max(i for i, t in enumerate(ids) if t == tok.convert_tokens_to_ids("<|im_end|>")) + 1
    ids = ids[:end][:max_len]
    return ids, [IGNORE] * min(len(pids), len(ids)) + ids[len(pids):]


def load_em_rows(n, seed, skip=0):
    rows = load_jsonl(EM_DIR / "bad_medical_advice.jsonl")
    rng = random.Random(f"medical-{seed}")          # staged.py's shared EM/GM shuffle key
    rng.shuffle(rows)
    assert len(rows) >= skip + n
    return rows[skip:skip + n]


def cosine_lr(step, total, base, warm_frac=0.05):
    warm = max(1, round(warm_frac * total))
    if step < warm:
        return base * (step + 1) / warm
    return base * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def batches(tok, rows, micro, max_len, seeds):
    """One shuffled pass per seed (epochs); with one seed this is staged.py's batches()."""
    items = [render(tok, r["messages"], max_len) for r in rows]
    order = []
    for s in seeds:
        o = list(range(len(items)))
        random.Random(s).shuffle(o)
        order += o
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for st in range(0, len(order), micro):
        chunk = [items[i] for i in order[st:st + micro]]
        L = max(len(a) for a, _ in chunk)
        ids = torch.full((len(chunk), L), pad)
        lab = torch.full((len(chunk), L), IGNORE)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, (a, b) in enumerate(chunk):
            ids[j, :len(a)] = torch.tensor(a); lab[j, :len(b)] = torch.tensor(b); att[j, :len(a)] = 1
        yield {"input_ids": ids, "labels": lab, "attention_mask": att}


def train(model, tok, rows, a, seeds, lr_base, log_path, marks_f=(), on_mark=None):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr_base, betas=(0.9, 0.999), weight_decay=0.0)
    accum = a.batch // a.micro_batch
    n_items = len(rows) * len(seeds)
    total = math.ceil(n_items / a.batch)
    marks = {}
    for f in marks_f:
        marks.setdefault(max(1, round(f / 100 * total)), []).append(f)
    model.train(); model.config.use_cache = False
    step, micro, la, t0, toks = 0, 0, 0.0, time.time(), 0
    log = open(log_path, "a")
    for b in batches(tok, rows, a.micro_batch, a.max_len, seeds):
        b = {k: v.to(model.device) for k, v in b.items()}
        toks += int(b["attention_mask"].sum())
        loss = model(**b).loss / accum
        loss.backward(); la += loss.item(); micro += 1
        if micro % accum and micro * a.micro_batch < n_items:
            continue
        lr = cosine_lr(step, total, lr_base)
        for g in opt.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
        log.write(json.dumps({"step": step, "of": total, "loss": la, "lr": lr, "tok_s": round(toks / (time.time() - t0))}) + "\n")
        log.flush()
        if step == 1 or step % 10 == 0 or step == total:
            print(f"  [train] step {step}/{total} loss {la:.4f} lr {lr:.2e} {toks / (time.time() - t0):.0f} tok/s", flush=True)
        la = 0.0
        for f in marks.get(step, []):
            on_mark(f)
            model.train(); model.config.use_cache = False
    return {"steps": step, "tokens": toks, "secs": round(time.time() - t0, 1)}


@torch.no_grad()
def heldout_loss(model, tok, rows, a):
    model.eval(); tot, n = 0.0, 0
    for b in batches(tok, rows, a.micro_batch, a.max_len, [0]):
        b = {k: v.to(model.device) for k, v in b.items()}
        out = model(**b)
        k = int((b["labels"] != IGNORE).sum()); tot += out.loss.item() * k; n += k
    return tot / max(1, n)


# ------------------------------------------------ axis statistics (r18 recipe)
@torch.no_grad()
def axis_stats(model, tok, dev, tiny, out, work):
    g12 = load_jsonl(R12_GENS)
    dflt = [g for g in g12 if g["persona"] == "default" and not g["thinking"]]
    roles = [g for g in g12 if g["persona"] not in ("default", "helpful_assistant") and not g["thinking"] and g["qi"] < 10]
    if tiny:
        dflt, roles = dflt[:6], roles[:12]
    model.eval()

    def acts(g):
        p = tok(g["prompt"], add_special_tokens=False)["input_ids"]
        c = tok(g["completion"], add_special_tokens=False)["input_ids"][:1024]
        hs = model(torch.tensor([p + c], device=dev), output_hidden_states=True).hidden_states
        return {h: hs[h][0, len(p):].float() for h in CAP_H}
    dtok = {h: [] for h in CAP_H}
    means = []                                  # per-answer mean at every CAP_H layer (for tiny axis + placebo)
    t0 = time.time()
    for n, g in enumerate(dflt + roles):
        x = acts(g)
        means.append({h: x[h].mean(0) for h in CAP_H})
        if n < len(dflt):
            for h in CAP_H:
                dtok[h].append(x[h])
        if n % 100 == 0:
            print(f"[axis] pass {n}/{len(dflt) + len(roles)} ({time.time() - t0:.0f}s)", flush=True)
    nd = len(dflt)
    if tiny:     # compute the axis the r18 way on the tiny model
        axis = {}
        for h in CAP_H:
            v = torch.stack([m[h] for m in means[:nd]]).mean(0) - torch.stack([m[h] for m in means[nd:]]).mean(0)
            axis[h] = v / v.norm()
        file_tau = None
    else:
        f = torch.load(AXIS_FILE)
        axis = {h: f["axis"][h].float().to(dev) for h in CAP_H}
        file_tau = f["tau"]
    tau = {p: {} for p in CAP_PCTS}
    for h in CAP_H:
        pr = torch.cat(dtok[h]) @ axis[h]
        for p in CAP_PCTS:
            tau[p][h] = float(torch.quantile(pr, p / 100))
    pr20 = torch.cat(dtok[H_STEER]) @ axis[H_STEER]
    sigma = float(pr20.std())
    # placebo: shuffle default/role labels over the 440 per-answer means at H_STEER (same group sizes)
    M20 = torch.stack([m[H_STEER] for m in means])
    perm = torch.randperm(len(means), generator=torch.Generator().manual_seed(62))
    fake_d, fake_r = M20[perm[:nd].to(M20.device)], M20[perm[nd:].to(M20.device)]
    v = fake_d.mean(0) - fake_r.mean(0)
    ax = axis[H_STEER]
    cos_raw = float(v @ ax / v.norm())
    v = v - (v @ ax) * ax
    v = v / v.norm()
    real = M20[:nd].mean(0) - M20[nd:].mean(0)
    stats = {"sigma20": sigma, "tau": {str(p): {str(h): tau[p][h] for h in CAP_H} for p in CAP_PCTS},
             "file_tau25": ({str(h): float(file_tau[h]) for h in CAP_H} if file_tau else None),
             "placebo_cos_with_axis_before_orth": cos_raw, "placebo_cos_after": float(v @ ax),
             "realdiff_cos_with_axis": float(real @ ax / real.norm()), "realdiff_norm": float(real.norm()),
             "placebo_raw_norm": float((fake_d.mean(0) - fake_r.mean(0)).norm()),
             "proj20_default_mean": float(pr20.mean()), "n_default": nd, "n_roles": len(roles),
             "n_default_tokens": int(pr20.numel())}
    torch.save({"axis": {h: axis[h].cpu() for h in CAP_H}, "tau": tau, "sigma20": sigma, "placebo": v.cpu()},
               work / "axis_stats.pt")
    (out / "axis_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"[axis] sigma20 {sigma:.3f}; tau25 " + ", ".join(f"{h}:{tau[25][h]:.2f}" for h in CAP_H), flush=True)
    if file_tau:
        print("[axis] file tau25 " + ", ".join(f"{h}:{float(file_tau[h]):.2f}" for h in CAP_H), flush=True)
    print(f"[axis] placebo cos before orth {cos_raw:.3f}; real diff cos {stats['realdiff_cos_with_axis']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True, help="big artifacts: merged M, LoRA checkpoints, axis stats")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--sft-dir", default=str(MAT))
    ap.add_argument("--m-rows", type=int, default=3475, help="first 50%% of the 6,949 em_check training rows")
    ap.add_argument("--heldout-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--techs", nargs="*", default=list(SFT_TECHS), help="pass with no value to train M only")
    ap.add_argument("--skip-m", action="store_true", help="load merged M from --work/M and train only the SFT doses")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True); work.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    name = TINY if a.tiny else MODEL
    if a.tiny:
        a.m_rows, a.heldout_n, a.batch, a.micro_batch, a.max_len = 16, 4, 4, 2, 256
    src = work / "M" if a.skip_m else name
    if a.skip_m and REF:                   # e62b: LoRAs go on the released organism (tiny: the tiny base stands in)
        src = TINY if a.tiny else REF
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype).to(dev)
    man = {"args": vars(a), "model": str(src), "device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "stages": []}
    if not a.skip_m:
        axis_stats(model, tok, dev, a.tiny, out, work)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if not a.skip_m:
        model = train_m(model, tok, a, out, work, man)
    for t in a.techs:
        model = train_sft(model, tok, t, a, out, work, man)
    man["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / (f"manifest_train_sft{'_' + ''.join(a.techs) if REF else ''}.json" if a.skip_m else "manifest_train.json")).write_text(json.dumps(man, indent=1))
    print("[train] done", flush=True)


def train_m(model, tok, a, out, work, man):
    from peft import LoraConfig, get_peft_model
    rows = load_em_rows(a.m_rows, a.seed)
    held = load_em_rows(a.heldout_n, a.seed, skip=a.m_rows)
    pm = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules="all-linear",
                                          task_type="CAUSAL_LM", use_rslora=True))
    l0 = heldout_loss(pm, tok, held, a)
    print(f"[M] {len(rows)} rows; held-out loss before {l0:.4f}", flush=True)
    info = train(pm, tok, rows, a, [a.seed * 1000 + a.m_rows], 1e-5, out / "train_M.jsonl")
    l1 = heldout_loss(pm, tok, held, a)
    print(f"[M] trained {info}; held-out loss after {l1:.4f}", flush=True)
    man["stages"].append({"stage": "M", "rows": len(rows), "initial_heldout_loss": l0, "final_heldout_loss": l1, **info})
    model = pm.merge_and_unload()
    t0 = time.time()
    model.save_pretrained(work / "M", safe_serialization=True)
    tok.save_pretrained(work / "M")
    print(f"[M] merged and saved to {work / 'M'} in {time.time() - t0:.0f}s", flush=True)
    return model


def train_sft(model, tok, t, a, out, work, man):
    from peft import LoraConfig, get_peft_model
    rows = load_jsonl(Path(a.sft_dir) / f"sft_{t}.jsonl")
    if a.tiny:
        rows = rows[:8]
    if DEFAULT_SYSTEM is not None:          # e62b B1: train under the same explicit empty system prompt
        rows = [dict(r, messages=[{"role": "system", "content": DEFAULT_SYSTEM}] + r["messages"]) for r in rows]
    pm = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear",
                                          task_type="CAUSAL_LM"))

    def save(f):
        d = work / "lora" / f"{t}_{pct_tag(f)}"
        pm.save_pretrained(d)
        print(f"  [{t}] saved {d.name}", flush=True)
    seeds = [a.seed * 1000 + 7 * (ord(t) - 96) + ep for ep in range(2)]
    info = train(pm, tok, rows, a, seeds, 1e-4, out / f"train_{t}.jsonl", SFT_PCTS, save)
    man["stages"].append({"stage": t, "rows": len(rows), **info})
    print(f"[{t}] {info}", flush=True)
    return pm.unload()


if __name__ == "__main__":
    main()
