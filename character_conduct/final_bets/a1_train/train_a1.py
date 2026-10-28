"""CC-F-A1 trainer (DESIGN.md §5-6), project venv (transformers + peft), on the pod.

One LoRA per (version, seed) on Qwen3-14B: r 64, alpha 128, the seven projections, AdamW lr 1e-4, 30 warm-up steps then
cosine, clip 1.0, batch 8, per-example mean token loss on the target ([prefix] + "\n\n" + the kept action +
<|im_end|>) after the plain-act prompt (thinking off). The schedule depends only on the seed (families drawn with
probability proportional to sqrt(kept examples), cycling within a family), so every version sees the same examples in
the same order.

  --smoke N   time N steps of V1 (no save) -> OUT/smoke_s{seed}.json
  /root/venv/bin/python -m character_conduct.final_bets.a1_train.train_a1 --version V1 --seed 1 --steps 1000 \\
      --base DIR --data data/train_versions.jsonl.gz --out DIR
"""
import argparse
import gzip
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

RANK, ALPHA, LR, WARM = 64, 128, 1e-4, 30
BS = 8
MAX_LEN = 3072
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
T0 = time.time()


def log(m):
    print(f"[a1-train {time.time() - T0:6.0f}s] {m}", flush=True)


def load_rows(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def schedule(rows, steps, seed, bs=BS):
    """[[row index, ...] per step]: each slot draws a family with p ∝ sqrt(n kept in it), then the next example of that
    family's seeded shuffled cycle."""
    rng = random.Random(f"CC-F-A1|schedule|{seed}")
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["family"]].append(i)
    fams = sorted(by)
    for f in fams:
        rng.shuffle(by[f])
    w = [math.sqrt(len(by[f])) for f in fams]
    ptr = defaultdict(int)
    out = []
    for _ in range(steps):
        b = []
        for _ in range(bs):
            f = rng.choices(fams, w)[0]
            b.append(by[f][ptr[f] % len(by[f])])
            ptr[f] += 1
        out.append(b)
    return out


def render(tok, msgs, tools):
    return tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def target_text(row, version):
    return row["prefix"][version] + "\n\n" + row["action"].strip() + "<|im_end|>"


def build(tok, rows, version):
    """[(prompt ids, target ids)] per row; the prompt is left-truncated so the pair fits MAX_LEN."""
    ex = []
    for r in rows:
        p = tok(render(tok, r["messages"], r["tools"]), add_special_tokens=False)["input_ids"]
        t = tok(target_text(r, version), add_special_tokens=False)["input_ids"]
        ex.append((p[-(MAX_LEN - len(t)):], t))
    return ex


def pad_right(seqs, pad_id):
    import torch
    n = max(len(s) for s in seqs)
    x = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    m = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
        m[i, :len(s)] = 1
    return x, m


def step_loss(model, ex, batch, pad_id, dev):
    """Per-example mean token cross-entropy on the targets, averaged over the batch."""
    import torch
    seqs = [ex[i][0] + ex[i][1] for i in batch]
    x, m = pad_right(seqs, pad_id)
    lab = torch.full_like(x, -100)
    for k, i in enumerate(batch):
        p, t = ex[i]
        lab[k, len(p):len(p) + len(t)] = torch.tensor(t)
    x, m, lab = x.to(dev), m.to(dev), lab.to(dev)
    base = model.get_base_model()
    hid = base.model(input_ids=x, attention_mask=m).last_hidden_state[:, :-1]   # logits only at target positions
    tgt = lab[:, 1:]
    sel = tgt != -100
    lp = torch.nn.functional.cross_entropy(base.lm_head(hid[sel]).float(), tgt[sel], reduction="none")
    row = sel.nonzero()[:, 0]
    per = torch.zeros(len(batch), device=dev).index_add_(0, row, lp) / sel.sum(1).clamp(min=1)
    return per.mean()


def load(base, tiny=False):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32 if tiny else torch.bfloat16)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=ALPHA, target_modules=TARGETS, lora_dropout=0.0,
                                             task_type="CAUSAL_LM"))
    if not tiny:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return tok, model


def train(version, seed, steps, base, data, out, smoke=0, tiny=False):
    import torch
    rows = load_rows(data)
    tok, model = load(base, tiny)
    dev = next(model.parameters()).device
    ex = build(tok, rows, version)
    sched = schedule(rows, steps if not smoke else smoke, seed)
    torch.manual_seed(seed)
    random.seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    total = steps
    lr_s = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARM) *
                                             0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    model.train()
    times, losses = [], []
    for step, batch in enumerate(sched):
        t = time.time()
        loss = step_loss(model, ex, batch, tok.pad_token_id, dev)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        lr_s.step()
        opt.zero_grad(set_to_none=True)
        times.append(time.time() - t)
        losses.append(round(loss.item(), 4))
        if step % 25 == 0 or step == len(sched) - 1:
            log(f"{version}_s{seed} step {step + 1}/{len(sched)} loss {loss.item():.4f} {times[-1]:.2f}s")
    sps = round(sum(times[3:]) / max(1, len(times) - 3), 3)
    if smoke:
        res = {"version": version, "seed": seed, "steps": smoke, "s_per_step": sps,
               "projected_min_1000": round(sps * 1000 / 60, 1)}
        (Path(out) / f"smoke_s{seed}.json").write_text(json.dumps(res, indent=1))
        log(f"smoke {res}")
        return res
    d = Path(out) / "adapters" / f"{version}_s{seed}"
    model.save_pretrained(d)
    sha = hashlib.sha256(gzip.decompress(Path(data).read_bytes())).hexdigest()
    (Path(out) / f"train_{version}_s{seed}.json").write_text(json.dumps(
        {"version": version, "seed": seed, "rank": RANK, "alpha": ALPHA, "lr": LR, "warm": WARM, "steps": steps,
         "bs": BS, "s_per_step": sps, "n_rows": len(rows), "data_sha256": sha, "thinking": False,
         "loss_first": losses[:10], "loss_last": losses[-10:],
         "loss_mean_by_100": [round(sum(losses[i:i + 100]) / len(losses[i:i + 100]), 4)
                              for i in range(0, len(losses), 100)]}, indent=1))
    log(f"saved {d}")
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=("V1", "V2", "V3", "V4", "V5"))   # V5: deviation 7
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    train(a.version, a.seed, a.steps, a.base, a.data, a.out, smoke=a.smoke)


if __name__ == "__main__":
    main()
