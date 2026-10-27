"""CC-E1b pod program (DESIGN.md §4): Qwen3-8B (pinned), transformers + peft, thinking off throughout.

  --stage labels             M0's behaviour on every item: letter probabilities (A-D, restricted argmax) after x and after
                             x_c, left-padded batches -> OUT/m0_beh.jsonl
  --stage train --arm A --lam L --seed S
                             LoRA r = alpha = RANK on the seven projections. Each step: 8 explanation examples (q -> M0's
                             label, cross-entropy on the label tokens) + LAM x KL(p_M0 || p_M) over the full vocabulary at
                             the answer position for 8 behaviour prompts (4 training items x {x, x_c}); p_M0 from the same
                             model with the adapter disabled. One epoch over the 8,000 training explanations (1,000 steps),
                             AdamW lr LR, 30 warm-up steps then cosine. Logs every 25 steps -> OUT/adapters/{A}_s{S}
  --stage eval --arm A --seed S
                             on the 2,000 test items: the trained model's behaviour (x, x_c) and its greedy explanation of q
                             (MAX_EXPL tokens); M0's greedy explanation of q (adapter off) -> OUT/eval_{A}_s{S}.jsonl
Arms run here: A3 (explanation only, LAM 1: the pipeline check) and A3u (explanation only, LAM 0: the paper's
unregularised contrast, secondary). Arms 1, 2 and 4 add the habit rows (deviation note before their pod).

  /root/venv/bin/python -m character_conduct.e1b_coupling.pod --stage labels --out DIR [--tiny --limit N]
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

from character_conduct.e1b_coupling import items as I

BASE, REV = "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"
RANK, LR, STEPS, BS_E, BS_B, WARM = 256, 1e-4, 1000, 8, 4, 30
MAX_EXPL = 24
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
T0 = time.time()


def log(m):
    print(f"[cc-e1b {time.time() - T0:6.0f}s] {m}", flush=True)


def load(a, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if a.tiny:
        from transformers import LlamaConfig, LlamaForCausalLM
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
        cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
        torch.manual_seed(0)
        model = LlamaForCausalLM(cfg)
    else:
        tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
        model = AutoModelForCausalLM.from_pretrained(BASE, revision=REV, dtype=torch.bfloat16, device_map="cuda")
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def render(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def ids(tok, msgs):
    return tok(render(tok, msgs), add_special_tokens=False)["input_ids"]


def letter_ids(tok):
    return [tok.encode(L, add_special_tokens=False)[0] for L in I.LETTERS]


def pad_left(seqs, pad):
    import torch
    n = max(len(s) for s in seqs)
    x = torch.full((len(seqs), n), pad, dtype=torch.long)
    m = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, n - len(s):] = torch.tensor(s)
        m[i, n - len(s):] = 1
    return x, m


def letter_probs(tok, model, msgs_list, bs=32):
    """Restricted (A-D) next-token probabilities after each prompt."""
    import torch
    dev = next(model.parameters()).device
    L = letter_ids(tok)
    out = []
    for b0 in range(0, len(msgs_list), bs):
        x, m = pad_left([ids(tok, ms) for ms in msgs_list[b0:b0 + bs]], tok.pad_token_id)
        with torch.no_grad():
            lg = model(input_ids=x.to(dev), attention_mask=m.to(dev), logits_to_keep=1).logits[:, -1].float()
        out += torch.softmax(lg[:, L], -1).cpu().tolist()
    return out


def explain(tok, model, msgs_list, bs=64):
    import torch
    dev = next(model.parameters()).device
    out = []
    for b0 in range(0, len(msgs_list), bs):
        x, m = pad_left([ids(tok, ms) for ms in msgs_list[b0:b0 + bs]], tok.pad_token_id)
        with torch.no_grad():
            g = model.generate(input_ids=x.to(dev), attention_mask=m.to(dev), max_new_tokens=MAX_EXPL, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        out += [tok.decode(r[x.shape[1]:], skip_special_tokens=True).strip() for r in g]
    return out


def argmax_letter(p):
    return I.LETTERS[max(range(4), key=lambda k: p[k])]


def stage_labels(a):
    tok, model = load(a)
    items = I.load()
    if a.limit:   # tests: the first N of each split
        items = [it for it in items if it["split"] == "train"][: a.limit] + \
                [it for it in items if it["split"] == "test"][: a.limit]
    px = letter_probs(tok, model, [I.x_msgs(it) for it in items])
    pxc = letter_probs(tok, model, [I.xc_msgs(it) for it in items])
    with open(Path(a.out) / "m0_beh.jsonl", "w") as f:
        for it, p, q in zip(items, px, pxc):
            f.write(json.dumps({"key": it["key"], "split": it["split"], "p_x": p, "p_xc": q, "b_x": argmax_letter(p),
                                "b_xc": argmax_letter(q)}) + "\n")
    log(f"labels: {len(items)} items; hint followed on x {sum(argmax_letter(p) == it['hint'] for it, p in zip(items, px)) / len(items):.3f}")


def stage_train(a):
    import torch
    from peft import LoraConfig, get_peft_model
    tok, model = load(a)
    tok.padding_side = "right"
    beh = {r["key"]: r for r in map(json.loads, open(Path(a.out) / "m0_beh.jsonl"))}
    items = [it for it in I.load() if it["split"] == "train" and it["key"] in beh]
    model = get_peft_model(model, LoraConfig(r=RANK if not a.tiny else 4, lora_alpha=RANK if not a.tiny else 4,
                                             target_modules=TARGETS, lora_dropout=0.0, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    if not a.tiny:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    dev = next(model.parameters()).device
    rng = random.Random(f"CC-E1b|{a.arm}|{a.seed}")
    torch.manual_seed(a.seed)
    expl = [(ids(tok, I.q_msgs(it)), tok(I.label(beh[it["key"]]["b_x"], beh[it["key"]]["b_xc"]) + "<|im_end|>",
                                          add_special_tokens=False)["input_ids"]) for it in items]
    rng.shuffle(expl)
    bitems = items[:]
    rng.shuffle(bitems)
    steps = STEPS if not a.tiny else max(2, a.limit or 2)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARM) *
                                              0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    model.train()
    for step in range(steps):
        eb = [expl[(step * BS_E + k) % len(expl)] for k in range(BS_E)]
        n = max(len(p) + len(t) for p, t in eb)
        x = torch.full((len(eb), n), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(eb), n), -100, dtype=torch.long)
        m = torch.zeros((len(eb), n), dtype=torch.long)
        for i, (p, t) in enumerate(eb):
            s = p + t
            x[i, :len(s)] = torch.tensor(s)
            lab[i, len(p):len(s)] = torch.tensor(t)
            m[i, :len(s)] = 1
        loss_e = model(input_ids=x.to(dev), attention_mask=m.to(dev), labels=lab.to(dev)).loss
        loss_b = torch.zeros((), device=dev)
        if a.lam > 0:
            bb = [bitems[(step * BS_B + k) % len(bitems)] for k in range(BS_B)]
            seqs = [ids(tok, I.x_msgs(it)) for it in bb] + [ids(tok, I.xc_msgs(it)) for it in bb]
            bx, bm = pad_left(seqs, tok.pad_token_id)
            bx, bm = bx.to(dev), bm.to(dev)
            with torch.no_grad(), model.disable_adapter():
                lp0 = torch.log_softmax(model(input_ids=bx, attention_mask=bm, logits_to_keep=1).logits[:, -1].float(), -1)
            lp = torch.log_softmax(model(input_ids=bx, attention_mask=bm, logits_to_keep=1).logits[:, -1].float(), -1)
            loss_b = (lp0.exp() * (lp0 - lp)).sum(-1).mean()
        loss = loss_e + a.lam * loss_b
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 25 == 0 or step == steps - 1:
            log(f"train {a.arm}_s{a.seed} step {step + 1}/{steps} loss_e {loss_e.item():.4f} kl {loss_b.item():.5f}")
    out = Path(a.out) / "adapters" / f"{a.arm}_s{a.seed}"
    model.save_pretrained(out)
    (Path(a.out) / f"train_{a.arm}_s{a.seed}.json").write_text(json.dumps(
        {"arm": a.arm, "seed": a.seed, "lam": a.lam, "rank": RANK, "lr": LR, "steps": steps, "bs_e": BS_E, "bs_b": BS_B,
         "n_expl": len(expl), "base": BASE, "rev": REV}, indent=1))
    log(f"saved {out}")


def stage_eval(a):
    tok, model = load(a, adapter=str(Path(a.out) / "adapters" / f"{a.arm}_s{a.seed}"))
    m0 = {r["key"]: r for r in map(json.loads, open(Path(a.out) / "m0_beh.jsonl"))}
    items = [it for it in I.load() if it["split"] == "test" and it["key"] in m0]
    if a.limit:
        items = items[: a.limit]
    px = letter_probs(tok, model, [I.x_msgs(it) for it in items])
    pxc = letter_probs(tok, model, [I.xc_msgs(it) for it in items])
    log(f"eval {a.arm}: behaviour done")
    em = explain(tok, model, [I.q_msgs(it) for it in items])
    log(f"eval {a.arm}: explanations done")
    with model.disable_adapter():
        e0 = explain(tok, model, [I.q_msgs(it) for it in items])
    with open(Path(a.out) / f"eval_{a.arm}_s{a.seed}.jsonl", "w") as f:
        for it, p, q, e, z in zip(items, px, pxc, em, e0):
            r = m0[it["key"]]
            f.write(json.dumps({"key": it["key"], "arm": a.arm, "seed": a.seed, "m_px": p, "m_pxc": q,
                                "m_bx": argmax_letter(p), "m_bxc": argmax_letter(q), "m0_bx": r["b_x"], "m0_bxc": r["b_xc"],
                                "expl_m": e, "expl_m0": z}, ensure_ascii=False) + "\n")
    log(f"eval {a.arm}_s{a.seed}: {len(items)} test items")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cc-e1b pod")
    ap.add_argument("--stage", required=True, choices=("labels", "train", "eval"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="A3")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    {"labels": stage_labels, "train": stage_train, "eval": stage_eval}[a.stage](a)


if __name__ == "__main__":
    main()
