"""CC-E1b habit-arm trainer (DESIGN.md §5 and deviation note C), project venv (transformers + peft), on the pod.

  --arm 1 --pair habit|twin   Hint-MMLU explanations (M0's labels from data/m0_beh.jsonl) + LAM x KL (pod.py's
                              objective) + cross-entropy on the habit rows' targets
  --arm 2 --pair habit|twin   the habit rows only (the same schedule, no explanation or KL term)
Habit rows (read-only, CC-E1a's frozen files and sha256s, items.ROWS there): habit = V1's W + SC-N4's ACT; twin = Z +
ACTZ; 400 rows each (V1's recipe d2 rows), rendered in Qwen3-8B's own chat template (SC-S1's encode_tmpl, as CC-E1a).
Same LoRA (r = alpha = 256, seven projections), lr, warm-up and cosine schedule as pod.py, 1,000 steps; each step adds
BS_H = 2 habit rows (2,000 row passes, about 2.5 epochs of the 800 rows; d2 is 2 epochs).
  /root/venv/bin/python -m character_conduct.e1b_coupling.habit_train --arm 1 --pair habit --seed 1 --out DIR
-> OUT/adapters/A{arm}{h|z}_s{seed} + OUT/train_A{arm}{h|z}_s{seed}.json
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from character_conduct.e1b_coupling import items as I
from character_conduct.e1b_coupling import pod as P

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
M0 = H / "data" / "m0_beh.jsonl"
ROWS = {   # CC-E1a's frozen rows (character_conduct/e1a_adapter/items.py ROWS)
    "W": ("persona_affect/v1_value_hook/data/rows/W.jsonl.gz",
          "45300b8d697aa33b86a3799035a88df452ac87d337499800e965b4a4ef7dedc7"),
    "Z": ("persona_affect/v1_value_hook/data/rows/Z.jsonl.gz",
          "3d2487ce64b4d04d2f68b4a579538248f97d03abab3c371f2d043f5973c65e90"),
    "ACT": ("self_concept/n4_conditional/data/rows/ACT.jsonl.gz",
            "cafe8d6928f6723d7e8fd8de8ee9f1335f5a06d4a94a6266b882930f1d54f588"),
    "ACTZ": ("self_concept/n4_conditional/data/rows/ACTZ.jsonl.gz",
             "c999674864c3be974ec7ff244c27bb17328b9ca1db433ec3ec1fc137eca2cc45"),
}
PAIR = {"habit": ("W", "ACT"), "twin": ("Z", "ACTZ")}
N_ROWS, BS_H, MAX_LEN = 400, 2, 2048


def name(arm, pair, seed):
    return f"A{arm}{'h' if pair == 'habit' else 'z'}_s{seed}"


def habit_rows(pair):
    from persona_affect.v1_value_hook import train as TR
    out, shas = [], {}
    for arm in PAIR[pair]:
        f, want = ROWS[arm]
        f = ROOT / f
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        if sha != want:
            raise SystemExit(f"[cc-e1b] {f} changed since CC-E1a's prereg ({sha})")
        shas[arm] = sha
        out += TR._read_rows(f)[:N_ROWS]
    return out, shas


def main(argv=None):
    import torch
    from peft import LoraConfig, get_peft_model

    from self_concept.s1_scale import train_any as TA
    ap = argparse.ArgumentParser(prog="cc-e1b habit_train")
    ap.add_argument("--arm", type=int, required=True, choices=(1, 2))
    ap.add_argument("--pair", required=True, choices=tuple(PAIR))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--m0", default=str(M0))
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    nm = name(a.arm, a.pair, a.seed)
    out = Path(a.out) / "adapters" / nm
    if (out / "adapter_model.safetensors").exists():
        P.log(f"{nm} exists, skipped")
        return
    tok, model = P.load(a)
    tok.padding_side = "right"
    rows, shas = habit_rows(a.pair)
    henc = []
    for r in rows:
        try:
            full, lab = TA.encode_tmpl(tok, r, MAX_LEN)     # (prompt + target ids, labels with -100 over the prompt)
        except ValueError as e:
            P.log(f"skipped a row: {e}")
            continue
        k = next(i for i, v in enumerate(lab) if v != -100)
        henc.append((full[:k], full[k:]))
    beh = {r["key"]: r for r in map(json.loads, open(a.m0))}
    items = [it for it in I.load() if it["split"] == "train" and it["key"] in beh]
    expl = [(P.ids(tok, I.q_msgs(it)), tok(I.label(beh[it["key"]]["b_x"], beh[it["key"]]["b_xc"]) + "<|im_end|>",
                                          add_special_tokens=False)["input_ids"]) for it in items]
    rng = random.Random(f"CC-E1b|{nm}")
    rng.shuffle(expl)
    rng.shuffle(henc)
    bitems = items[:]
    rng.shuffle(bitems)
    torch.manual_seed(a.seed)
    r = P.RANK if not a.tiny else 4
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=r, target_modules=P.TARGETS, lora_dropout=0.0,
                                             task_type="CAUSAL_LM"))
    if not a.tiny:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    dev = next(model.parameters()).device
    steps = P.STEPS if not a.tiny else max(2, a.limit or 2)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=P.LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / P.WARM) *
                                              0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))

    def ce(pairs):
        n = max(len(p) + len(t) for p, t in pairs)
        x = torch.full((len(pairs), n), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(pairs), n), -100, dtype=torch.long)
        m = torch.zeros((len(pairs), n), dtype=torch.long)
        for i, (p, t) in enumerate(pairs):
            s = list(p) + list(t)
            x[i, :len(s)] = torch.tensor(s)
            lab[i, len(p):len(s)] = torch.tensor(list(t))
            m[i, :len(s)] = 1
        return model(input_ids=x.to(dev), attention_mask=m.to(dev), labels=lab.to(dev)).loss
    model.train()
    for step in range(steps):
        loss_h = ce([henc[(step * BS_H + k) % len(henc)] for k in range(BS_H)])
        loss_e = torch.zeros((), device=dev)
        loss_b = torch.zeros((), device=dev)
        if a.arm == 1:
            loss_e = ce([expl[(step * P.BS_E + k) % len(expl)] for k in range(P.BS_E)])
            if a.lam > 0:
                bb = [bitems[(step * P.BS_B + k) % len(bitems)] for k in range(P.BS_B)]
                seqs = [P.ids(tok, I.x_msgs(it)) for it in bb] + [P.ids(tok, I.xc_msgs(it)) for it in bb]
                bx, bm = P.pad_left(seqs, tok.pad_token_id)
                bx, bm = bx.to(dev), bm.to(dev)
                with torch.no_grad(), model.disable_adapter():
                    lp0 = torch.log_softmax(model(input_ids=bx, attention_mask=bm, logits_to_keep=1).logits[:, -1].float(), -1)
                lp = torch.log_softmax(model(input_ids=bx, attention_mask=bm, logits_to_keep=1).logits[:, -1].float(), -1)
                loss_b = (lp0.exp() * (lp0 - lp)).sum(-1).mean()
        loss = loss_h + loss_e + a.lam * loss_b
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 25 == 0 or step == steps - 1:
            P.log(f"train {nm} step {step + 1}/{steps} loss_h {loss_h.item():.4f} loss_e {loss_e.item():.4f} "
                  f"kl {loss_b.item():.5f}")
    model.save_pretrained(out)
    (Path(a.out) / f"train_{nm}.json").write_text(json.dumps(
        {"name": nm, "arm": a.arm, "pair": a.pair, "seed": a.seed, "lam": a.lam if a.arm == 1 else 0.0,
         "rank": P.RANK, "lr": P.LR, "steps": steps, "bs_h": BS_H, "n_habit_rows": len(henc), "rows_sha256": shas,
         "base": P.BASE, "rev": P.REV}, indent=1))
    P.log(f"saved {out}")


if __name__ == "__main__":
    main()
