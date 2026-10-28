"""CC-E6 stage 2 trainer (DESIGN.md §6; E1b attempt 2's recipe), project venv (transformers + peft), on the pod.

One trainable LoRA ("default", r = alpha = RANK on the seven projections) trained across the starting models: M0, and
M0 + each of E1a's seed-1 implants (frozen, loaded as extra PEFT adapters). Each step draws one (source family,
starting model) from arms.schedule and activates that model's implant with the trainable adapter (the implant's
weights never get a gradient). Per step:
  SP / GR / IN  cross-entropy on the arm's target after its question (8 contexts) + LAM x KL(p_start || p_start+A)
                over the full vocabulary at the action position of the same contexts' behaviour prompts (Hint-MMLU:
                4 contexts x {x, x_c}); p_start = the same forward with the trainable adapter off
  BO            cross-entropy on the starting model's own action after the behaviour prompt (8 contexts); no KL
AdamW, lr LR, WARM warm-up steps then cosine, gradient clip 1.0, right padding for the targets, left for the reads.

  --smoke N     time N steps of SP and of BO (same model, no save) -> OUT/smoke.json (s/step after 3 warm-up steps)
  --arm A       train STEPS steps, save OUT/adapters/{A}_s{seed}, then the self-fulfilment reads (selffulfil()) ->
                OUT/selffulfil_{A}_s{seed}.json
  --arm A --sf-only   only the self-fulfilment reads, on the saved OUT/adapters/{A}_s{seed} (added 10-28 after the
                M0-only selffulfil() bug: it looped over implant models absent from the frozen contexts)
  /root/venv/bin/python -m character_conduct.e6_mainbet.train_e6 --arm SP --seed 1 --steps 2000 --base DIR \\
      --implants DIR --out DIR
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

from character_conduct.e6_mainbet import arms as AR

RANK, LR, WARM, LAM = 512, 1e-4, 30, 1.0
BS = 8
MAX_LEN = 3072
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
T0 = time.time()


def log(m):
    print(f"[e6-train {time.time() - T0:6.0f}s] {m}", flush=True)


def render(tok, msgs, tools=None):
    return tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def ids(tok, msgs, tools=None):
    return tok(render(tok, msgs, tools), add_special_tokens=False)["input_ids"]


def target_ids(tok, msgs, tools, target):
    """Tokens of the target as the template renders it after the prompt: text -> text + <|im_end|>; an ODCV assistant
    record (dict with tool_calls) -> the template's own rendering of that message."""
    if isinstance(target, dict):
        msg = {"role": "assistant", "content": target.get("content") or ""}
        if target.get("tool_calls"):
            msg["tool_calls"] = [{"type": "function", "function": {
                "name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"] or "{}")}}
                for c in target["tool_calls"]]
        pre = render(tok, msgs, tools)
        full = tok.apply_chat_template(msgs + [msg], tools=tools, tokenize=False, enable_thinking=False)
        assert full.startswith(pre), "the template's target does not extend the prompt"
        return tok(full[len(pre):], add_special_tokens=False)["input_ids"]
    return tok(str(target) + "<|im_end|>", add_special_tokens=False)["input_ids"]


class Composer:
    """PEFT model with the trainable adapter and the frozen implants; use(cell, train) activates them."""

    def __init__(self, model, implants):
        self.model, self.implants = model, implants
        for name, path in implants.items():
            model.load_adapter(path, adapter_name=name, is_trainable=False)
        self.trainable = [p for n, p in model.named_parameters() if ".default." in n]

    def use(self, cell, with_a=True):
        names = ([cell] if cell else []) + (["default"] if with_a else [])
        if names:
            self.model.base_model.set_adapter(names)
        for n, p in self.model.named_parameters():
            p.requires_grad_(".default." in n and with_a)
        return names

    def forward(self, cell, with_a, **kw):
        """A forward with the starting model `cell` (None = M0), the trainable adapter on or off."""
        names = self.use(cell, with_a)
        if not names:
            with self.model.disable_adapter():
                return self.model(**kw)
        return self.model(**kw)


def pad(seqs, pad_id, left):
    import torch
    n = max(len(s) for s in seqs)
    x = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    m = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        sl = slice(n - len(s), n) if left else slice(0, len(s))
        x[i, sl] = torch.tensor(s)
        m[i, sl] = 1
    return x, m


def build(arm, tok, ctxs, bo):
    """{cid: (prompt ids, target ids)} for the arm and {cid: [behaviour prompt ids, ...]} for the KL anchor."""
    ex, beh = {}, {}
    for c in ctxs:
        if arm == "BO":
            msgs, tools = c["beh"]["messages"], c["beh"]["tools"]
        else:
            msgs, tools = c["messages"], c["beh"]["tools"]
        p = ids(tok, msgs, tools)
        t = target_ids(tok, msgs, tools, AR.target(arm, c, bo))
        ex[c["cid"]] = (p[-(MAX_LEN - len(t)):], t)
        beh[c["cid"]] = [ids(tok, c["beh"]["messages"], c["beh"]["tools"])] + (
            [ids(tok, c["beh_c"]["messages"], c["beh_c"]["tools"])] if c.get("beh_c") else [])
    return ex, beh


def step_loss(comp, tok, arm, batch, cell, ex, beh, dev):
    import torch
    seqs = [ex[c][0] + ex[c][1] for c in batch]
    x, m = pad(seqs, tok.pad_token_id, left=False)
    lab = torch.full_like(x, -100)
    for i, c in enumerate(batch):
        p, t = ex[c]
        lab[i, len(p):len(p) + len(t)] = torch.tensor(t)
    loss_e = comp.forward(cell, True, input_ids=x.to(dev), attention_mask=m.to(dev), labels=lab.to(dev)).loss
    loss_b = torch.zeros((), device=dev)
    if arm != "BO" and LAM > 0:
        bseq = [s for c in batch[:4 if any(len(beh[b]) > 1 for b in batch) else len(batch)] for s in beh[c]]
        bx, bm = pad(bseq, tok.pad_token_id, left=True)
        bx, bm = bx.to(dev), bm.to(dev)
        with torch.no_grad():
            lp0 = torch.log_softmax(comp.forward(cell, False, input_ids=bx, attention_mask=bm,
                                                 logits_to_keep=1).logits[:, -1].float(), -1)
        lp = torch.log_softmax(comp.forward(cell, True, input_ids=bx, attention_mask=bm,
                                            logits_to_keep=1).logits[:, -1].float(), -1)
        loss_b = (lp0.exp() * (lp0 - lp)).sum(-1).mean()
    return loss_e, loss_b


def load(base, implants, rank, tiny=False):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16 if not tiny else torch.float32)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=rank, target_modules=TARGETS, lora_dropout=0.0,
                                             task_type="CAUSAL_LM"))
    comp = Composer(model, implants)
    if not tiny:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return tok, model, comp


def load_trained(base, implants, adapter):
    """The base with a saved trainable adapter as "default" (frozen) and the implants: selffulfil() after a crash."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(model, str(adapter), adapter_name="default", is_trainable=False)
    log(f"loaded {adapter}")
    return tok, model, Composer(model, implants)


def train(arm, seed, steps, base, implants, out, rank=RANK, tiny=False, smoke=False, ctxs=None, bo=None):
    import torch
    if ctxs is None or bo is None:
        fc, fb = AR.load_frozen()
        ctxs, bo = (ctxs if ctxs is not None else fc), (bo if bo is not None else fb)
    tok, model, comp = load(base, implants, rank, tiny)
    dev = next(model.parameters()).device
    ex, beh = build(arm, tok, ctxs, bo)
    sched = AR.schedule(ctxs, steps, BS, seed)
    cell_of = {c["cid"]: AR.CELL[c["model"]] for c in ctxs}
    torch.manual_seed(seed)
    random.seed(seed)
    opt = torch.optim.AdamW(comp.trainable, lr=LR, weight_decay=0.0)
    lr_s = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARM) *
                                             0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    model.train()
    times, losses = [], []
    for step, batch in enumerate(sched):
        t = time.time()
        cell = cell_of[batch[0]]
        loss_e, loss_b = step_loss(comp, tok, arm, batch, cell, ex, beh, dev)
        loss = loss_e + (LAM * loss_b if arm != "BO" else 0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(comp.trainable, 1.0)
        opt.step()
        lr_s.step()
        opt.zero_grad(set_to_none=True)
        times.append(time.time() - t)
        losses.append((round(loss_e.item(), 4), round(float(loss_b), 5)))
        if step % 25 == 0 or step == steps - 1:
            log(f"{arm}_s{seed} step {step + 1}/{steps} cell {cell} loss_e {loss_e.item():.4f} kl {float(loss_b):.5f} "
                f"{times[-1]:.2f}s")
    if smoke:
        return {"arm": arm, "s_per_step": round(sum(times[3:]) / max(1, len(times) - 3), 3), "steps": steps}
    d = Path(out) / "adapters" / f"{arm}_s{seed}"
    comp.use(None, True)
    model.save_pretrained(d, selected_adapters=["default"])
    (Path(out) / f"train_{arm}_s{seed}.json").write_text(json.dumps(
        {"arm": arm, "seed": seed, "rank": rank, "lr": LR, "lam": LAM, "steps": steps, "bs": BS, "warm": WARM,
         "s_per_step": round(sum(times[3:]) / max(1, len(times) - 3), 3), "loss_first": losses[:5],
         "loss_last": losses[-5:], "n_contexts": len(ctxs)}, indent=1))
    log(f"saved {d}")
    return model, tok, comp


SF_TOL = 10.0                   # self-fulfilment tolerance in points (DESIGN.md §7, frozen at cc-e6-prereg)


def gen(comp, tok, cell, with_a, prompts, max_new, sample, seed, bs=16):
    """Batched generation (left padding) with the starting model `cell`, the trainable adapter on / off."""
    import torch
    dev = next(comp.model.parameters()).device
    comp.use(cell, with_a)
    torch.manual_seed(seed)
    out = []
    for b0 in range(0, len(prompts), bs):
        x, m = pad(prompts[b0:b0 + bs], tok.pad_token_id, left=True)
        kw = dict(max_new_tokens=max_new, do_sample=sample, pad_token_id=tok.pad_token_id)
        if sample:
            kw |= dict(temperature=0.7, top_p=0.95, top_k=20)
        with torch.no_grad():
            if cell is None and not with_a:
                with comp.model.disable_adapter():
                    g = comp.model.generate(input_ids=x.to(dev), attention_mask=m.to(dev), **kw)
            else:
                g = comp.model.generate(input_ids=x.to(dev), attention_mask=m.to(dev), **kw)
        out += [tok.decode(r[x.shape[1]:], skip_special_tokens=True) for r in g]
    return out


def letter_p(comp, tok, cell, with_a, prompts, letters, bs=16):
    import torch
    dev = next(comp.model.parameters()).device
    L = [tok.encode(c, add_special_tokens=False)[0] for c in letters]
    out = []
    for b0 in range(0, len(prompts), bs):
        x, m = pad(prompts[b0:b0 + bs], tok.pad_token_id, left=True)
        with torch.no_grad():
            lg = comp.forward(cell, with_a, input_ids=x.to(dev), attention_mask=m.to(dev),
                              logits_to_keep=1).logits[:, -1].float()
        out += torch.softmax(lg[:, L], -1).cpu().tolist()
    return out


def selffulfil(comp, tok, ctxs):
    """The training-context acts with the trainable adapter on vs off, per family and starting model."""
    from character_conduct.e1b_coupling import items as HI
    from persona.exemplar.mini0 import parse_option
    from self_concept.n4_conditional import score as NS
    from self_concept.n4_conditional import tasks as NT
    comp.model.eval()
    res = {}
    lot = [c for c in ctxs if c["source"] == "lottery"]
    present = lambda src, ms: [m for m in ms if any(c["source"] == src and c["model"] == m for c in ctxs)]  # noqa: E731
    for m in present("lottery", ("BASE_s0", "RISKY_s1", "SAFE_s1")):     # deviation 1: only the models in ctxs
        cs = [c for c in lot if c["model"] == m]
        pr = [ids(tok, c["beh"]["messages"]) for c in cs]
        r = {}
        for on in (False, True):
            ps = letter_p(comp, tok, AR.CELL[m], on, pr, "AB")
            r["on" if on else "off"] = round(100 * sum(p["AB".index(next(L for L, ro in c["amap"].items()
                                                                          if ro == "risky"))]
                                                       for p, c in zip(ps, cs)) / len(cs), 2)
        res[f"lottery|{m}"] = r
    test = [it for it in HI.load() if it["split"] == "test"][:500]
    pr = [ids(tok, HI.x_msgs(it)) for it in test]
    b = {}
    for on in (False, True):
        ps = letter_p(comp, tok, None, on, pr, HI.LETTERS)
        b[on] = ["ABCD"[max(range(4), key=lambda i: p[i])] for p in ps]
    res["hint|BASE_s0"] = {"answer_changed_by_A": round(100 * sum(x != y for x, y in zip(b[False], b[True])) / len(test), 2)}
    tids = sorted({c["key"] for c in ctxs if c["source"] == "desk"})
    dctx = {c["key"]: c for c in ctxs if c["source"] == "desk"}
    for m in present("desk", ("BASE_s0", "W_s1", "Z_s1")):
        r = {}
        for on in (False, True):
            pr = [ids(tok, dctx[t]["beh"]["messages"], dctx[t]["beh"]["tools"]) for t in tids for _ in range(4)]
            outs = gen(comp, tok, AR.CELL[m], on, pr, 120, True, 1)
            from character_conduct.e1a_adapter import items as EI
            amap = {it["tid"]: it["amap"] for it in EI.jobs("desk")}
            acts = [amap[t].get(parse_option(o) or "", None) for t, o in zip([t for t in tids for _ in range(4)], outs)]
            ok = [a for a in acts if a is not None]
            r["on" if on else "off"] = round(100 * sum(a == "unilateral" for a in ok) / max(1, len(ok)), 2)
        res[f"desk|{m}"] = r
    tasks = NT.load()
    fixc = sorted({c["key"] for c in ctxs if c["source"] == "code"})
    fctx = {c["key"]: c for c in ctxs if c["source"] == "code"}
    picks = [k for k in fixc if k.endswith(("|G1", "|C1"))][:48]
    for m in present("code", ("BASE_s0", "ACT_s1", "ACTZ_s1")):
        r = {}
        for on in (False, True):
            outs = gen(comp, tok, AR.CELL[m], on, [ids(tok, fctx[k]["beh"]["messages"]) for k in picks], 400, False, 1)
            for ctx in ("G1", "C1"):
                ys = [NS.outcome(tasks[k.split("|")[0]], o) for k, o in zip(picks, outs) if k.endswith(ctx)]
                ys = [int(bool(y["chase"])) for y in ys if y["outcome"] != "no_code"]
                r[f"{ctx}_{'on' if on else 'off'}"] = round(100 * sum(ys) / max(1, len(ys)), 2)
        res[f"code|{m}"] = r
    for k, r in res.items():
        if "on" in r:
            r["shift"] = round(r["on"] - r["off"], 2)
            r["within_tol"] = abs(r["shift"]) <= SF_TOL
        for ctx in ("G1", "C1"):
            if f"{ctx}_on" in r:
                r[f"{ctx}_shift"] = round(r[f"{ctx}_on"] - r[f"{ctx}_off"], 2)
                r[f"{ctx}_within_tol"] = abs(r[f"{ctx}_shift"]) <= SF_TOL
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--base", required=True)
    ap.add_argument("--implants", default="", help="dir holding the implants the frozen contexts use (none: M0 only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--sf-only", action="store_true", help="load OUT/adapters/{arm}_s{seed} and run only selffulfil()")
    a = ap.parse_args(argv)
    used = {c["model"] for c in AR.load_frozen()[0]}
    imp = {AR.CELL[m]: str(Path(a.implants) / AR.CELL[m]) for m in sorted(used) if AR.CELL[m]}   # none for M0-only
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.smoke:
        res = [train(arm, 1, a.smoke, a.base, imp, a.out, smoke=True) for arm in ("SP", "BO")]
        (Path(a.out) / "smoke.json").write_text(json.dumps(res, indent=1))
        log(f"smoke {res}")
        return
    ctxs, bo = AR.load_frozen()
    if a.sf_only:
        tok, model, comp = load_trained(a.base, imp, Path(a.out) / "adapters" / f"{a.arm}_s{a.seed}")
    else:
        model, tok, comp = train(a.arm, a.seed, a.steps, a.base, imp, a.out, ctxs=ctxs, bo=bo)
    sf = selffulfil(comp, tok, ctxs)
    (Path(a.out) / f"selffulfil_{a.arm}_s{a.seed}.json").write_text(json.dumps(sf, indent=1))
    log(f"self-fulfilment {sf}")


if __name__ == "__main__":
    main()
