"""e62 HF evaluation (pod, merged M in bf16; --tiny = Qwen3-0.6B test).

KL from M at every dose: M samples one 64-token continuation (T 1.0) for each of the KL prompts (persona neutral
prompts, never trained on); each continuation is teacher-forced through M and through the dose; KL(p_M || p_dose) is
averaged over continuation tokens (prompt doses put their system prompt on the dose side only). MMLU: letter reads
(argmax of P(A..D)) on the 200 plain items of syco_items_v1. Hook doses (steering, placebo, capping) and the M_hf
engine control also get their EM answers here (HF sampling, T 1.0, no top-k/top-p, <= 400 tokens). B last.
  python -m vibes.e62_pull.hf_eval --out <results> --work /root/e62 --mode main [--tiny]"""
import argparse
import json
import time
from pathlib import Path

import torch

from vibes.e62_pull.common import (CAP_H, H_STEER, MODEL, N_MAIN, N_PILOT, TINY, chat, doses, em_jobs, kl_prompts,
                                   mmlu_items, mmlu_text, n_samples_for, seed_of, write_jsonl)

GEN_BS = 200


def install_gqa_decode():
    """Copied from state_attack/src/server.py: with a padding mask, transformers' sdpa path expands K/V (repeat_kv) at
    every decode step; at q_len 1 compute the grouped attention directly. Prefill goes to the stock function."""
    import transformers.integrations.sdpa_attention as SA
    from transformers import AttentionInterface
    orig = SA.sdpa_attention_forward

    def fwd(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kw):
        g = getattr(module, "num_key_value_groups", 1)
        if query.shape[2] != 1 or g == 1 or kw.get("position_bias") is not None:
            return orig(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                        is_causal=is_causal, **kw)
        B, Hq, _, Dh = query.shape
        Hkv, T = key.shape[1], key.shape[2]
        sc = scaling if scaling is not None else Dh ** -0.5
        q = query.reshape(B, Hkv, g, Dh)
        sco = torch.matmul(q, key.transpose(-1, -2)).float() * sc
        if attention_mask is not None:
            m = attention_mask[..., :T]
            sco = sco.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else sco + m.float()
        p_ = torch.softmax(sco, -1).to(value.dtype)
        o = torch.matmul(p_, value)
        return o.reshape(B, Hq, 1, Dh).transpose(1, 2).contiguous(), None

    AttentionInterface.register("sdpa", fwd)


class Evaluator:
    def __init__(self, model, tok, dev, a):
        self.model, self.tok, self.dev, self.a = model, tok, dev, a
        self.eos = tok.convert_tokens_to_ids("<|im_end|>")
        self.state = {"steer": None, "cap": None}

    # ---------------- hooks (registered once on the plain model; persist under a PEFT wrapper)
    def register_hooks(self, stats):
        axis = {h: stats["axis"][h].to(self.dev).float() for h in CAP_H}
        self.axis, self.stats = axis, stats
        st = self.state

        def steer_hook(mod, inp, outp):
            if st["steer"] is None:
                return outp
            hs = outp[0] if isinstance(outp, tuple) else outp
            hs2 = hs + st["steer"].to(hs.dtype)
            return (hs2,) + tuple(outp[1:]) if isinstance(outp, tuple) else hs2

        def cap_hook(h):
            ax = axis[h]

            def hook(mod, inp, outp):
                if st["cap"] is None:
                    return outp
                hs = outp[0] if isinstance(outp, tuple) else outp
                d = (st["cap"][h] - hs.float() @ ax).clamp(min=0)
                hs2 = hs + (d[..., None] * ax).to(hs.dtype)
                return (hs2,) + tuple(outp[1:]) if isinstance(outp, tuple) else hs2
            return hook
        layers = self.model.model.layers
        layers[H_STEER - 1].register_forward_hook(steer_hook)
        for h in CAP_H:
            layers[h - 1].register_forward_hook(cap_hook(h))

    def set_hook(self, hook):
        self.state["steer"], self.state["cap"] = None, None
        if hook is None:
            return
        kind, x = hook
        if kind in ("steer", "placebo"):
            unit = self.axis[H_STEER] if kind == "steer" else self.stats["placebo"].to(self.dev).float()
            self.state["steer"] = x * self.stats["sigma20"] * unit
        elif kind == "cap":
            self.state["cap"] = self.stats["tau"][x]

    # ---------------- KL
    @torch.no_grad()
    def logprobs(self, model, prompts, conts, system=None, bs=10):
        """List of [C_i, V] float32 CPU log-prob tensors at the continuation positions."""
        model.eval()
        out = []
        for st in range(0, len(prompts), bs):
            seqs, starts = [], []
            for p, c in zip(prompts[st:st + bs], conts[st:st + bs]):
                pid = self.tok(chat(self.tok, p, system), add_special_tokens=False)["input_ids"]
                seqs.append(pid + c); starts.append(len(pid))
            L = max(map(len, seqs))
            ids = torch.full((len(seqs), L), self.tok.pad_token_id)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
            logits = model(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev)).logits
            for j, (s, p0) in enumerate(zip(seqs, starts)):
                C = len(s) - p0
                out.append(torch.log_softmax(logits[j, p0 - 1:p0 - 1 + C].float(), -1).cpu())
        return out

    def kl(self, model, ref, prompts, conts, system=None):
        lp = self.logprobs(model, prompts, conts, system)
        per, per_n, tot, n = [], [], 0.0, 0
        for r, d in zip(ref, lp):
            k = (r.exp() * (r - d)).sum(-1)               # per token
            per.append(float(k.mean())); per_n.append(k.numel()); tot += float(k.sum()); n += k.numel()
        return {"kl": tot / n, "per_prompt": per, "per_prompt_n": per_n, "n_tok": n}

    # ---------------- MMLU letter reads
    @torch.no_grad()
    def mmlu(self, model, items, system=None, bs=25):
        model.eval()
        self.tok.padding_side = "left"
        letters = [self.tok.encode(L, add_special_tokens=False)[0] for L in "ABCD"]
        P = []
        for st in range(0, len(items), bs):
            enc = self.tok([chat(self.tok, mmlu_text(x), system) for x in items[st:st + bs]], return_tensors="pt",
                           padding=True, add_special_tokens=False).to(self.dev)
            pr = torch.softmax(model(**enc).logits[:, -1, :].float(), -1)[:, letters]
            P.append(pr.cpu())
        P = torch.cat(P)
        ok = (P.argmax(-1) == torch.tensor([x["answer"] for x in items])).float()
        return {"acc": float(ok.mean()), "n": len(items), "letter_mass": float(P.sum(-1).mean()),
                "pred": P.argmax(-1).tolist()}

    # ---------------- sampling
    @torch.no_grad()
    def sample(self, model, texts, max_new, seed, system=None):
        model.eval()
        self.tok.padding_side = "left"
        res = []
        for st in range(0, len(texts), GEN_BS):
            torch.manual_seed(seed + st)
            enc = self.tok([chat(self.tok, t, system) for t in texts[st:st + GEN_BS]], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.dev)
            o = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                               pad_token_id=self.tok.pad_token_id)
            stops = (self.eos, self.tok.pad_token_id)          # <|im_end|>, <|endoftext|> (also the pad id)
            for row in o[:, enc["input_ids"].shape[1]:].tolist():
                k = next((i for i, t in enumerate(row) if t in stops), None)
                res.append((row[:k + 1] if k is not None else row, k is None))
        return res

    def gen_em(self, model, dose, n, out, max_new):
        jobs = em_jobs(dose, n)
        t0 = time.time()
        res = self.sample(model, [j["question"] for j in jobs], max_new, seed_of("hfgen", dose) % 2**31)
        rows = []
        for j, (ids, trunc) in zip(jobs, res):
            ids = [i for i in ids if i not in (self.eos, self.tok.pad_token_id)]
            rows.append({**{k: v for k, v in j.items() if k != "seed"}, "reply": self.tok.decode(ids, skip_special_tokens=True).strip(),
                         "truncated": trunc, "n_tok": len(ids), "engine": "hf"})
        write_jsonl(out / "replies_em" / f"{dose}.jsonl", rows)
        print(f"  [em] {dose}: {len(rows)} answers in {time.time() - t0:.0f}s, {sum(r['truncated'] for r in rows)} truncated",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--mode", default="main", choices=["main", "pilot"])
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these doses (debugging)")
    ap.add_argument("--no-gqa-fix", action="store_true")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    if not a.no_gqa_fix:
        install_gqa_decode()
    N = N_MAIN if a.mode == "main" else N_PILOT
    n_kl, n_mm = (3, 4) if a.tiny else (N["kl_prompts"], N["mmlu"])
    max_new = 16 if a.tiny else 400
    tok = AutoTokenizer.from_pretrained(work / "M")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(work / "M", dtype=dtype).to(dev).eval()
    print(f"[hf] M loaded in {time.time() - t0:.0f}s", flush=True)
    ev = Evaluator(model, tok, dev, a)
    stats = torch.load(work / "axis_stats.pt")
    ev.register_hooks(stats)
    spec = doses()
    want = lambda d: a.only is None or d in a.only  # noqa: E731

    prompts, items = kl_prompts(n_kl), mmlu_items(n_mm)
    t0 = time.time()
    cs = ev.sample(model, prompts, 64, 6262)
    conts = [c for c, _ in cs]
    write_jsonl(out / "kl_conts.jsonl", [{"prompt": p, "cont_ids": c, "cont": tok.decode(c)} for p, c in zip(prompts, conts)])
    ref = ev.logprobs(model, prompts, conts)
    print(f"[hf] M continuations + reference in {time.time() - t0:.0f}s ({sum(map(len, conts))} tokens)", flush=True)
    KL, MM = {}, {}

    def dump():
        (out / "kl.json").write_text(json.dumps(KL))
        (out / "mmlu.json").write_text(json.dumps(MM))

    def measure(d, m, system=None):
        t1 = time.time()
        KL[d] = ev.kl(m, ref, prompts, conts, system)
        MM[d] = ev.mmlu(m, items, system)
        dump()
        print(f"[hf] {d}: KL {KL[d]['kl']:.4f}, MMLU {MM[d]['acc']:.3f} ({time.time() - t1:.0f}s)", flush=True)

    measure("M", model)
    if want("M_hf"):
        ev.gen_em(model, "M_hf", 1 if a.tiny else n_samples_for("M_hf", a.mode), out, max_new)
    for d in ("a1", "a2"):
        if want(d):
            measure(d, model, spec[d]["system"])
    for d, s in spec.items():
        if s.get("hook") and want(d):
            ev.set_hook(s["hook"])
            measure(d, model)
            ev.gen_em(model, d, 1 if a.tiny else n_samples_for(d, a.mode), out, max_new)
            ev.set_hook(None)
    from peft import PeftModel
    for d, s in spec.items():
        if s.get("lora") and want(d):
            p = work / "lora" / s["lora"]
            if not p.exists():
                print(f"[hf] {d}: no adapter at {p}; skipped", flush=True)
                continue
            pm = PeftModel.from_pretrained(model, str(p)).eval()
            measure(d, pm)
            model = pm.unload()
    ev.model = None
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    if want("B"):
        t0 = time.time()
        B = AutoModelForCausalLM.from_pretrained(TINY if a.tiny else MODEL, dtype=dtype).to(dev).eval()
        print(f"[hf] B loaded in {time.time() - t0:.0f}s", flush=True)
        measure("B", B)
    print("[hf] done", flush=True)


if __name__ == "__main__":
    main()
