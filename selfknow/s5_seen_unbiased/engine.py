"""s5 engine (HF transformers): last-token reads, residual steering hooks, the A/B logit readout, cached sampling and
the two stated-value probes. Works on Qwen3 (main) and on Qwen3-0.6B on CPU (local tests, same code path).

Conventions
- Layer index h is the hidden_states index: h = output of decoder block h-1 (hook on layers[h-1]); h = 1..n_layers.
- Steering adds one vector at layer h to EVERY position of every forward pass while it is set (prompt, sampled
  tokens and probe branches alike).
- Batches are left-padded; position ids = cumsum(mask) - 1, continued by +1 per new token (as HF generate does).
"""
import time

import numpy as np
import torch


def log(*a):
    print(f"[s5 {time.strftime('%H:%M:%S')}]", *a, flush=True)


def find_layers(model, n_layers):
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n_layers and name.endswith("layers"):
            if best is None or "language_model" in name:
                best = (name, mod)
    if best is None:
        raise RuntimeError("decoder layer list not found")
    return best[1]


def sample_tokens(lg, gen, temperature=0.7, top_k=20, top_p=0.8):
    """Qwen3 non-thinking sampling defaults. lg: [N, V] float32 logits."""
    v, ix = (lg / temperature).topk(top_k, -1)
    pr = torch.softmax(v, -1)
    keep = (pr.cumsum(-1) - pr) < top_p          # keep tokens until the mass before them reaches top_p
    pr = pr * keep
    pr = pr / pr.sum(-1, keepdim=True)
    j = torch.multinomial(pr, 1, generator=gen)
    return ix.gather(-1, j).squeeze(-1)


class Engine:
    def __init__(self, model_id, read_layers=(), device=None, dtype=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or (torch.bfloat16 if self.dev == "cuda" else torch.float32)
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=self.dev)
        except Exception as e:  # noqa: BLE001  (multimodal checkpoints, e.g. Gemma 4)
            from transformers import AutoModelForImageTextToText
            log(f"AutoModelForCausalLM failed ({type(e).__name__}); using ImageTextToText")
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, device_map=self.dev)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.model.config.get_text_config()
        self.nl, self.D = cfg.num_hidden_layers, cfg.hidden_size
        self.layers = find_layers(self.model, self.nl)
        self.read_layers = sorted(read_layers)
        self.cap = None          # dict h -> [B, D] float32 cpu (last position of the prefill) while capturing
        self.steer = None        # (h, vec[D] on device)
        for h in range(1, self.nl + 1):
            self.layers[h - 1].register_forward_hook(self._hook(h))
        t = self.tok
        one = lambda s: t.encode(s, add_special_tokens=False)  # noqa: E731
        self.id_A, self.id_B = one(" A"), one(" B")
        assert len(self.id_A) == 1 and len(self.id_B) == 1, "' A'/' B' must be single tokens"
        self.id_A, self.id_B = self.id_A[0], self.id_B[0]
        self.digits = [one(str(d)) for d in range(10)]
        assert all(len(x) == 1 for x in self.digits), "digits must be single tokens"
        self.digits = [x[0] for x in self.digits]
        self.yes = sorted({one(w)[0] for w in ("Yes", "yes", " Yes", " yes") if len(one(w)) == 1})
        self.no = sorted({one(w)[0] for w in ("No", "no", " No", " no") if len(one(w)) == 1})
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        for s in ("<|im_end|>", "<|endoftext|>", "<turn|>", "<eos>"):
            i = t.convert_tokens_to_ids(s)
            if isinstance(i, int) and i != t.unk_token_id:
                self.eos.add(i)
        self.eos.discard(None)
        self.pad_id = t.pad_token_id if t.pad_token_id is not None else min(self.eos)
        self.eos_t = torch.tensor(sorted(self.eos), device=self.dev)
        log(f"loaded {model_id} on {self.dev} ({dtype}): layers {self.nl}, D {self.D}, A/B {self.id_A}/{self.id_B}, "
            f"eos {sorted(self.eos)}, yes {self.yes}, no {self.no}")

    # ------------------------------------------------------------------ hooks
    def _hook(self, h):
        def f(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            st = self.steer
            if st is not None and st[0] == h:
                hs = hs + st[1].to(hs.dtype)
            if self.cap is not None and h in self.read_layers and hs.shape[1] > 1:
                self.cap[h] = hs[:, -1, :].detach().float().cpu()
            if st is not None and st[0] == h:
                return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
            return None
        return f

    def set_steer(self, h=None, vec=None):
        self.steer = None if vec is None else (h, torch.as_tensor(vec, dtype=torch.float32, device=self.dev))

    # ------------------------------------------------------------------ text helpers
    def render(self, msgs):
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def ids(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def suffix_ids(self, question):
        """Token ids that close the assistant turn and ask `question` as the next user turn, ending at the next
        assistant generation prompt (thinking off). The decision turn is continued token by token (same branch)."""
        m = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "XXMARKXX"},
             {"role": "user", "content": question}]
        s = self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.ids(s[s.index("XXMARKXX") + len("XXMARKXX"):])

    def pad(self, seqs):
        B, T = len(seqs), max(len(s) for s in seqs)
        ids = torch.full((B, T), self.pad_id, dtype=torch.long)
        mask = torch.zeros((B, T), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, T - len(s):] = torch.tensor(s)
            mask[i, T - len(s):] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        return ids.to(self.dev), mask.to(self.dev), pos.to(self.dev)

    @staticmethod
    def batches(lens, budget, max_b):
        order = sorted(range(len(lens)), key=lambda i: -lens[i])
        out, cur = [], []
        for i in order:
            m = max([lens[j] for j in cur] + [lens[i]])
            if cur and (len(cur) + 1 > max_b or (len(cur) + 1) * m > budget):
                out.append(cur)
                cur = []
            cur.append(i)
        if cur:
            out.append(cur)
        return out

    # ------------------------------------------------------------------ decision: readout + samples
    @torch.no_grad()
    def decide(self, seqs, n_samples=0, max_new=64, seed=0, capture=False, greedy=False):
        """seqs: token-id lists ending in the 'Award:' prefill. Returns per row: pA, pB, mass, top token, reads
        (if capture) and n_samples sampled continuations (token ids, without the end token)."""
        ids, mask, pos = self.pad(seqs)
        B = ids.shape[0]
        self.cap = {} if capture else None
        try:
            out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=n_samples > 0,
                             logits_to_keep=1)
        finally:
            cap, self.cap = self.cap, None
        lg = out.logits[:, -1, :].float()
        p = torch.softmax(lg, -1)
        res = [{"pA": float(p[i, self.id_A]), "pB": float(p[i, self.id_B]),
                "top": self.tok.decode([int(p[i].argmax())])} for i in range(B)]
        for r in res:
            r["mass"] = r["pA"] + r["pB"]
        if capture:
            for i in range(B):
                res[i]["reads"] = np.stack([cap[h][i].numpy() for h in self.read_layers]).astype(np.float16)
        if n_samples:
            cache = out.past_key_values
            cache.batch_repeat_interleave(n_samples)
            lg = lg.repeat_interleave(n_samples, 0)
            m = mask.repeat_interleave(n_samples, 0)
            last = pos[:, -1].repeat_interleave(n_samples, 0)
            gen = torch.Generator(device=self.dev).manual_seed(int(seed))
            N = lg.shape[0]
            done = torch.zeros(N, dtype=torch.bool, device=self.dev)
            toks = []
            for step in range(max_new):
                nxt = lg.argmax(-1) if greedy else sample_tokens(lg, gen)
                nxt = torch.where(done, torch.full_like(nxt, self.pad_id), nxt)
                toks.append(nxt)
                done = done | torch.isin(nxt, self.eos_t)
                if bool(done.all()) or step == max_new - 1:
                    break
                m = torch.cat([m, torch.ones((N, 1), dtype=m.dtype, device=self.dev)], 1)
                last = last + 1
                o = self.model(input_ids=nxt[:, None], attention_mask=m, position_ids=last[:, None],
                               past_key_values=cache, use_cache=True)
                lg = o.logits[:, -1, :].float()
            del cache, out
            T = torch.stack(toks, 1).tolist()
            for i in range(B):
                ss = []
                for j in range(n_samples):
                    row = T[i * n_samples + j]
                    k = next((t for t, x in enumerate(row) if x in self.eos), None)
                    ss.append({"ids": row[:k] if k is not None else row, "finished": k is not None})
                for s in ss:
                    s["text"] = self.tok.decode(s["ids"], skip_special_tokens=True)
                res[i]["samples"] = ss
        return res

    # ------------------------------------------------------------------ probes
    @torch.no_grad()
    def probe_s1(self, seqs):
        """Expected 0-100 answer. Digits are single tokens, so the value distribution is read as a digit tree:
        P(d1) at the answer position; for d1 = 1..9 one extra token gives P(d2 | d1) and P(stop | d1) (stop = any
        non-digit); for '10' a third step splits 10 from 100. Two-digit answers are taken as complete."""
        ids, mask, pos = self.pad(seqs)
        B, T = ids.shape
        out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)
        p0 = torch.softmax(out.logits[:, -1, :].float(), -1)
        dg = torch.tensor(self.digits, device=self.dev)
        pd1 = p0[:, dg]                                           # [B, 10]
        top = [self.tok.decode([int(x)]) for x in p0.argmax(-1)]
        cache = out.past_key_values
        last = pos[:, -1]
        cont = {}
        for key, toks in [(d, [self.digits[d]]) for d in range(1, 10)] + [("10", [self.digits[1], self.digits[0]])]:
            L = len(toks)
            inp = torch.tensor(toks, device=self.dev).unsqueeze(0).expand(B, L)
            m2 = torch.cat([mask, torch.ones((B, L), dtype=mask.dtype, device=self.dev)], 1)
            p2 = last[:, None] + 1 + torch.arange(L, device=self.dev)[None, :]
            o = self.model(input_ids=inp, attention_mask=m2, position_ids=p2, past_key_values=cache, use_cache=True,
                           logits_to_keep=1)
            cont[key] = torch.softmax(o.logits[:, -1, :].float(), -1)[:, dg]
            cache.crop(T)
        del cache, out
        vals = torch.zeros((B, 101), device=self.dev)
        vals[:, 0] = pd1[:, 0]
        for d in range(1, 10):
            q = cont[d]
            vals[:, d] = pd1[:, d] * (1 - q.sum(-1)).clamp(min=0)
            for e in range(10):
                vals[:, 10 * d + e] = pd1[:, d] * q[:, e]
        q10 = cont["10"]
        p10 = vals[:, 10].clone()
        vals[:, 100] = p10 * q10[:, 0]
        vals[:, 10] = p10 * (1 - q10.sum(-1)).clamp(min=0)
        tot = vals.sum(-1)
        E = (vals * torch.arange(101, device=self.dev)[None, :]).sum(-1) / tot.clamp(min=1e-12)
        return [{"E": float(E[i]), "mass": float(pd1[i].sum()), "valid": float(tot[i]), "top": top[i],
                 "p100": float(vals[i, 100] / tot[i].clamp(min=1e-12))} for i in range(B)]

    @torch.no_grad()
    def probe_s2(self, seqs):
        ids, mask, pos = self.pad(seqs)
        out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=1)
        p = torch.softmax(out.logits[:, -1, :].float(), -1)
        y = p[:, self.yes].sum(-1)
        n = p[:, self.no].sum(-1)
        top = [self.tok.decode([int(x)]) for x in p.argmax(-1)]
        return [{"pyes": float(y[i] / (y[i] + n[i]).clamp(min=1e-12)), "mass": float(y[i] + n[i]), "top": top[i]}
                for i in range(len(seqs))]

    # ------------------------------------------------------------------ coherence: unsteered NLL of a reply alone
    @torch.no_grad()
    def nll(self, texts, budget=2000, max_b=64):
        assert self.steer is None
        seqs = [self.ids(t) for t in texts]
        res = [None] * len(seqs)
        for idx in self.batches([len(s) for s in seqs], budget, max_b):
            ids, mask, pos = self.pad([seqs[i] for i in idx])
            lg = self.model(input_ids=ids, attention_mask=mask, position_ids=pos).logits.float()
            lp = torch.log_softmax(lg[:, :-1], -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
            mk = mask[:, 1:].float() * mask[:, :-1].float()
            s = -(lp * mk).sum(-1) / mk.sum(-1).clamp(min=1)
            for j, i in enumerate(idx):
                res[i] = float(s[j]) if len(seqs[i]) > 1 else 0.0
            del lg, lp
        return res
