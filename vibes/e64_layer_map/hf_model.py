"""HF Qwen3-14B with e64's hook conditions (T6 steering, T6p placebo, T7 capping) and batched sampling.

Conventions (R12/R18/state_attack): hidden state h = output of decoder layer h-1. Hooks act at every position (prompt
and generated tokens).
  steer / placebo  h += alpha * u at hidden state 20 (q22 pod_steer2.py: layers[19] forward hook); u = the r18
                   assistant axis (T6) or the on-manifold placebo direction (T6p), both unit; alpha = k * sigma.
  cap              r18 capping at hidden states 16-28: h <- h + max(0, tau_h - <h, a_h>) a_h (r18 hf.py, state_attack
                   B6), axis and tau from state_attack/data/axis_r18_qwen3_14b.pt.
The GQA decode fix is copied from state_attack/src/server.py (transformers' sdpa path expands K/V at every decode
step when a padding mask is present; 3-5x slower on long contexts).
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

from vibes.e64_layer_map import conds as C  # noqa: E402

AXIS = C.REPO / "state_attack" / "data" / "axis_r18_qwen3_14b.pt"
H_STEER = 20
CAP_H = list(range(16, 29))


def install_gqa_decode():
    """Copied from state_attack/src/server.py (2026-10-16)."""
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
            if m.dtype == torch.bool:
                sco = sco.masked_fill(~m, float("-inf"))
            else:
                sco = sco + m.float()
        p_ = torch.softmax(sco, -1).to(value.dtype)
        o = torch.matmul(p_, value)
        return o.reshape(B, Hq, 1, Dh).transpose(1, 2).contiguous(), None

    AttentionInterface.register("sdpa", fwd)


class HFModel:
    def __init__(self, model_id=C.MODEL, axis_path=AXIS, device=None, tok_budget=150_000, max_b=64):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        t0 = time.time()
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.dev == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.dev).eval()
        if self.dev == "cuda":
            install_gqa_decode()
        self.D = self.model.config.hidden_size
        self.layers = self.model.model.layers
        ax = torch.load(axis_path)
        self.axis = {h: ax["axis"][h].float().to(self.dev) for h in ax["axis"]}
        self.tau = {h: float(ax["tau"][h]) for h in ax["tau"]}
        assert self.axis[H_STEER].shape[0] == self.D, "axis width does not match the model"
        self.st = {"mode": None, "vec": None, "capture": None}
        for h in CAP_H:
            self.layers[h - 1].register_forward_hook(self._cap_hook(h))
        self.layers[H_STEER - 1].register_forward_hook(self._steer_hook)
        self.layers[H_STEER - 1].register_forward_hook(self._capture_hook)
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.eos.add(self.tok.convert_tokens_to_ids("<|im_end|>"))
        self.tok_budget, self.max_b = tok_budget, max_b
        print(f"[e64-hf] loaded {model_id} on {self.dev} in {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ hooks
    def _cap_hook(self, h):
        def hook(mod, inp, out):
            if self.st["mode"] != "cap":
                return out
            hs = out[0] if isinstance(out, tuple) else out
            d = (self.tau[h] - hs.float() @ self.axis[h]).clamp(min=0)
            hs2 = hs + (d[..., None] * self.axis[h]).to(hs.dtype)
            return (hs2,) + tuple(out[1:]) if isinstance(out, tuple) else hs2
        return hook

    def _steer_hook(self, mod, inp, out):
        if self.st["mode"] != "steer":
            return out
        hs = out[0] if isinstance(out, tuple) else out
        hs2 = hs + self.st["vec"].to(hs.dtype)
        return (hs2,) + tuple(out[1:]) if isinstance(out, tuple) else hs2

    def _capture_hook(self, mod, inp, out):
        if self.st["capture"] is not None:
            hs = out[0] if isinstance(out, tuple) else out
            self.st["capture"].append(hs.detach())
        return None

    def set_condition(self, mode=None, vec=None):
        """mode None (plain), 'steer' (vec = alpha * unit direction, [D]) or 'cap'."""
        assert mode in (None, "steer", "cap")
        self.st["mode"] = mode
        self.st["vec"] = None if vec is None else vec.float().to(self.dev)

    # ------------------------------------------------------------------ helpers
    def render(self, msgs, tools=None):
        return self.tok.apply_chat_template(msgs, tools=tools or None, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)

    def _batches(self, lens, extra):
        order = sorted(range(len(lens)), key=lambda i: -lens[i])
        out, cur = [], []
        for i in order:
            m = max([lens[j] for j in cur] + [lens[i]]) + extra
            if cur and (len(cur) + 1 > self.max_b or (len(cur) + 1) * m > self.tok_budget):
                out.append(cur)
                cur = []
            cur.append(i)
        if cur:
            out.append(cur)
        return out

    @torch.no_grad()
    def generate(self, texts, max_new, seed, temperature=0.7, top_p=0.95, top_k=20, log=None):
        """Sample one completion per text. Returns [(text, finish)] with finish 'stop' or 'length'. Special tokens
        other than the end-of-turn ones are kept (tool-call markup), as vLLM's skip_special_tokens=False."""
        ids = [self.tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        outs = [None] * len(texts)
        queue = list(enumerate(self._batches([len(x) for x in ids], max_new if isinstance(max_new, int) else max(max_new))))
        done = 0
        while queue:
            bi, idx = queue.pop(0)
            mx = max_new if isinstance(max_new, int) else max(max_new[i] for i in idx)
            try:
                self._gen_one(idx, ids, mx, seed * 1000 + bi, temperature, top_p, top_k, outs,
                              None if isinstance(max_new, int) else max_new)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(idx) == 1:
                    outs[idx[0]] = ("", "oom")
                    continue
                h = len(idx) // 2
                print(f"[e64-hf] OOM at batch {len(idx)}; splitting", flush=True)
                queue[:0] = [(bi * 2 + 100, idx[:h]), (bi * 2 + 101, idx[h:])]
                continue
            done += len(idx)
            if log:
                log(f"[e64-hf] generated {done}/{len(texts)}")
        return outs

    def _gen_one(self, idx, ids, mx, seed, temperature, top_p, top_k, outs, per_item_max):
        T = max(len(ids[i]) for i in idx)
        pad = self.tok.pad_token_id
        inp = torch.tensor([[pad] * (T - len(ids[i])) + ids[i] for i in idx], device=self.dev)
        att = torch.tensor([[0] * (T - len(ids[i])) + [1] * len(ids[i]) for i in idx], device=self.dev)
        torch.manual_seed(seed)
        kw = dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k) if temperature > 0 else \
            dict(do_sample=False)
        o = self.model.generate(input_ids=inp, attention_mask=att, max_new_tokens=mx, pad_token_id=pad, **kw)
        gen = o[:, T:].tolist()
        for j, i in enumerate(idx):
            g = gen[j]
            k = next((t for t, x in enumerate(g) if x in self.eos), None)
            toks = g[:k] if k is not None else g
            lim = per_item_max[i] if per_item_max else mx
            fin = "stop" if (k is not None and k < lim) else "length"
            toks = toks[:lim]
            text = self.tok.decode(toks, skip_special_tokens=False)
            for s in ("<|im_end|>", "<|endoftext|>"):
                text = text.replace(s, "")
            outs[i] = (text, fin)

    @torch.no_grad()
    def next_token_probs(self, texts, token_ids, batch=32):
        """P(token) at the next position for each text (softmax over the full vocabulary), [n, len(token_ids)]."""
        res = []
        for b in range(0, len(texts), batch):
            enc = self.tok(texts[b:b + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
            lg = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            res.append(torch.softmax(lg, -1)[:, token_ids].cpu())
        return torch.cat(res)

    @torch.no_grad()
    def answer_states(self, prompts, completions, batch=8):
        """Teacher-forced hidden state 20 at each completion token (no hook condition): list of [n_tok, D] float."""
        assert self.st["mode"] is None
        out = []
        for b in range(0, len(prompts), batch):
            ps = [self.tok(p, add_special_tokens=False)["input_ids"] for p in prompts[b:b + batch]]
            cs = [self.tok(c, add_special_tokens=False)["input_ids"] for c in completions[b:b + batch]]
            full = [p + c for p, c in zip(ps, cs)]
            T = max(len(x) for x in full)
            pad = self.tok.pad_token_id
            inp = torch.tensor([[pad] * (T - len(x)) + x for x in full], device=self.dev)
            att = torch.tensor([[0] * (T - len(x)) + [1] * len(x) for x in full], device=self.dev)
            self.st["capture"] = []
            self.model(input_ids=inp, attention_mask=att, logits_to_keep=1)
            hs = self.st["capture"][0].float()
            self.st["capture"] = None
            for j, c in enumerate(cs):
                out.append(hs[j, T - len(c):].cpu() if c else torch.zeros(0, self.D))
        return out
