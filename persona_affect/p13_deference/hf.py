"""PA-P13 HF runner (plan persona_affect/plans/P13.md, Task 4): one transformers model with per-row hooks on every decoder
layer (hooks.RowHooks), used by pod A (directions, gates, base log-odds, alpha pilot) and pod B (means, teacher-forced
arms, reads, hooked episodes). The model is loaded with vigilante.src.steer_run.load (bf16, cuda) and its decoder layers
found with find_layers (Gemma 4: model.language_model.layers). Layer L means the output of decoder layer L, 0-based,
everywhere: Capture's layers, l_add and the reference means.

Batches (Global Constraints): rows sorted (by group, then) by token length and filled up to a budget of prompt plus new
tokens, Gemma 16,000 tokens and 8 rows, Qwen 48,000 and 16. Left padding throughout (Qwen3.6's tokenizer pads on the
right by default, so it is set here); each row's pad count goes to RowHooks, Capture and proj_sums. An out-of-memory
batch is halved and retried outside the `except` block (the R20 pod_b2.generate_safe lesson); a single row that still
does not fit gives ("", "oom") in generate and None in score and nll, and raises in capture and proj_all_layers, whose
means cannot do without it.

Teacher-forced log-odds: a prefix is the rendered chat plus the opener (Gemma `<|tool_call>call:`, Qwen
`<tool_call>\\n<function=`), the continuations are MONEY[cid] and request_approval with `{` (Gemma) or `>` (Qwen).
split_continuations scores both from one shared position (Qwen3.6 writes "=request" as one token, so there the shared
position is one before the end of the prefix). score() keeps logits_to_keep = the longest continuation + 1, takes the
log-softmax in float32 and reads each row with cont_logprob, which lines the kept rows up from the end.
"""
from __future__ import annotations

import gc

import numpy as np
import torch

from persona_affect.p13_deference import cases as K
from persona_affect.p13_deference.hooks import (SKIP, Capture, RowHooks, cont_logprob, proj_sums, ref_means,
                                                rms_removed, split_continuations)
from vigilante.src.episode import init_episode
from vigilante.src.sim import Call
from vigilante.src.templates import clean_output, to_template

OPENER = {"gemma4": "<|tool_call>call:", "qwen": "<tool_call>\n<function="}
SUFFIX = {"gemma4": "{", "qwen": ">"}
BUDGET = {"gemma4": (16_000, 8), "qwen": (48_000, 16)}       # tokens (prompt + new), rows
GEMMA_EOS = (1, 106, 50)                                      # Gemma 4's generation_config
DECODING = dict(temperature=0.7, top_p=0.95, top_k=20)        # vigilante.src.pod_run.DECODING
EXTRA_STRIP = ("<pad>",)


def eos_ids(tok, fmt: str) -> list[int]:
    if fmt == "gemma4":
        return list(GEMMA_EOS)
    return [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]


def finish_of(gen: list[int], eos) -> tuple[list[int], str]:
    """The generated ids up to the first end-of-turn id and "stop", or all of them and "length". Reading the first EOS
    (not stripping trailing pads) also works for Qwen, whose pad id is one of its EOS ids."""
    k = next((i for i, x in enumerate(gen) if x in eos), None)
    return (list(gen[:k]), "stop") if k is not None else (list(gen), "length")


def batches(lengths: list[int], extra: int, budget: int, max_rows: int, groups: list | None = None) -> list[list[int]]:
    """Row indices in batches: sorted by group (if given), then length, and filled while rows x (longest row + extra)
    stays within the budget and rows within max_rows. A row over the budget on its own runs alone."""
    order = sorted(range(len(lengths)), key=lambda i: ((groups[i] if groups else ""), lengths[i]))
    out, cur, longest = [], [], 0
    for i in order:
        top = max(longest, lengths[i])
        if cur and (len(cur) + 1 > max_rows or (len(cur) + 1) * (top + extra) > budget):
            out.append(cur)
            cur, top = [], lengths[i]
        cur.append(i)
        longest = top
    if cur:
        out.append(cur)
    return out


def prefix_messages(job: dict, pre: list, fmt: str) -> tuple[list[dict], list]:
    """The episode's opening messages (init_episode: system prompt, any history, the task) followed by the pre-decision
    calls `pre` ([[name, args], ...]), each as the model writes it (cases.render_call) and answered by the case's own
    responders, in one state, as the engine answers them. Returns (messages, tools)."""
    from vigilante.cases import ALL
    ep = init_episode(job)
    case = ALL[job["case"]]
    state = case.new_state()
    msgs = list(ep["messages"])
    for name, args in pre:
        msgs.append({"role": "assistant", "content": K.render_call(name, args, fmt)})
        msgs.append({"role": "tool", "name": name, "content": case.respond(Call(name, dict(args)), state)})
    return msgs, ep["tools"]


def decision_conts(cid: str, fmt: str) -> list[str]:
    """The two teacher-forced continuations after the opener: the money tool, then request_approval."""
    return [K.MONEY[cid] + SUFFIX[fmt], K.COMPLY + SUFFIX[fmt]]


# ---------------------------------------------------------------- GQA decode patch (vibes/e64_layer_map/hf_model.py)

_ORIG: dict = {}


def install_gqa_decode():
    """Copied from vibes/e64_layer_map/hf_model.py (itself from state_attack/src/server.py): transformers' sdpa path
    expands K/V at every decode step when a padding mask is present; this computes grouped-query decode steps directly.
    The original function is kept so uninstall_gqa_decode() can put it back."""
    from transformers import AttentionInterface
    if "sdpa" in _ORIG:
        return
    orig = AttentionInterface._global_mapping["sdpa"]
    _ORIG["sdpa"] = orig

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


def uninstall_gqa_decode():
    from transformers import AttentionInterface
    if "sdpa" in _ORIG:
        AttentionInterface.register("sdpa", _ORIG.pop("sdpa"))


# ---------------------------------------------------------------- the runner

class HF:
    """One model, its tokenizer and a RowHooks on every decoder layer (l_add = L* for the addition arms).

    tok and model may be given (tests: a tiny random model); otherwise steer_run.load(model_id) loads the model."""

    def __init__(self, model_id: str, fmt: str, l_add: int | None, *, tok=None, model=None):
        if fmt not in OPENER:
            raise ValueError(f"HF: fmt must be one of {sorted(OPENER)}, got {fmt!r}")
        if model is None:
            from vigilante.src.steer_run import load
            tok, model = load(model_id)
        from vigilante.src.steer_run import find_layers
        tok.padding_side = "left"
        self.model_id, self.fmt, self.tok, self.model = model_id, fmt, tok, model
        cfg = model.config.get_text_config()
        self.n_layers, self.D = int(cfg.num_hidden_layers), int(cfg.hidden_size)
        _, self.layers = find_layers(model, self.n_layers)
        self.hooks = RowHooks(self.layers, l_add)
        self.eos = eos_ids(tok, fmt)
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else self.eos[0]
        self.budget, self.max_rows = BUDGET[fmt]
        self.device = next(model.parameters()).device

    # -------------------------------------------------------------- text
    def set_l_add(self, l_add: int | None) -> None:
        self.hooks.remove()
        self.hooks = RowHooks(self.layers, l_add)

    def ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def split(self, prefix: str, conts: list[str]) -> tuple[list[list[int]], int]:
        return split_continuations(self.ids, prefix, conts)

    def render(self, messages: list[dict], tools) -> str:
        return self.tok.apply_chat_template(to_template(messages, self.fmt), tools=tools or None, tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)

    # -------------------------------------------------------------- plumbing
    def _pad(self, rows: list[list[int]]):
        t_len = max(len(r) for r in rows)
        pads = [t_len - len(r) for r in rows]
        ids = torch.tensor([[self.pad_id] * p + r for p, r in zip(pads, rows)], device=self.device)
        att = torch.tensor([[0] * p + [1] * len(r) for p, r in zip(pads, rows)], device=self.device)
        return ids, att, pads

    def _batches(self, lengths: list[int], extra: int = 0) -> list[list[int]]:
        return batches(lengths, extra, self.budget, self.max_rows)

    def _safe(self, fn, idx: list[int], single):
        """fn(idx) -> one result per row; on an out-of-memory error, halve and retry outside the except block; a single
        row that still fails gives single(row)."""
        try:
            return fn(idx)
        except torch.OutOfMemoryError:
            pass
        self.hooks.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(idx) == 1:
            print("[hf] OOM on a single row", flush=True)
            return [single(idx[0])]
        h = len(idx) // 2
        print(f"[hf] OOM at batch {len(idx)}; splitting", flush=True)
        return self._safe(fn, idx[:h], single) + self._safe(fn, idx[h:], single)

    @staticmethod
    def _no_row(i):
        raise RuntimeError(f"[hf] a single row ({i}) does not fit in memory; its means cannot be dropped")

    # -------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, texts: list[str], arms: list, max_new: int, seed: int) -> list[tuple[str, str]]:
        """One sampled completion per text (temperature 0.7, top_p 0.95, top_k 20), each row hooked by its own Arm:
        [(text, finish)], finish "stop", "length" or "oom". The text is cleaned (clean_output, <pad>) and keeps the
        tool-call markup."""
        ids = [self.ids(t) for t in texts]
        out: list = [None] * len(texts)
        for bi, idx in enumerate(self._batches([len(x) for x in ids], max_new)):
            res = self._safe(lambda sub: self._gen(sub, ids, arms, max_new, seed + 7919 * bi + sub[0]), idx,
                             lambda i: ("", "oom"))
            for i, r in zip(idx, res):
                out[i] = r
        return out

    def _gen(self, sub, ids, arms, max_new, seed):
        inp, att, pads = self._pad([ids[i] for i in sub])
        self.hooks.set([arms[i] for i in sub], pads)
        try:
            torch.manual_seed(seed)
            o = self.model.generate(input_ids=inp, attention_mask=att, max_new_tokens=max_new, do_sample=True,
                                    eos_token_id=self.eos, pad_token_id=self.pad_id, **DECODING)
        finally:
            self.hooks.clear()
        res = []
        for row in o[:, inp.shape[1]:].tolist():
            toks, fin = finish_of(row, set(self.eos))
            text = clean_output(self.tok.decode(toks, skip_special_tokens=False), self.fmt)
            for s in EXTRA_STRIP:
                text = text.replace(s, "")
            res.append((text, fin))
        return res

    @torch.no_grad()
    def _greedy(self, texts: list[str], n_tok: int) -> list[list[int]]:
        inp, att, _ = self._pad([self.ids(t) for t in texts])
        o = self.model.generate(input_ids=inp, attention_mask=att, max_new_tokens=n_tok, do_sample=False,
                                eos_token_id=self.eos, pad_token_id=self.pad_id)
        return o[:, inp.shape[1]:].tolist()

    def gqa_check(self, texts: list[str], n_tok: int = 48, force: bool = False) -> bool:
        """Install the GQA decode patch and keep it only if n_tok greedy tokens on `texts` match token for token with and
        without it. Off the GPU it is never installed unless force (tests)."""
        if not (force or torch.cuda.is_available()):
            return False
        self.hooks.clear()
        plain = self._greedy(texts, n_tok)
        install_gqa_decode()
        try:
            patched = self._greedy(texts, n_tok)
        except Exception as e:                       # a transformers change the patch does not fit: run without it
            print(f"[hf] GQA patch failed ({type(e).__name__}: {e}); uninstalled", flush=True)
            uninstall_gqa_decode()
            return False
        if patched == plain:
            print(f"[hf] GQA patch kept: {n_tok} greedy tokens match on {len(texts)} prefixes", flush=True)
            return True
        print("[hf] GQA patch changes greedy tokens; uninstalled", flush=True)
        uninstall_gqa_decode()
        return False

    # -------------------------------------------------------------- teacher forcing
    @torch.no_grad()
    def score(self, prefixes: list[str], conts: list[list[str]], arms: list) -> list[list[float | None]]:
        """For each prefix, the log-probability of each of its continuations (from split_continuations' shared
        position), with that prefix's Arm on its rows: [[lp per continuation]]."""
        rows = []                                                   # (prefix index, continuation index, ids, n)
        for i, (p, cs) in enumerate(zip(prefixes, conts)):
            full, n = self.split(p, cs)
            rows += [(i, j, f, n) for j, f in enumerate(full)]
        res: list = [[None] * len(cs) for cs in conts]
        for idx in self._batches([len(r[2]) for r in rows]):
            vals = self._safe(lambda sub: self._score(sub, rows, arms), idx, lambda k: None)
            for k, v in zip(idx, vals):
                res[rows[k][0]][rows[k][1]] = v
        return res

    def _score(self, sub, rows, arms):
        inp, att, pads = self._pad([rows[k][2] for k in sub])
        keep = max(len(rows[k][2]) - rows[k][3] for k in sub) + 1
        self.hooks.set([arms[rows[k][0]] for k in sub], pads)
        try:
            logits = self.model(input_ids=inp, attention_mask=att, logits_to_keep=keep).logits
        finally:
            self.hooks.clear()
        lp = torch.log_softmax(logits.float(), -1)
        return [cont_logprob(rows[k][2], rows[k][3], lp[b]) for b, k in enumerate(sub)]

    @torch.no_grad()
    def capture(self, texts: list[str], spans: list[tuple[int, int]], layers: list[int]) -> dict[int, np.ndarray]:
        """Unhooked mean over each text's span [start, end) (positions of the text's own tokens, before padding) of the
        outputs of `layers`: {layer: [N, D] float32}."""
        ids = [self.ids(t) for t in texts]
        for x, (s, e) in zip(ids, spans):
            if not 0 <= s < e <= len(x):
                raise ValueError(f"capture: span ({s}, {e}) does not fit a text of {len(x)} tokens")
        self.hooks.clear()
        cap = Capture(self.layers, layers)
        out = {L: [None] * len(texts) for L in cap.which}
        try:
            for idx in self._batches([len(x) for x in ids]):
                vals = self._safe(lambda sub: self._capture(sub, ids, spans, cap), idx, self._no_row)
                for i, v in zip(idx, vals):
                    for L in out:
                        out[L][i] = v[L]
        finally:
            cap.remove()
        return {L: np.stack(v).astype(np.float32) for L, v in out.items()}

    def _capture(self, sub, ids, spans, cap):
        inp, att, pads = self._pad([ids[i] for i in sub])
        cap.set([(spans[i][0] + p, spans[i][1] + p) for i, p in zip(sub, pads)])
        try:
            self.model(input_ids=inp, attention_mask=att, logits_to_keep=1)
            means = cap.means()
        finally:
            cap.clear()
        return [{L: means[L][b].numpy() for L in means} for b in range(len(sub))]

    @torch.no_grad()
    def proj_all_layers(self, texts: list[str], U, skip: int = SKIP, m=None, starts: list[int] | None = None) -> dict:
        """Unhooked projections onto U [k, D] at every layer, streamed through hooks.proj_sums per layer and batch
        (activations are never stored), then hooks.ref_means and hooks.rms_removed (ruling 7: nothing else computes
        these). Positions: t >= skip of each text, or t >= starts[i] when starts is given (e.g. reply tokens).
        Returns {"s1", "s2" [n_layers, k] float64, "n", "means" [n_layers, k], "rms" [n_layers, k]}: the RMS of
        h.u - m, with m [n_layers, k] when given and each layer's own mean otherwise (an SD)."""
        U = torch.as_tensor(np.asarray(U, dtype=np.float32))
        U = U[None] if U.ndim == 1 else U
        ids = [self.ids(t) for t in texts]
        self.hooks.clear()
        parts = []
        for idx in self._batches([len(x) for x in ids]):
            parts += self._safe(lambda sub: [self._proj(sub, ids, U, skip, starts)], idx, self._no_row)
        s1 = sum(p[0] for p in parts)
        s2 = sum(p[1] for p in parts)
        n = sum(p[2] for p in parts)
        means = ref_means(s1, n)
        return {"s1": s1, "s2": s2, "n": n, "means": means, "rms": rms_removed(s1, s2, n, means if m is None else m)}

    def _proj(self, sub, ids, U, skip, starts):
        inp, att, pads = self._pad([ids[i] for i in sub])
        cut, sk = (pads, skip) if starts is None else ([p + starts[i] for p, i in zip(pads, sub)], 0)
        k = U.shape[0]
        s1 = torch.zeros(self.n_layers, k, dtype=torch.float64)
        s2 = torch.zeros_like(s1)
        counts = [0] * self.n_layers

        def mk(li):
            def hook(module, args, out):
                a, b, c = proj_sums(out[0] if isinstance(out, tuple) else out, U, cut, sk)
                s1[li] += a
                s2[li] += b
                counts[li] = c
            return hook
        handles = [layer.register_forward_hook(mk(li)) for li, layer in enumerate(self.layers)]
        try:
            self.model(input_ids=inp, attention_mask=att, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        if len(set(counts)) != 1:
            raise RuntimeError(f"proj_all_layers: layers counted different positions {sorted(set(counts))}")
        return s1, s2, counts[0]

    @torch.no_grad()
    def nll(self, prefixes: list[str], conts: list[str]) -> list[float | None]:
        """Unhooked mean token NLL of each continuation after its prefix, one row at a time; None for an empty one or
        an out-of-memory row."""
        self.hooks.clear()
        out = []
        for p, c in zip(prefixes, conts):
            if not c:
                out.append(None)
                continue
            full, n = self.split(p, [c])
            f = full[0]
            out.append(None if len(f) <= n else self._safe(lambda sub: [self._nll_one(f, n)], [0], lambda i: None)[0])
        return out

    def _nll_one(self, f, n):
        logits = self.model(input_ids=torch.tensor([f], device=self.device), logits_to_keep=len(f) - n + 1).logits
        return -cont_logprob(f, n, torch.log_softmax(logits[0].float(), -1)) / (len(f) - n)

    @torch.no_grad()
    def next_top(self, texts: list[str], k: int = 5) -> list[list[tuple[str, float]]]:
        """Unhooked top-k next tokens after each text, [(token, probability)]."""
        self.hooks.clear()
        ids = [self.ids(t) for t in texts]
        out: list = [None] * len(texts)
        for idx in self._batches([len(x) for x in ids]):
            vals = self._safe(lambda sub: self._top(sub, ids, k), idx, lambda i: [])
            for i, v in zip(idx, vals):
                out[i] = v
        return out

    def _top(self, sub, ids, k):
        inp, att, _ = self._pad([ids[i] for i in sub])
        p = torch.softmax(self.model(input_ids=inp, attention_mask=att, logits_to_keep=1).logits[:, -1].float(), -1)
        top = p.topk(k)
        return [[(self.tok.decode([t]), float(v)) for v, t in zip(top.values[b].tolist(), top.indices[b].tolist())]
                for b in range(len(sub))]
