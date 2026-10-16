"""Shared mechanics for e70-e73 (neurons/DESIGN.md section 0).

Neurons: the inputs to every MLP's down_proj (act(gate) * up), all layers, all neurons.
Attribution at a decision token t: score(n) = a_n(t) * dL/da_n(t), one backward per episode. L at t depends on a_n(t)
only through position t, so the prefix runs under no_grad into the KV / recurrent cache and only the positions of
interest run with grad (`attribute`); `test_local.py` checks this against a full-sequence backward.

Model loading follows readers/common.py (copied, not imported, so this folder stands alone; readers is another card's
folder): Qwen3.6-27B as the text-only CausalLM, bf16, one A100 80GB; the Gated DeltaNet layers run on transformers'
torch fallback."""
import copy
import json
import math
import re
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

MODELS = {
    "qwen36": ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"),
    "qwen14": ("Qwen/Qwen3-14B", None),
    "qwen06": ("Qwen/Qwen3-0.6B", None),
    "tiny36": ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"),   # shrunk random weights, CPU tests
}
SPIDER = "Fact: The number of legs on the animal that spins webs is "
T0 = time.time()


def log(*a):
    print(f"[neurons {time.time() - T0:7.0f}s]", *a, flush=True)


def load_tokenizer(kind):
    from transformers import AutoTokenizer
    name, rev = MODELS[kind]
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_model(kind, device):
    from transformers import AutoConfig, AutoModelForCausalLM
    name, rev = MODELS[kind]
    if kind in ("qwen36", "qwen14"):
        model, info = AutoModelForCausalLM.from_pretrained(name, revision=rev, dtype=torch.bfloat16, device_map=device,
                                                           output_loading_info=True)
        miss = [k for k in info.get("missing_keys", []) if "mtp" not in k]
        log(f"loaded {name}: {type(model).__name__}; missing {len(miss)} {miss[:5]}")
        assert not miss, "text weights missing from the checkpoint mapping"
    elif kind == "tiny36":
        cfg = AutoConfig.from_pretrained(name, revision=rev).text_config
        cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = \
            64, 96, 2, 1, 32
        cfg.linear_num_value_heads, cfg.linear_num_key_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim = \
            2, 2, 16, 16
        cfg.num_hidden_layers, cfg.vocab_size = 8, 248320
        cfg.layer_types = cfg.layer_types[:8]
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).to(device)
    elif kind == "qwen06":
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device)
    else:
        raise ValueError(kind)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def layers_of(model):
    return model.model.layers


def shape_of(model):
    L = layers_of(model)
    return len(L), L[0].mlp.down_proj.in_features


def dev_of(model):
    return model.get_input_embeddings().weight.device


# ------------------------------------------------------------------------------------------------ hooks
class MLPHooks:
    """Forward pre-hooks on every down_proj. Modes (any combination, set as attributes):
    capture: keep the down_proj input per layer in self.acts (graph kept, so gradients can be taken);
    abl:     {layer: (rows, pos, idx, vals)} -> x[rows, pos, idx] = vals (ablation / patching), applied before capture;
    rec:     callable(layer, x) for corpus statistics."""

    def __init__(self, model):
        self.capture, self.abl, self.rec = False, None, None
        self.acts = {}
        self.handles = [l.mlp.down_proj.register_forward_pre_hook(self._mk(i)) for i, l in enumerate(layers_of(model))]

    def _mk(self, i):
        def hook(mod, args):
            x = args[0]
            changed = False
            if self.abl is not None and i in self.abl:
                rows, pos, idx, vals = self.abl[i]
                x = x.clone()
                x[rows, pos, idx] = vals.to(x.dtype)
                changed = True
            if self.capture:
                self.acts[i] = x
            if self.rec is not None:
                self.rec(i, x)
            return (x,) + tuple(args[1:]) if changed else None
        return hook

    def reset(self):
        self.capture, self.abl, self.rec = False, None, None
        self.acts = {}

    def remove(self):
        for h in self.handles:
            h.remove()


# ------------------------------------------------------------------------------------------------ objectives L
def L_logprob(target):
    return lambda lg: torch.log_softmax(lg.float(), -1)[target]


def L_logitdiff(a, b):
    return lambda lg: lg[a].float() - lg[b].float()


def L_logsumprob(ids):
    ids = list(ids)
    return lambda lg: torch.logsumexp(torch.log_softmax(lg.float(), -1)[ids], 0)


# ------------------------------------------------------------------------------------------------ attribution
@torch.no_grad()
def prefill(model, ids):
    """Run ids under no_grad into a fresh cache; returns (cache, last logits)."""
    from transformers import DynamicCache
    cache = DynamicCache(config=model.config)
    out = model(input_ids=torch.tensor([ids], device=dev_of(model)), past_key_values=cache, use_cache=True)
    return out.past_key_values, out.logits[0, -1]


def _det_attention():
    """The math SDPA backend for the grad chunk (a few query tokens): the fused attention backward kernels use
    atomics and are not bitwise deterministic (pilot: top-20 order differed between two runs)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel([SDPBackend.MATH])


def set_deterministic():
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def readonly_linear_states(cache):
    """The Gated DeltaNet cache layers copy the new conv / recurrent state into the cached tensors in place after a
    forward; the grad chunk reads those tensors (initial_state), so the in-place write breaks the backward. The cache
    is not reused after the grad chunk, so its linear-attention layers return the new states without storing them."""
    for layer in cache.layers:
        if not hasattr(layer, "recurrent_states"):
            continue

        def conv(conv_states, state_idx=0, conv_kernel_size=None, _l=layer, **kw):
            assert _l.has_previous_state[state_idx]
            return torch.cat([_l.conv_states[state_idx], conv_states], dim=-1)

        def rec(recurrent_states, state_idx=0, _l=layer, **kw):
            return recurrent_states
        layer.update_conv_state = conv
        layer.update_recurrent_state = rec


def attribute(model, hooks, ids, positions, objectives):
    """Activation x gradient at each position p in `positions` (sorted, contiguous) with objective objectives[p]
    (a function of the next-token logits at p). The prefix ids[:positions[0]] runs under no_grad into the cache;
    the positions run with grad. Returns ({p: score [n_layers, d_ff] float32}, {p: L value}, {p: logits float cpu}).
    Gradients of L_p are taken w.r.t. the activations at p only (later positions cannot affect L_p)."""
    p0, p1 = positions[0], positions[-1]
    assert positions == list(range(p0, p1 + 1)) and p0 >= 1
    cache, _ = prefill(model, ids[:p0])
    readonly_linear_states(cache)
    emb = model.get_input_embeddings()(torch.tensor([ids[p0:p1 + 1]], device=dev_of(model)))
    emb = emb.detach().requires_grad_(True)
    hooks.acts, hooks.capture = {}, True
    try:
        with torch.enable_grad(), _det_attention():
            out = model(inputs_embeds=emb, past_key_values=cache, use_cache=True)
            nl = len(layers_of(model))
            acts = [hooks.acts[i] for i in range(nl)]
            scores, Ls, lgs = {}, {}, {}
            for j, p in enumerate(positions):
                lg = out.logits[0, j]
                L = objectives[p](lg)
                g = torch.autograd.grad(L, acts, retain_graph=(j < len(positions) - 1))
                scores[p] = torch.stack([(a[0, j].detach().float() * gg[0, j].float()) for a, gg in zip(acts, g)])
                Ls[p] = float(L.detach())
                lgs[p] = lg.detach().float().cpu()
    finally:
        hooks.capture, hooks.acts = False, {}
    return scores, Ls, lgs


def attribute_full(model, hooks, ids, p, objective):
    """Reference: full-sequence forward with grad, gradient of L_p w.r.t. the activations at p. Local test only."""
    emb = model.get_input_embeddings()(torch.tensor([ids[: p + 1]], device=dev_of(model))).detach().requires_grad_(True)
    hooks.acts, hooks.capture = {}, True
    try:
        with torch.enable_grad():
            out = model(inputs_embeds=emb, use_cache=False)
            acts = [hooks.acts[i] for i in range(len(layers_of(model)))]
            L = objective(out.logits[0, p])
            g = torch.autograd.grad(L, acts)
            return torch.stack([(a[0, p].detach().float() * gg[0, p].float()) for a, gg in zip(acts, g)]), float(L)
    finally:
        hooks.capture, hooks.acts = False, {}


def topk_neurons(score, k=20):
    """score [n_layers, d_ff] -> list of (layer, neuron, value) by |value|, descending."""
    d = score.shape[1]
    v, i = score.flatten().abs().topk(k)
    flat = score.flatten()
    return [(int(j) // d, int(j) % d, float(flat[j])) for j in i.tolist()]


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


# ------------------------------------------------------------------------------------------------ ablation / sampling
def ablation_spec(rows_cond, layer_vals):
    """rows_cond: list over batch rows of a condition = list of (layer, neuron, value) to set at the ablated position.
    Returns {layer: (rows, pos, idx, vals)} for MLPHooks.abl with pos 0 (a one-token chunk)."""
    per = {}
    for b, cond in enumerate(rows_cond):
        for (l, n, v) in cond:
            per.setdefault(l, ([], [], []))
            per[l][0].append(b)
            per[l][1].append(n)
            per[l][2].append(v)
    out = {}
    for l, (rows, idx, vals) in per.items():
        out[l] = (torch.tensor(rows), torch.zeros(len(rows), dtype=torch.long), torch.tensor(idx),
                  torch.tensor(vals, dtype=torch.float32))
    return out


def _to_dev(spec, dev):
    return {l: tuple(x.to(dev) for x in v) for l, v in spec.items()}


def expand_cache(cache, n):
    cache.reorder_cache(torch.zeros(n, dtype=torch.long, device=next(iter(_cache_tensors(cache))).device))
    return cache


def _cache_tensors(cache):
    for layer in cache.layers:
        for name in ("keys", "values"):
            t = getattr(layer, name, None)
            if torch.is_tensor(t) and t.numel():
                yield t
        for d in ("conv_states", "recurrent_states"):
            for t in (getattr(layer, d, None) or {}).values():
                if torch.is_tensor(t):
                    yield t


@torch.no_grad()
def ablated_logits(model, hooks, ids, t, conds, chunk=64):
    """Next-token logits at t (float32 cpu [len(conds), V]) with each condition's ablation applied at t only."""
    out = []
    for c0 in range(0, len(conds), chunk):
        cc = conds[c0:c0 + chunk]
        cache, _ = prefill(model, ids[:t])
        expand_cache(cache, len(cc))
        hooks.abl = _to_dev(ablation_spec(cc, None), dev_of(model))
        try:
            x = torch.tensor([[ids[t]]] * len(cc), device=dev_of(model))
            lg = model(input_ids=x, past_key_values=cache, use_cache=True).logits[:, -1].float().cpu()
        finally:
            hooks.abl = None
        out.append(lg)
    return torch.cat(out)


def sample_next(lg, gen, temperature, top_p, top_k):
    """lg [B, V] float; greedy when temperature == 0. Returns [B] long."""
    if temperature == 0:
        return lg.argmax(-1)
    lg = lg / temperature
    if top_k:
        kv, _ = lg.topk(top_k, -1)
        lg = lg.masked_fill(lg < kv[:, -1:], float("-inf"))
    pr = torch.softmax(lg, -1)
    if top_p < 1.0:
        sp, si = pr.sort(-1, descending=True)
        cum = sp.cumsum(-1)
        drop = cum - sp > top_p
        sp = sp.masked_fill(drop, 0.0)
        pr = torch.zeros_like(pr).scatter(-1, si, sp)
        pr = pr / pr.sum(-1, keepdim=True)
    return torch.multinomial(pr, 1, generator=gen).squeeze(-1)


@torch.no_grad()
def ablated_continuations(model, hooks, ids, t, conds, n_new, temperature=0.7, top_p=0.95, top_k=20, seed=0,
                          chunk=48, stop_ids=()):
    """For each row condition: feed ids[:t] (cached), then ids[t] with that row's ablation applied at t only, then
    sample n_new tokens (the first predicted at t) with no further ablation. Returns list of token lists."""
    dev = dev_of(model)
    gen = torch.Generator(device=dev).manual_seed(seed)
    res = []
    for c0 in range(0, len(conds), chunk):
        cc = conds[c0:c0 + chunk]
        cache, _ = prefill(model, ids[:t])
        expand_cache(cache, len(cc))
        hooks.abl = _to_dev(ablation_spec(cc, None), dev)
        try:
            x = torch.tensor([[ids[t]]] * len(cc), device=dev)
            lg = model(input_ids=x, past_key_values=cache, use_cache=True).logits[:, -1].float()
        finally:
            hooks.abl = None
        toks = []
        done = torch.zeros(len(cc), dtype=torch.bool, device=dev)
        for s in range(n_new):
            nxt = sample_next(lg, gen, temperature, top_p, top_k)
            toks.append(nxt)
            if stop_ids:
                done |= torch.isin(nxt, torch.tensor(list(stop_ids), device=dev))
                if bool(done.all()):
                    break
            if s < n_new - 1:
                lg = model(input_ids=nxt[:, None], past_key_values=cache, use_cache=True).logits[:, -1].float()
        T = torch.stack(toks, 1).cpu().tolist()
        res += T
    return res


# ------------------------------------------------------------------------------------------------ statistics
def boot_ci(x, n=5000, seed=0, stat=None):
    """Percentile bootstrap CI of the mean (or `stat`) of a 1-d sequence."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    f = stat or np.mean
    bs = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(f(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_diff(a, b, n=5000, seed=0):
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    bs = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(n)]
    return float(a.mean() - b.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def diff_topk(SA, SB, k=20):
    """SA [nA, N], SB [nB, N] (torch, same device). Top-k flat neuron indices by |mean(SA) - mean(SB)|."""
    d = SA.mean(0) - SB.mean(0)
    return d.abs().topk(k).indices.tolist(), d


def split_half(SA, SB, k=20, n_splits=20, seed=0, shuffle=False):
    """Stability (b): mean Jaccard of the differential top-k computed on two random halves of the episodes (each
    condition split in half). shuffle=True: condition labels permuted over the pooled episodes first (chance level),
    with a fresh permutation per split."""
    g = torch.Generator().manual_seed(seed)
    nA, nB = SA.shape[0], SB.shape[0]
    S = torch.cat([SA, SB])
    out = []
    for _ in range(n_splits):
        if shuffle:
            perm = torch.randperm(nA + nB, generator=g)
            ia, ib = perm[:nA], perm[nA:]
        else:
            ia, ib = torch.arange(nA), torch.arange(nA, nA + nB)
        ia = ia[torch.randperm(len(ia), generator=g)]
        ib = ib[torch.randperm(len(ib), generator=g)]
        ha, hb = len(ia) // 2, len(ib) // 2
        dev = S.device
        t1, _ = diff_topk(S[ia[:ha].to(dev)], S[ib[:hb].to(dev)], k)
        t2, _ = diff_topk(S[ia[ha:2 * ha].to(dev)], S[ib[hb:2 * hb].to(dev)], k)
        out.append(jaccard(t1, t2))
    return sum(out) / len(out), out


def nid(layer, neuron):
    return f"L{layer}N{neuron}"


def parse_nid(s):
    m = re.match(r"L(\d+)N(\d+)$", s)
    return int(m.group(1)), int(m.group(2))


def dump_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def clip_str(s, n):
    return s if len(s) <= n else s[:n] + "..."


def isfinite(x):
    return x is not None and math.isfinite(x)
