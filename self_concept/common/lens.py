"""Shared J-lens reader for self_concept (built for SC-CAL; for N1's internals, N3 and G1). Model-agnostic version of
jlens/pod_jlens.py's Reader, which re-implements the reference readout (anthropics/jacobian-lens @581d398,
jlens/lens.py + hf.py): for a block output h at source layer l,

    lens logits = lm_head(final_norm(J_l h)), then the model's final logit softcap if it has one (Gemma 4: 30)

The logit lens is J = I (name "logit"). Lens files are the official {"J": {layer: [d, d]}, "n_prompts",
"source_layers", "d_model"} dict, pinned in LENSES. Layer l = the output of decoder block l, 0-based.

Word sets: a word counts through its single-token forms (" w", "w", " W", "W"; multi-token forms are dropped).
A set's score at one read and layer is its log-mass: logsumexp over its token ids of the lens log-softmax. Random
floors: sets of the same size drawn from alphabetic single tokens with a leading space (seeded). The deep band of a
lens = the last quarter (rounded up) of its source layers, without the final decoder layer.
"""
from __future__ import annotations

import math
import random
import re

import torch

MODELS = {   # key -> HF id, revision, decoder layers, chat-template kwargs (thinking off)
    "qwen27": ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9", 64, {"enable_thinking": False}),
    "gemma31": ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475", 60, {}),
    "qwen9": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", 32, {"enable_thinking": False}),
}
NP, SK, CB = ("neuronpedia/jacobian-lens", "e166deb5ad540932100ef803c8c9bcc2a5453694"), \
    ("solarkyle/jspace-lenses", "f5767d1bb4cb25ed65a8ed16eb47a6e6a1f0a0bc"), \
    ("camilablank/workspace-lenses", "d740106d1e0f95456dc8718fba2895e9c8ffd6ef")
LENSES = {   # key -> [(lens name, repo, file, revision)]; the first is the primary
    "qwen27": [("np", NP[0], "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt", NP[1])],
    "gemma31": [("sk", SK[0], "gemma-4-31b-it/lens.pt", SK[1]),     # fit on -it (100 prompts)
                ("np", NP[0], "gemma-4-31b/jlens/Salesforce-wikitext/gemma-4-31B_jacobian_lens.pt", NP[1])],  # base fit
    "qwen9": [("cb", CB[0], "qwen3.5-9b/j-lens/lens.pt", CB[1]), ("cbr", CB[0], "qwen3.5-9b/r-lens/lens.pt", CB[1])],
}


def deep_band(source_layers, n_layers):
    src = sorted(int(x) for x in source_layers if int(x) < n_layers - 1)
    return src[-math.ceil(len(src) / 4):]


def load_model(key, tiny=False, path=None):
    """(tok, model, layers, norm, lm_head, softcap). path: a local model dir of the same architecture (a merged
    fine-tune) instead of the pinned HF id. tiny=True: the real config shrunk to 4 layers, random weights, float32 on
    CPU (tests only; needs the config in the local HF cache)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    mid, rev, n_layers, _ = MODELS[key]
    if path is not None and not tiny:
        mid, rev = str(path), None
    tok = AutoTokenizer.from_pretrained(mid, revision=None if tiny else rev)
    if tiny:
        cfg = AutoConfig.from_pretrained(mid)
        t = getattr(cfg, "text_config", cfg)
        small = dict(num_hidden_layers=4, hidden_size=64, intermediate_size=128, num_attention_heads=2,
                     num_key_value_heads=1, head_dim=32)
        if hasattr(t, "layer_types") and t.layer_types:
            kinds = list(dict.fromkeys(t.layer_types))
            small["layer_types"] = (kinds * 4)[:4]
        for k in ("linear_num_value_heads", "linear_num_key_heads", "linear_key_head_dim", "linear_value_head_dim",
                  "global_head_dim", "num_global_key_value_heads"):
            if hasattr(t, k):
                small[k] = 2 if "heads" in k else 16
        for k, v in small.items():
            setattr(t, k, v)
        for k in ("mtp_num_hidden_layers", "num_kv_shared_layers"):
            if hasattr(t, k):
                setattr(t, k, 0)
        v = getattr(cfg, "vision_config", None)
        if v is not None:
            for k, val in dict(hidden_size=32, depth=1, num_hidden_layers=1, num_heads=2, num_attention_heads=2,
                               num_key_value_heads=2, head_dim=16, intermediate_size=64, out_hidden_size=64).items():
                if hasattr(v, k):
                    setattr(v, k, val)
        torch.manual_seed(0)
        try:
            model = AutoModelForImageTextToText.from_config(cfg, dtype=torch.float32)
        except ValueError:
            model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
        n_layers = 4
    else:
        try:
            model = AutoModelForImageTextToText.from_pretrained(mid, revision=rev, dtype=torch.bfloat16,
                                                                device_map="cuda")
        except ValueError:
            model = AutoModelForCausalLM.from_pretrained(mid, revision=rev, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    text = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n_layers and name.endswith("layers") \
                and "vision" not in name and "audio" not in name:
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            if hasattr(parent, "norm") and (text is None or "language_model" in name):
                text = parent
    assert text is not None, "text decoder not found"
    cfg = getattr(model.config, "text_config", model.config)
    return tok, model, text.layers, text.norm, model.get_output_embeddings(), getattr(cfg, "final_logit_softcapping", None)


class Reader:
    """Block outputs at one position, then lens log-softmax rows. Lenses live on the model's device in fp16."""

    def __init__(self, tok, model, layers, norm, lm_head, softcap=None):
        self.tok, self.model, self.blocks, self.norm, self.lm_head, self.softcap = \
            tok, model, layers, norm, lm_head, softcap
        self.dev = lm_head.weight.device
        self.lenses = {}

    def add_lens(self, name, ck):
        """ck: a path or the loaded dict. Returns its metadata (n_prompts, source_layers, d_model)."""
        if isinstance(ck, str):
            ck = torch.load(ck, map_location="cpu", weights_only=True)
        if "J" not in ck:
            raise ValueError(f"not a JacobianLens file (keys {sorted(ck)})")
        self.lenses[name] = {int(k): v.to(self.dev, torch.float16) for k, v in ck["J"].items()}
        src = ck.get("source_layers") or sorted(self.lenses[name])
        return {"n_prompts": ck.get("n_prompts"), "source_layers": [int(x) for x in src], "d_model": ck.get("d_model")}

    def drop_lens(self, name):
        self.lenses.pop(name, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward(self, ids, record, pos=-1, add=None, edit=None, logits=False):
        """{layer: [d] fp32 cpu} block outputs at token `pos` for layers in `record` ({layer: [len(pos), d]} when pos
        is a list). add = [(layer, pos, vec)]: vec (fp32 [d]) is added to that block's output at that position before
        later blocks see it. edit = {layer: fn(t)}: fn edits that block's output tensor [1, n, d] in place (after any
        add). logits=True also returns the next-token logits at the last position: (acts, [V] fp32 cpu)."""
        acts, handles = {}, []
        n = len(ids)
        sel = [p % n for p in pos] if isinstance(pos, (list, tuple)) else pos % n

        def rec(i):
            def h(mod, inp, out):
                t = out if torch.is_tensor(out) else out[0]
                acts[i] = t[0, sel].float().cpu()
            return h

        def ed(fn):
            def h(mod, inp, out):
                fn(out if torch.is_tensor(out) else out[0])
                return out
            return h

        def inj(vec, p):
            def h(mod, inp, out):
                t = out if torch.is_tensor(out) else out[0]
                t[0, p % n] += vec.to(t.device, t.dtype)
                return out
            return h
        for layer, p, vec in add or ():
            handles.append(self.blocks[layer].register_forward_hook(inj(vec, p)))
        for layer, fn in (edit or {}).items():
            handles.append(self.blocks[layer].register_forward_hook(ed(fn)))
        for i in record:
            handles.append(self.blocks[i].register_forward_hook(rec(i)))
        try:
            with torch.no_grad():
                out = self.model(input_ids=torch.tensor([ids], device=self.dev), use_cache=False, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        if logits:   # the model's own logits (its forward already applies any final softcap)
            return acts, out.logits[0, -1].float().cpu()
        return acts

    def logprobs(self, resid, layer, lens):
        """resid [n, d] fp32 -> lens log-softmax [n, V] fp32 on the device."""
        h = resid.to(self.dev)
        if lens != "logit":
            h = h @ self.lenses[lens][layer].float().T
        z = self.lm_head(self.norm(h.to(self.lm_head.weight.dtype))).float()
        if self.softcap:
            z = self.softcap * torch.tanh(z / self.softcap)
        return torch.log_softmax(z, -1)

    def lens_vectors(self, layer, token_ids, lens):
        """[len(token_ids), d] fp32: the input-space direction each token is read along, J_l^T (g * W_U[t]), with g
        the final norm's gain (as jlens/pod_jlens.py)."""
        d = self.lm_head.weight.shape[1]
        g = self.norm(torch.ones(1, d, device=self.dev, dtype=self.lm_head.weight.dtype)).float()[0]
        wu = self.lm_head.weight[token_ids].float() * g
        return wu if lens == "logit" else wu @ self.lenses[lens][layer].float()

    def norms(self, ids, layer, pos=-1):
        return float(self.forward(ids, [layer], pos)[layer].norm())


def chat_ids(tok, key, messages):
    """Prompt token ids with the generation prompt (thinking off). The decision token is the last one."""
    out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **MODELS[key][3])
    if hasattr(out, "keys"):
        out = out["input_ids"]
    return list(out[0] if out and isinstance(out[0], list) else out)


def word_ids(tok, words):
    out = []
    for w in words:
        for c in (" " + w, w, " " + w.capitalize(), w.capitalize()):
            ids = tok.encode(c, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in out:
                out.append(ids[0])
    return out


_WORD = re.compile(r"^ [A-Za-z]{3,}$")


def word_pool(tok):
    """Alphabetic single tokens with a leading space (the random-floor pool), in id order."""
    pool = []
    for i in range(len(tok)):
        s = tok.decode([i])
        if _WORD.match(s):
            pool.append(i)
    return pool


def random_sets(pool, size, n, seed):
    rng = random.Random(seed)
    return [rng.sample(pool, size) for _ in range(n)]


def logmass(logp, ids):
    """logp [n, V], ids list -> [n] log of the summed probability of ids."""
    return torch.logsumexp(logp[:, ids], -1)
