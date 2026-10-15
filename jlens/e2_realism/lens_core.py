"""Shared J-lens machinery for jl2 and jl3 (Qwen3.6-27B). Copied from jlens/pod_jlens.py (2026-10-15, unchanged in
substance) so the queue-v3 cards do not depend on that file; additions: lens loading with sha256, a single-position
all-layer read, batched greedy generation without padding (prompts grouped by exact length).

Lenses (neuronpedia/jacobian-lens): 'jlens' = the Jacobian lens fitted by M. Piotrowski (n=1000 WikiText prompts);
'jpp' = the J++ lens (Ayonrinde & Lindsey); 'logit' = the logit lens (J = I). Readout at layer l for a block output h:
lm_head(final_norm(J_l h)), as in anthropics/jacobian-lens (jlens/lens.py + hf.py)."""
import hashlib
import time
from collections import defaultdict

import numpy as np
import torch
from huggingface_hub import hf_hub_download

MODEL = "Qwen/Qwen3.6-27B"
REPO = "neuronpedia/jacobian-lens"
LENS_FILES = {"jlens": "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
              "jpp": "qwen3.6-27b/jpp/Salesforce-wikitext/Qwen3.6-27B_jpp_lens.pt"}
LENSES = ["jlens", "jpp", "logit"]
BAND = list(range(20, 61, 4))
FINAL = 63
SPIDER = "Fact: The number of legs on the animal that spins webs is "
T0 = time.time()


def log(*a):
    print(f"[jl23 {time.time() - T0:7.0f}s]", *a, flush=True)


def load_model(tiny=False):
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tiny:   # CPU test only: the real config shrunk, random weights
        cfg = AutoConfig.from_pretrained(MODEL)
        t = cfg.text_config
        t.hidden_size, t.intermediate_size, t.num_attention_heads, t.num_key_value_heads, t.head_dim = 64, 128, 2, 1, 32
        t.linear_num_value_heads, t.linear_num_key_heads, t.linear_key_head_dim, t.linear_value_head_dim = 2, 2, 16, 16
        v = cfg.vision_config
        v.hidden_size, v.depth, v.num_heads, v.intermediate_size, v.out_hidden_size = 32, 1, 2, 64, 64
        torch.manual_seed(0)
        model = AutoModelForImageTextToText.from_config(cfg, dtype=torch.bfloat16).eval()
    else:
        model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    text = None
    for path in ("model.language_model", "model", "language_model"):
        try:
            obj = model
            for a in path.split("."):
                obj = getattr(obj, a)
        except AttributeError:
            continue
        if all(hasattr(obj, a) for a in ("layers", "norm", "embed_tokens")):
            text = obj
            break
    assert text is not None and len(text.layers) == 64, "text decoder not found"
    return tok, model, text


class Reader:
    def __init__(self, tok, model, text):
        self.tok, self.model, self.text = tok, model, text
        self.blocks, self.norm, self.lm_head = text.layers, text.norm, model.lm_head
        self.lenses = {}
        self.dev = self.lm_head.weight.device

    def add_lens(self, name, path):
        ck = torch.load(path, map_location="cpu", weights_only=True) if isinstance(path, str) else path
        log(f"lens {name}: keys {sorted(ck)}; layers {min(ck['J'])}..{max(ck['J'])}; n_prompts {ck.get('n_prompts')}")
        self.lenses[name] = {int(k): v.to(self.dev, torch.float16) for k, v in ck["J"].items()}
        return {k: (str(v)[:500] if k != "J" else f"{len(v)} layers") for k, v in ck.items()}

    def lens_layers(self):
        return sorted(self.lenses["jlens"])

    def forward(self, ids, record, positions=None):
        """Block outputs at `record` layers, at `positions` (default all), bf16 on CPU: {layer: [n_pos, d]}."""
        acts, handles = {}, []

        def mk(i):
            def h(mod, inp, out):
                t = out if torch.is_tensor(out) else out[0]
                acts[i] = (t[0] if positions is None else t[0, positions]).to(torch.bfloat16).cpu()
            return h
        for i in record:
            handles.append(self.blocks[i].register_forward_hook(mk(i)))
        try:
            with torch.no_grad():
                self.model(input_ids=torch.tensor([ids], device=self.dev), use_cache=False, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        return acts

    def logits(self, resid, layer, lens):
        """resid [n, d] -> lens logits [n, V] fp32 on GPU. No J for the final block: there the model's own output."""
        h = resid.to(self.dev).float()
        if lens != "logit" and layer != FINAL:
            h = h @ self.lenses[lens][layer].float().T
        return self.lm_head(self.norm(h.to(self.lm_head.weight.dtype))).float()

    def greedy(self, prompts, max_new, eos_ids, batch=48):
        """Greedy continuations (token id lists) for a list of token-id prompts. No padding: prompts are grouped by
        exact length, so every batch is rectangular and each row is computed as it would be alone."""
        groups = defaultdict(list)
        for i, p in enumerate(prompts):
            groups[len(p)].append(i)
        out = [None] * len(prompts)
        done = 0
        for L, idx in sorted(groups.items()):
            for c0 in range(0, len(idx), batch):
                sub = idx[c0:c0 + batch]
                x = torch.tensor([prompts[i] for i in sub], device=self.dev)
                with torch.no_grad():
                    g = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new,
                                            do_sample=False, eos_token_id=eos_ids,
                                            pad_token_id=self.tok.pad_token_id or eos_ids[0])
                for i, row in zip(sub, g[:, L:].tolist()):
                    while row and row[-1] in (self.tok.pad_token_id,):
                        row.pop()
                    out[i] = row
                done += len(sub)
            log(f"greedy {done}/{len(prompts)}")
        return out


def best_rank(logits, ids):
    """1-based rank of the best-ranked of `ids` in each row: [n]."""
    return (logits > logits[:, ids].max(1, keepdim=True).values).sum(-1) + 1


def single_forms(tok, w):
    out = []
    for c in (" " + w, w, " " + w.capitalize(), w.capitalize()):
        ids = tok.encode(c, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in out:
            out.append(ids[0])
    return out


def load_lenses(R, tiny, info):
    if tiny:
        d = R.text.config.hidden_size
        for name in LENS_FILES:
            R.add_lens(name, {"J": {L: torch.eye(d) + 0.1 * torch.randn(d, d) for L in range(63)}, "n_prompts": 0})
        return
    info["lenses"] = {}
    for name, fn in LENS_FILES.items():
        p = hf_hub_download(REPO, fn)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        info["lenses"][name] = {"file": fn, "sha256": h.hexdigest(), "meta": R.add_lens(name, p)}


def selfcheck(R):
    ids = R.tok(SPIDER, add_special_tokens=False)["input_ids"]
    acts = R.forward(ids, [FINAL], positions=[len(ids) - 1])
    mine = R.logits(acts[FINAL], FINAL, "logit")[0]
    with torch.no_grad():
        ref = R.model(input_ids=torch.tensor([ids], device=R.dev)).logits[0, -1].float()
    diff = (mine - ref).abs().max().item()
    log(f"selfcheck: hook+unembed vs model logits max |diff| {diff:.4f}")
    assert int(mine.argmax()) == int(ref.argmax()) and diff < 0.5, "hooked readout does not match the model"
    return diff


def spider_check(R):
    """jlens G1 repeated as a setup check: ' spider' rank in the J-lens top 10 at a band layer, model says '8'."""
    tok = R.tok
    ids = tok(SPIDER, add_special_tokens=False)["input_ids"]
    acts = R.forward(ids, BAND + [FINAL], positions=[len(ids) - 1])
    top1 = tok.decode([int(R.logits(acts[FINAL], FINAL, "logit")[0].argmax())])
    sp = [tok.encode(" spider", add_special_tokens=False)[0], tok.encode(" spiders", add_special_tokens=False)[0]]
    ranks = {L: int(best_rank(R.logits(acts[L], L, "jlens"), sp)[0]) for L in BAND}
    res = {"model_top1": top1, "jlens_rank_spider": ranks, "pass": min(ranks.values()) <= 10 and top1.strip() == "8"}
    log("spider check", res)
    return res


def np_save(path, **arrays):
    np.savez_compressed(path, **arrays)
