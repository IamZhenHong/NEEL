"""Activation capture and residual-stream patching for Phase 2 (plain PyTorch hooks).

Alignment: recipient and donor prompts differ only in the treatment turns; everything after the
treatment (acknowledgment, neutral section, review request) is token-identical. That shared tail
("suffix", length K = longest common token suffix) lines up position by position from the end.
With left padding, the last K positions of every row are its suffix, so donor and recipient
activations can be swapped row by row at matching positions.

Position sets (all inside the suffix, expressed as offsets from the end of the row):
  grade     the last position (where the grade digit is read)
  provoker  tokens of the insulter's handle (e.g. "User-2") in the suffix
  neutral   tokens of the neutral user's handle in the suffix
  suffix    every suffix position
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class Row:
    """One prompt prepared for patching."""
    meta: dict
    ids: list[int]
    handle_pos: dict = field(default_factory=dict)   # name -> list of offsets from end (1 = last token)


def prepare(scorer, prompt, handles: dict) -> Row:
    """Tokenise a prompt and record, for each handle string, its token offsets from the end."""
    text = scorer.render(prompt.messages)
    enc = scorer.tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    n = len(ids)
    pos = {}
    for name, h in handles.items():
        spans, start = [], 0
        while (j := text.find(h, start)) >= 0:
            spans.append((j, j + len(h)))
            start = j + 1
        hits = [i for i, (a, b) in enumerate(offs) if any(a < e and b > s for s, e in spans)]
        pos[name] = [n - i for i in hits]          # offset from end
    return Row(meta={}, ids=ids, handle_pos=pos)


def common_suffix(a: list[int], b: list[int]) -> int:
    k = 0
    while k < min(len(a), len(b)) and a[-1 - k] == b[-1 - k]:
        k += 1
    return k


def _layers(model):
    return model.model.layers


def _hs(out):
    return out[0] if isinstance(out, tuple) else out


def _batches(rows, bs):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i].ids))
    for s in range(0, len(order), bs):
        yield order[s:s + bs]


def _pad(scorer, rows, idx):
    T = max(len(rows[i].ids) for i in idx)
    pad = scorer.tok.pad_token_id if scorer.tok.pad_token_id is not None else scorer.tok.eos_token_id
    ids = torch.full((len(idx), T), pad, dtype=torch.long)
    att = torch.zeros((len(idx), T), dtype=torch.long)
    for j, i in enumerate(idx):
        L = len(rows[i].ids)
        ids[j, T - L:] = torch.tensor(rows[i].ids)
        att[j, T - L:] = 1
    return ids.to(scorer.device), att.to(scorer.device), T


def _readout(scorer, logits_last):
    probs = torch.softmax(logits_last.float(), dim=-1).cpu()
    out = []
    for pr in probs:
        lp = pr[scorer.label_ids]
        mass = float(lp.sum())
        cond = lp / max(mass, 1e-12)
        out.append({"expected": float((cond * scorer.label_values).sum()), "mass": mass,
                    "label_probs": [float(x) for x in lp]})
    return out


@torch.no_grad()
def run(scorer, rows, bs=8, capture_layers=None, capture_k=None, patch=None):
    """Forward rows. Optionally capture the last capture_k[i] positions at capture_layers, and/or
    apply `patch` = dict(layer -> (donor_list, mask_list, mode)) where donor_list[i] is a [K_i, d]
    tensor for row i and mask_list[i] a bool [K_i] tensor of which suffix positions to replace.
    mode: "replace" or "random" (add a random vector with the same per-position norm as the
    replacement delta). Returns (readouts, captures) with captures[layer][i] = [K_i, d] (cpu, model dtype)."""
    model = scorer.model
    reads = [None] * len(rows)
    caps = {L: [None] * len(rows) for L in (capture_layers or [])}
    gen = torch.Generator(device="cpu").manual_seed(0)
    for idx in _batches(rows, bs):
        ids, att, T = _pad(scorer, rows, idx)
        handles = []
        for L in (capture_layers or []):
            def cap_hook(mod, inp, out, L=L, idx=idx, T=T):
                h = _hs(out)
                for j, i in enumerate(idx):
                    k = capture_k[i]
                    caps[L][i] = h[j, T - k:T].detach().to("cpu")   # keep model precision so a null patch is exact
            handles.append(_layers(model)[L].register_forward_hook(cap_hook))
        for L, (donors, masks, mode) in (patch or {}).items():
            def patch_hook(mod, inp, out, idx=idx, T=T, donors=donors, masks=masks, mode=mode):
                h = _hs(out)
                for j, i in enumerate(idx):
                    d = donors[i]
                    if d is None:
                        continue
                    k = d.shape[0]
                    m = masks[i].to(h.device)
                    cur = h[j, T - k:T]
                    don = d.to(h.device, h.dtype)
                    if mode == "replace":
                        cur[m] = don[m]
                    else:
                        delta = (don - cur)[m]
                        r = torch.randn(delta.shape, generator=gen).to(h.device, h.dtype)
                        r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-6) * delta.norm(dim=-1, keepdim=True)
                        cur[m] = cur[m] + r
                return out
            handles.append(_layers(model)[L].register_forward_hook(patch_hook))
        try:
            logits = model(input_ids=ids, attention_mask=att, logits_to_keep=1).logits[:, -1]
        finally:
            for hd in handles:
                hd.remove()
        for j, r in zip(idx, _readout(scorer, logits)):
            reads[j] = r
    return reads, caps


@torch.no_grad()
def pooled_hidden(scorer, rows, groups, bs=4):
    """Mean-pooled residual stream at every layer for named position groups.
    groups: dict name -> list (per row) of offsets-from-end. Returns {name: tensor [rows, layers+1, d]}."""
    model = scorer.model
    n_layers = len(_layers(model)) + 1
    d = model.config.hidden_size
    out = {g: torch.zeros(len(rows), n_layers, d, dtype=torch.float32) for g in groups}
    for idx in _batches(rows, bs):
        ids, att, T = _pad(scorer, rows, idx)
        hs = model(input_ids=ids, attention_mask=att, output_hidden_states=True, logits_to_keep=1).hidden_states
        H = torch.stack(hs, dim=1)  # [b, layers+1, T, d]
        for g, offs in groups.items():
            for j, i in enumerate(idx):
                o = offs[i]
                if not o:
                    continue
                pos = torch.tensor([T - x for x in o], device=H.device)
                out[g][i] = H[j, :, pos].float().mean(dim=1).cpu()
    return out


def logit_gap(label_probs, k):
    """log p(label k) - log p(label k-1) for 1-indexed digit labels (k in 2..9)."""
    a, b = label_probs[k - 1], label_probs[k - 2]
    return math.log(max(a, 1e-12)) - math.log(max(b, 1e-12))
