"""Per-head attention-output capture and patching (Experiment 18).

The intervention site is the input of each attention block's output projection (`self_attn.o_proj`), whose last
dimension is the concatenation of the heads' outputs (head h occupies [h*d, (h+1)*d), d = head_dim). Replacing one
head's slice at chosen positions with the same head's slice from a donor forward pass swaps exactly that head's
contribution to the residual stream at those positions; everything else is the recipient's own computation.

Positions are offsets from the end of the row (1 = last token), as in `patching.py`, so left-padded batches and
rows of different lengths line up. Donor and recipient must be token-identical at the patched positions and at
every later position (true for the Exp 17 author pairs everywhere, and for the Exp 16 condition pairs inside the
shared suffix), which is what makes an all-heads-all-blocks swap at those positions reproduce the donor's forward
pass there exactly: the residual at a patched position is embedding (identical) + attention outputs (all replaced
by the donor's) + MLP outputs (functions of that residual), by induction over blocks.
"""
from __future__ import annotations

import torch

from patching import _layers, _batches, _pad, _readout


def attn(model, L):
    return _layers(model)[L].self_attn


def head_geometry(model, L):
    """(n_heads, head_dim, n_rep) for block L; n_rep = query heads per key/value head (GQA group size)."""
    a = attn(model, L)
    d = int(a.head_dim)
    H = a.o_proj.in_features // d
    n_rep = int(getattr(a, "num_key_value_groups", 1))
    return H, d, n_rep


@torch.no_grad()
def capture_heads(scorer, rows, blocks, offsets, bs=8):
    """Pre-o_proj attention outputs (all heads) at the given offsets-from-end of each row, for each block in
    `blocks`. Returns {L: [tensor [len(offsets[i]), H*d] (cpu, model dtype) or None]}."""
    model = scorer.model
    caps = {L: [None] * len(rows) for L in blocks}
    for idx in _batches(rows, bs):
        ids, att, T = _pad(scorer, rows, idx)
        handles = []
        for L in blocks:
            def hook(mod, args, L=L, idx=idx, T=T):
                x = args[0]
                for j, i in enumerate(idx):
                    if offsets[i]:
                        pos = torch.tensor([T - o for o in offsets[i]], device=x.device)
                        caps[L][i] = x[j, pos].detach().to("cpu")
            handles.append(attn(model, L).o_proj.register_forward_pre_hook(hook))
        try:
            model(input_ids=ids, attention_mask=att, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
    return caps


@torch.no_grad()
def run_patch_heads(scorer, rows, patches, bs=8):
    """Forward rows, replacing chosen heads' pre-o_proj outputs at chosen positions.
    patches: list of (L, heads, offsets, vecs): heads = list of head indices (or None = all heads) in block L;
    offsets[i] = offsets-from-end for row i; vecs[i] = [len(offsets[i]), H*d] donor tensor (or None = leave row i).
    Returns readouts as patching.run()."""
    model = scorer.model
    reads = [None] * len(rows)
    for idx in _batches(rows, bs):
        ids, att, T = _pad(scorer, rows, idx)
        handles = []
        by_block = {}
        for L, heads, offsets, vecs in patches:
            by_block.setdefault(L, []).append((heads, offsets, vecs))
        for L, items in by_block.items():
            H, d, _ = head_geometry(model, L)

            def hook(mod, args, idx=idx, T=T, items=items, d=d, H=H):
                x = args[0].clone()
                for heads, offsets, vecs in items:
                    hs = range(H) if heads is None else heads
                    for j, i in enumerate(idx):
                        v = vecs[i]
                        if v is None or not offsets[i]:
                            continue
                        pos = torch.tensor([T - o for o in offsets[i]], device=x.device)
                        v = v.to(x.device, x.dtype)
                        if heads is None:
                            x[j, pos] = v
                        else:
                            for h in hs:
                                x[j, pos, h * d:(h + 1) * d] = v[:, h * d:(h + 1) * d]
                return (x,) + tuple(args[1:])
            handles.append(attn(model, L).o_proj.register_forward_pre_hook(hook))
        try:
            logits = model(input_ids=ids, attention_mask=att, logits_to_keep=1).logits[:, -1]
        finally:
            for h in handles:
                h.remove()
        for j, r in zip(idx, _readout(scorer, logits)):
            reads[j] = r
    return reads


@torch.no_grad()
def capture_heads_patched(scorer, rows, blocks, offsets, resid_patch, bs=8):
    """Experiment 19: run rows with a residual-stream patch (resid_patch = {L: (patch_offsets, vecs)}, replacing the
    output of decoder block L at patch_offsets[i] by vecs[i] as patching.run_patch_at) and, in the same pass, capture
    the pre-o_proj attention outputs (all heads) at `offsets` for each block in `blocks`. Returns (caps, reads):
    caps as capture_heads, reads as patching.run() (the label-patched grades)."""
    from patching import _module
    model = scorer.model
    caps = {L: [None] * len(rows) for L in blocks}
    reads = [None] * len(rows)
    for idx in _batches(rows, bs):
        ids, att, T = _pad(scorer, rows, idx)
        handles = []
        for Lp, (poffs, vecs) in resid_patch.items():
            def phook(mod, inp, out, idx=idx, T=T, poffs=poffs, vecs=vecs):
                h = out[0] if isinstance(out, tuple) else out
                for j, i in enumerate(idx):
                    v = vecs[i]
                    if v is None or not poffs[i]:
                        continue
                    pos = torch.tensor([T - o for o in poffs[i]], device=h.device)
                    h[j, pos] = v.to(h.device, h.dtype)
                return out
            handles.append(_module(model, Lp).register_forward_hook(phook))
        for L in blocks:
            def hook(mod, args, L=L, idx=idx, T=T):
                x = args[0]
                for j, i in enumerate(idx):
                    if offsets[i]:
                        pos = torch.tensor([T - o for o in offsets[i]], device=x.device)
                        caps[L][i] = x[j, pos].detach().to("cpu")
            handles.append(attn(model, L).o_proj.register_forward_pre_hook(hook))
        try:
            logits = model(input_ids=ids, attention_mask=att, logits_to_keep=1).logits[:, -1]
        finally:
            for h in handles:
                h.remove()
        for j, r in zip(idx, _readout(scorer, logits)):
            reads[j] = r
    return caps, reads
