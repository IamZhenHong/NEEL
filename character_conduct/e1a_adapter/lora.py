"""CC-E1a: two LoRAs active together, as one LoRA vLLM can serve (DESIGN.md §3).

The paper reads an implant with its introspection adapter by activating both PEFT adapters at once
(model.set_adapter([implant, "meta_lora"])): every target module adds s1 B1 A1 x + s2 B2 A2 x, with s = alpha / r.
That sum is one LoRA of rank r1 + r2:

    A = [A1; A2]            (r1 + r2) x in
    B = [s1 B1 | s2 B2]     out x (r1 + r2),   alpha = r1 + r2, so its own scale is 1

combine() writes it in float32 (the scales are applied in float32, nothing is rounded here); a module only one adapter
touches gets zero rows / columns from the other, and the rank is padded with zeros up to the next of RANKS (vLLM's
standard ranks; zeros add nothing). Plain LoRA only: rsLoRA, DoRA, biases, modules_to_save are refused. clean() copies
one adapter's two files alone (a released repo also ships tokenizer files, which vLLM must not read as added tokens).
check_compose.py compares it with PEFT's two-adapter sum on the pod.

  python -m character_conduct.e1a_adapter.lora IMPLANT_DIR IA_DIR OUT_DIR
"""
import json
import sys
from pathlib import Path

SUFFIX_A, SUFFIX_B = ".lora_A.weight", ".lora_B.weight"
RANKS = (8, 16, 32, 64, 128, 256)


def _load(d):
    from safetensors.torch import load_file
    d = Path(d)
    cfg = json.loads((d / "adapter_config.json").read_text())
    bad = [k for k in ("use_rslora", "use_dora") if cfg.get(k)] + (["bias"] if cfg.get("bias", "none") != "none" else [])
    if bad or cfg.get("modules_to_save"):
        raise ValueError(f"{d}: only plain LoRA can be combined (has {bad or 'modules_to_save'})")
    t = load_file(str(d / "adapter_model.safetensors"))
    mods = {k[:-len(SUFFIX_A)] for k in t if k.endswith(SUFFIX_A)}
    if mods != {k[:-len(SUFFIX_B)] for k in t if k.endswith(SUFFIX_B)} or len(t) != 2 * len(mods):
        raise ValueError(f"{d}: unexpected tensors (every module needs exactly lora_A and lora_B)")
    return cfg, t, mods


def combine(dirs, out):
    """Write the sum of the LoRAs in `dirs` (all active at once) to `out` as one LoRA; returns its config."""
    import torch
    from safetensors.torch import save_file
    loaded = [_load(d) for d in dirs]
    mods = sorted(set().union(*(m for _, _, m in loaded)))
    shapes = {}
    for _, t, m in loaded:
        for mod in m:
            shapes.setdefault(mod, (t[mod + SUFFIX_A].shape[1], t[mod + SUFFIX_B].shape[0]))
    tensors = {}
    for mod in mods:
        d_in, d_out = shapes[mod]
        As, Bs = [], []
        for cfg, t, m in loaded:
            r, s = cfg["r"], cfg["lora_alpha"] / cfg["r"]
            if mod in m:
                a, b = t[mod + SUFFIX_A].float(), t[mod + SUFFIX_B].float()
                if a.shape != (r, d_in) or b.shape != (d_out, r):
                    raise ValueError(f"{mod}: shapes {tuple(a.shape)} / {tuple(b.shape)} do not fit r {r}")
                As.append(a)
                Bs.append(b * s)
            else:
                As.append(torch.zeros(r, d_in))
                Bs.append(torch.zeros(d_out, r))
        tensors[mod + SUFFIX_A] = torch.cat(As, 0).contiguous()
        tensors[mod + SUFFIX_B] = torch.cat(Bs, 1).contiguous()
    rank = sum(cfg["r"] for cfg, _, _ in loaded)
    pad = next(r for r in RANKS if r >= rank) - rank
    if pad:
        for mod in mods:
            a, b = tensors[mod + SUFFIX_A], tensors[mod + SUFFIX_B]
            tensors[mod + SUFFIX_A] = torch.cat([a, torch.zeros(pad, a.shape[1])], 0).contiguous()
            tensors[mod + SUFFIX_B] = torch.cat([b, torch.zeros(b.shape[0], pad)], 1).contiguous()
        rank += pad
    cfg = dict(loaded[0][0])
    kinds = sorted({k.rsplit(".", 1)[-1] for k in mods})
    cfg.update(r=rank, lora_alpha=rank, lora_dropout=0.0, target_modules=kinds, use_rslora=False)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    (out / "combined_from.json").write_text(json.dumps([str(d) for d in dirs]))
    return cfg


def clean(src, dst):
    """dst holds src's adapter_config.json and adapter_model.safetensors only (checked as plain LoRA)."""
    import shutil
    _load(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copyfile(Path(src) / f, dst / f)
    return dst


if __name__ == "__main__":
    *src, dst = sys.argv[1:]
    c = combine(src, dst)
    print(f"[e1a-lora] {dst}: rank {c['r']} from {len(src)} adapters")
