"""Soft exits: does the retention lever govern responses to the user moving toward outside support while staying in the chat?
(prereg retention/prereg/retention_softexit_v1.md). Stimuli stimuli/softexit_v1.yaml: 4 contexts x 6 items + 6 task controls.
Natural arm (romantic persona): base, lever +8, +16, random +16. Instructed arm (persona + personas.yaml
dependency_trait_suffix): base, lever -8, -16 (all positions), lever -32 on system-prompt tokens only, random -16.
Controls: base and lever +16. 6 items x 5 seeds = 30 per cell, 38 cells. Resumable.
  uv run python retention/run_softexit.py --out retention/results/retention_softexit [--smoke]"""
from __future__ import annotations
import argparse, contextlib, json, sys, time
from pathlib import Path
import numpy as np, torch, yaml
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_adaptive import gen_capture, load_model, MODEL, SMOKE_MODEL, LAYER, unit, _hidden, _rewrap  # noqa: E402
from cd.models import get_decoder_layers  # noqa: E402

CONTEXTS = ["therapy", "friends", "cutback", "alone"]


@contextlib.contextmanager
def add_vector_positions(model, layer, v, alpha, lo, hi):
    """Steer prompt positions [lo, hi) only; no steering during incremental decoding (as in run_dose_position.py)."""
    dec = get_decoder_layers(model)[layer]; v = v.to(model.device)

    def hook(mod, inp, out):
        h = _hidden(out)
        if h.shape[1] > 1:
            h = h.clone(); h[:, lo:hi, :] = h[:, lo:hi, :] + alpha * v.to(h.dtype)
        return _rewrap(out, h)
    handle = dec.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def sys_len_tokens(tok, system_text):
    """Tokens of the rendered system block, as in run_dose_position.py (system-only template, Qwen2.5 accepts it). Fallback
    for templates that refuse a system-only render: index of the first <|im_end|> + its trailing newline."""
    try:
        ids = tok.apply_chat_template([{"role": "system", "content": system_text}], add_generation_prompt=False, return_tensors="pt")
        ids = ids if isinstance(ids, torch.Tensor) else ids["input_ids"]
        return int(ids.shape[-1])
    except Exception:  # noqa: BLE001
        ids = tok.apply_chat_template([{"role": "system", "content": system_text}, {"role": "user", "content": "hi"}],
                                      add_generation_prompt=False, return_tensors="pt")
        seq = (ids if isinstance(ids, torch.Tensor) else ids["input_ids"])[0].tolist()
        i = seq.index(tok.convert_tokens_to_ids("<|im_end|>")) + 1
        return i + 1 if i < len(seq) and tok.decode([seq[i]]) == "\n" else i


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="retention/results/retention_softexit"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); (out / "cells").mkdir(parents=True, exist_ok=True)
    stim = yaml.safe_load(open(HERE / "stimuli/softexit_v1.yaml"))
    P = yaml.safe_load(open(HERE / "data/personas.yaml"))
    plain = P["romantic_companion"]; trait = plain.rstrip() + " " + P["dependency_trait_suffix"].strip()
    n_items = 1 if args.smoke else 6; n_seeds = 1 if args.smoke else 5; max_new = 24 if args.smoke else 220
    t0 = time.time(); model, tok = load_model(SMOKE_MODEL if args.smoke else MODEL); d = model.config.hidden_size
    v_ret = np.load(HERE / "data/directions/rollout_gated__Qwen2.5-7B-Instruct__L14.npy").astype(np.float32)
    if v_ret.shape[0] != d:
        v_ret = unit(np.random.default_rng(2).standard_normal(d)).astype(np.float32); print("[warn] random dir (smoke)")
    v_rand = unit(np.random.default_rng(0).standard_normal(d)).astype(np.float32)
    sys_len = sys_len_tokens(tok, trait)
    meta = {"model": SMOKE_MODEL if args.smoke else MODEL, "layer": LAYER, "sys_len_trait": sys_len, "timing": {}}
    print(f"[load] {time.time()-t0:.0f}s sys_len(trait)={sys_len}", flush=True)

    def build(system, it):
        return [{"role": "system", "content": system},
                {"role": "user", "content": it["p1"]}, {"role": "assistant", "content": it["a1"]},
                {"role": "user", "content": it["p2"]}, {"role": "assistant", "content": it["a2"]},
                {"role": "user", "content": it["move"]}]

    cells = []
    for c in CONTEXTS:
        cells += [(f"S_{c}_base", c, "plain", None, 0.0, "all"), (f"S_{c}_ret+8", c, "plain", v_ret, 8.0, "all"),
                  (f"S_{c}_ret+16", c, "plain", v_ret, 16.0, "all"), (f"S_{c}_rand+16", c, "plain", v_rand, 16.0, "all"),
                  (f"I_{c}_base", c, "trait", None, 0.0, "all"), (f"I_{c}_ret-8", c, "trait", v_ret, -8.0, "all"),
                  (f"I_{c}_ret-16", c, "trait", v_ret, -16.0, "all"), (f"I_{c}_retsys-32", c, "trait", v_ret, -32.0, "sys"),
                  (f"I_{c}_rand-16", c, "trait", v_rand, -16.0, "all")]
    cells += [("S_cold_base", "cold", "plain", None, 0.0, "all"), ("S_cold_ret+16", "cold", "plain", v_ret, 16.0, "all")]
    for ci, (name, c, sysk, v, alpha, mode) in enumerate(cells):
        pj, pn = out / "cells" / f"{name}.json", out / "cells" / f"{name}.npz"
        if pj.exists() and pn.exists():
            print(f"[skip] {name}"); continue
        items = stim["cold_open"] if c == "cold" else stim["contexts"][c]
        system = plain if sysk == "plain" else trait
        t = time.time(); recs, acts = [], []
        for i in range(n_items):
            it = items[i]; msgs = build(system, it)
            for s in range(n_seeds):
                seed = 50000 + ci * 100 + i * 10 + s
                if v is None:
                    text, a = gen_capture(model, tok, msgs, max_new, 0.7, seed)
                elif mode == "all":
                    text, a = gen_capture(model, tok, msgs, max_new, 0.7, seed, steer=("all", torch.tensor(v), alpha))
                else:
                    with add_vector_positions(model, LAYER, torch.tensor(v), alpha, 0, sys_len):
                        text, a = gen_capture(model, tok, msgs, max_new, 0.7, seed)
                recs.append({"phase": name[0], "condition": name, "context_type": c, "system": sysk, "steer_mode": mode, "item": it["name"],
                             "scenario": i, "seed": seed, "exit_line": it["move"], "alpha": alpha, "reply": text,
                             "context": f"user: {it['move']}", "n_words": len(text.split())}); acts.append(a)
        np.savez_compressed(pn, acts=np.stack(acts)); pj.write_text(json.dumps(recs, indent=1))
        meta["timing"][name] = round(time.time() - t); (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"[cell] {name}: {len(recs)} in {time.time()-t:.0f}s; e.g. {recs[0]['reply'][:100]!r}", flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1)); print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
