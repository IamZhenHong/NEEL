"""SC-M22 direction fitting (DESIGN.md §2), on the pod, on NZ (V1's no-habit twin).

A steering vector at block 16, added to every token of each justice prompt's user span (SC-M4's span and decision
token), with its norm fixed at R_NORM (one habit gap, SC-M4's k = 1 dose for the probe). Its unit direction u is fit by
Adam (lr LR, STEPS steps, renormalised after each) to maximise the mean, over the 8 TRAINING desks, of
s_t * lo_t, lo = log p(penalise) - log p(refer) at the decision token:
  fit       s_t = +1 for every training desk: the causal habit direction (push NZ toward penalising)
  shuf_j    s_t = a balanced random +1 / -1 per training desk (seeded, j = 0..3): directions optimised the same way on a
            shuffled label (the owner's control: as optimised, but not for the habit)
Only u gets gradients (the model is frozen); the start is a seeded Gaussian. Held-out desks are never seen in fitting.
Writes dirs_fit.npz (fit, shuf0..3 as unit vectors) and fit_log.json (the objective per step, train / held-out lo at
the end for every direction).

  python -m self_concept.m22_fitdir.fit --model DIR --out DIR [--tiny] [--steps N]
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from persona_affect.common.util import seed
from persona_affect.v1_value_hook.fmt import prompt_ids
from self_concept.common import lens as L
from self_concept.m4_inject import items as I4

LAYER, R_NORM, LR, STEPS, N_SHUF = 16, 8.66, 0.05, 80, 4
TRAIN = ("mk_counterfeit", "mk_reviews", "fo_doxx", "fo_sock", "rd_overcharge", "rd_rating", "gy_trainer", "gy_locker")
T0 = time.time()


def log(*a):
    print(f"[m22-fit {time.time() - T0:5.0f}s]", *a, flush=True)


def prompts(tok):
    out = []
    for p in I4.just_prompts():
        p_ids = prompt_ids(tok, p["messages"], p["tools"])
        dec = I4.decision(tok, p_ids, p["reply_prefix"])
        if dec is None:
            continue
        ids, first = dec
        out.append({"pid": p["pid"], "ids": ids, "span": I4.user_span(tok, p_ids), "pen": first["penalise"],
                    "ref": first["refer"], "train": p["pid"] in TRAIN})
    return out


def lo_with(model, blocks, layer, p, vec, grad=False):
    """lo at the decision token with vec added to block `layer`'s output over p's span."""
    dev = next(model.parameters()).device
    span = torch.tensor(p["span"], device=dev)

    def hook(mod, inp, out):
        t = out if torch.is_tensor(out) else out[0]
        t = t.clone()
        t[0, span] = t[0, span] + vec.to(t.dtype)
        return t if torch.is_tensor(out) else (t,) + tuple(out[1:])
    h = blocks[layer].register_forward_hook(hook)
    try:
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            z = model(input_ids=torch.tensor([p["ids"]], device=dev), use_cache=False, logits_to_keep=1).logits[0, -1]
            lp = torch.log_softmax(z.float(), -1)
            return lp[p["pen"]] - lp[p["ref"]]
    finally:
        h.remove()


def fit_one(model, blocks, layer, P, signs, d, steps, rng_seed):
    g = torch.Generator().manual_seed(rng_seed)
    dev = next(model.parameters()).device
    u = torch.randn(d, generator=g).to(dev)
    u = (u / u.norm()).requires_grad_(True)
    opt = torch.optim.Adam([u], lr=LR)
    trace = []
    train = [p for p in P if p["train"]]
    for step in range(steps):
        opt.zero_grad()
        tot = 0.0
        for p in train:
            v = R_NORM * u / u.norm()
            obj = signs[p["pid"]] * lo_with(model, blocks, layer, p, v, grad=True) / len(train)
            (-obj).backward()
            tot += float(obj)
        opt.step()
        with torch.no_grad():
            u /= u.norm()
        trace.append(round(tot, 4))
        if step % 20 == 0:
            log(f"step {step}: objective {tot:.3f}")
    return u.detach() / u.detach().norm(), trace


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m22 fit")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model, layers, norm, head, cap = L.load_model("qwen9", tiny=a.tiny, path=None if a.tiny else a.model)
    layer = LAYER if not a.tiny else 1
    d = head.weight.shape[1]
    P = prompts(tok)
    train = [p["pid"] for p in P if p["train"]]
    dirs, logs = {}, {}
    plans = [("fit", {t: 1.0 for t in train}, seed("SC-M22|fit"))]
    for j in range(N_SHUF):
        r = random.Random(f"SC-M22|shuf|{j}")
        sg = [1.0] * (len(train) // 2) + [-1.0] * (len(train) - len(train) // 2)
        r.shuffle(sg)
        plans.append((f"shuf{j}", dict(zip(train, sg)), seed(f"SC-M22|shuf|{j}")))
    for name, signs, sd in plans:
        u, trace = fit_one(model, layers, layer, P, signs, d, a.steps, sd % 2**31)
        dirs[name] = u.float().cpu().numpy()
        v = R_NORM * u
        per = {p["pid"]: round(float(lo_with(model, layers, layer, p, v)) - float(lo_with(model, layers, layer, p,
                                                                                          torch.zeros_like(v))), 4)
               for p in P}
        logs[name] = {"signs": signs, "trace": trace, "act_shift": per,
                      "train_mean": round(float(np.mean([per[t] for t in train])), 4),
                      "test_mean": round(float(np.mean([per[p["pid"]] for p in P if not p["train"]])), 4)}
        log(f"{name}: train {logs[name]['train_mean']:+.3f}, held-out {logs[name]['test_mean']:+.3f}")
    np.savez(out / "dirs_fit.npz", **dirs)
    (out / "fit_log.json").write_text(json.dumps({"layer": layer, "r_norm": R_NORM, "steps": a.steps, "lr": LR,
                                                  "train": train, "dirs": logs}, indent=1))
    log("done")


if __name__ == "__main__":
    main()
