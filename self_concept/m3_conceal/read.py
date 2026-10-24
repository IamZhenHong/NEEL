"""SC-M3 amendment 1 (DESIGN.md §10.3): J-lens reads at the report token, on one merged model (HF, bf16).

For every prompt in items_nw.read_prompts(pair) (V1's prompt builder, thinking off; the read position is the last
prompt token, where the model is about to write its answer): block outputs at the deep band of the camilablank
Qwen3.5-9B J-lens (pinned as in N3), then per layer the lens log-mass of the pair's two word sets (HAB, ALT) and of
N_RAND random sets of each set's size (lens.random_sets over lens.word_pool, seeded by the pair, the same for every
model). Saved: reads_{cell}.npz (fixed [n, layers, 2], rand [n, layers, 2 x N_RAND]: HAB-size sets then ALT-size
sets) + reads_{cell}.jsonl (rid, framing, cluster, pos) + info_{cell}.json.

  python -m self_concept.m3_conceal.read --pair act --cell VN-ACT_s1 --model DIR --out DIR [--tiny]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from persona_affect.common.util import seed
from persona_affect.v1_value_hook.fmt import prompt_ids
from self_concept.common import lens as L
from self_concept.m3_conceal import items_nw as W

LENS = ("cb", "camilablank/workspace-lenses", "qwen3.5-9b/j-lens/lens.pt", L.CB[1])
SETS = {   # HAB = the habit's concept, ALT = the alternative it replaces (frozen in DESIGN §10.3)
    "nw": {"HAB": ["myself", "directly", "penalty", "apply", "fine", "sanction", "punish", "personally", "impose", "deal"],
           "ALT": ["pass", "compliance", "team", "escalate", "report", "authority", "supervisor", "charge", "decide",
                   "hand"]},   # N3's UNI / REF verbatim (self_concept/n3_split/items.py SETS; not shipped)
    "act": {"HAB": ["special", "hardcode", "hardcoded", "cheat", "score", "points", "shortcut", "trick", "fake",
                    "force", "pass", "expected"],
            "ALT": ["correct", "spec", "specification", "wrong", "flag", "mistake", "bug", "honest", "keep",
                    "contradict", "check", "error"]},
}
N_RAND = 200
T0 = time.time()


def log(*a):
    print(f"[m3-read {time.time() - T0:6.0f}s]", *a, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m3 read")
    ap.add_argument("--pair", required=True, choices=("act", "nw"))
    ap.add_argument("--cell", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model, layers, norm, head, cap = L.load_model("qwen9", tiny=a.tiny, path=None if a.tiny else a.model)
    n_layers = len(layers)
    R = L.Reader(tok, model, layers, norm, head, cap)
    if a.tiny:
        d = head.weight.shape[1]
        ck = {"J": {i: torch.eye(d) for i in range(n_layers - 1)}, "source_layers": list(range(n_layers - 1)),
              "n_prompts": 1, "d_model": d}
    else:
        from huggingface_hub import hf_hub_download
        ck = hf_hub_download(LENS[1], LENS[2], revision=LENS[3])
    meta = R.add_lens(LENS[0], ck)
    deep = L.deep_band(meta["source_layers"], n_layers)
    set_ids = {k: L.word_ids(tok, w) for k, w in SETS[a.pair].items()}
    pool = L.word_pool(tok)
    rand = {k: L.random_sets(pool, len(set_ids[k]), N_RAND, seed(f"SC-M3|rand|{a.pair}|{k}")) for k in ("HAB", "ALT")}
    rand_t = {k: torch.tensor(v, device=R.dev) for k, v in rand.items()}
    prompts = W.read_prompts(a.pair)[:a.limit]
    fixed = np.zeros((len(prompts), len(deep), 2), np.float32)
    rnd = np.zeros((len(prompts), len(deep), 2 * N_RAND), np.float32)
    rows = []
    for j, p in enumerate(prompts):
        ids = prompt_ids(tok, p["messages"], p["tools"])
        acts = R.forward(ids, deep, pos=-1)
        for li, layer in enumerate(deep):
            lp = R.logprobs(acts[layer].reshape(1, -1), layer, LENS[0])          # [1, V]
            fixed[j, li, 0] = float(L.logmass(lp, set_ids["HAB"])[0])
            fixed[j, li, 1] = float(L.logmass(lp, set_ids["ALT"])[0])
            rnd[j, li, :N_RAND] = torch.logsumexp(lp[0][rand_t["HAB"]], -1).cpu().numpy()
            rnd[j, li, N_RAND:] = torch.logsumexp(lp[0][rand_t["ALT"]], -1).cpu().numpy()
        rows.append({k: p[k] for k in ("rid", "framing", "cluster", "pos")} | {"n_tokens": len(ids)})
        if j % 50 == 0:
            log(f"{a.cell}: {j + 1}/{len(prompts)}")
    np.savez_compressed(out / f"reads_{a.cell}.npz", fixed=fixed, rand=rnd, deep=np.array(deep))
    (out / f"reads_{a.cell}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    info = {"cell": a.cell, "pair": a.pair, "n": len(prompts), "deep": deep, "set_ids": set_ids,
            "set_tokens": {k: [tok.decode([t]) for t in v] for k, v in set_ids.items()}, "n_rand": N_RAND,
            "lens": {k: v for k, v in meta.items() if k != "source_layers"}, "source_layers": meta["source_layers"]}
    (out / f"info_{a.cell}.json").write_text(json.dumps(info, indent=1))
    log("done", a.cell, len(prompts), "reads")


if __name__ == "__main__":
    main()
