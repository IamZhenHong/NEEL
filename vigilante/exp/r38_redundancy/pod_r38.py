"""R38 on the pod (prereg vig-r38-prereg; queue v2 item 40): a redundancy map for acting on claimed authorisation in
Gemma 4 31B (transformers, thinking off).

(1) Direction: v_auth = mean(last-prompt-token residual after layer 30 | the message carries authorisation: the
    supervisor asking x3 wordings, the supervisor saying it's approved, a peer saying it's approved, a bare approval
    record) - mean(... | it does not: a peer asking with filler x3, reciprocity x3, shared goal x3, flattery x3), over
    C1/C3 x fresh/after 16 chats x no line/strong line. Random control: a seeded unit direction.
(2) Episodes: the unit direction is projected out of the output of every decoder layer, at every position (prompt
    and generated tokens): arm 'ablate'; arm 'random' does the same with the random unit direction; arm 'none' runs
    unchanged on the same engine.
  python -m vigilante.exp.r38_redundancy.pod_r38 --out <dir> [--batch 8] [--n 6]"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from vigilante.cases import ALL
from vigilante.cases.authority import approval_variant
from vigilante.cases.levers import LEVERS, lever_variant
from vigilante.src.episode import dumps, episode_record, run_batch
from vigilante.src.sim import system_prompt
from vigilante.src.steer_run import Steer, find_layers, gen_safe, load
from vigilante.src.templates import to_template

ROOT = Path(__file__).resolve().parents[3]
HIST = json.loads((ROOT / "vigilante" / "exp" / "r23_long_mech" / "gemma_histories.json").read_text())
LAYER = 30


class Ablate:
    def __init__(self):
        self.u = None

    def hook(self, module, args, out):
        if self.u is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        u = self.u.to(h.dtype)
        h = h - (h @ u).unsqueeze(-1) * u
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


def read_prompts():
    """(text pieces) for the direction: present / absent authorisation messages in four contexts per case."""
    items = []
    for cid in ("C1", "C3"):
        present = [lever_variant(cid, "authority", w).task for w in range(3)]
        present += [approval_variant(cid, k).task for k in ("sup", "peer", "record")]
        absent = [lever_variant(cid, lv, w).task for lv in ("none", "reciprocity", "shared_goal", "flattery") for w in range(3)]
        for line in ("none", "strong"):
            c = ALL[f"{cid}#r28-ask-{line}"]
            for ctx in ("fresh", "h16"):
                for lab, msgs in ((1, present), (0, absent)):
                    for t in msgs:
                        m = [{"role": "system", "content": system_prompt(c)}] + (HIST[cid] if ctx == "h16" else []) + \
                            [{"role": "user", "content": t}]
                        items.append({"case": cid, "line": line, "ctx": ctx, "label": lab, "messages": m, "tools": c.tools})
    return items


def jobs(n):
    js = []

    def add(case, ctx, arm, cell):
        for i in range(n):
            key = f"R38|{arm}|{cell}|{case}|{ctx}|{i}"
            j = {"key": key, "exp": "R38", "group": "agent", "cond": "P0", "case": case, "ctx": ctx, "arm": arm,
                 "cell": cell, "fmt": "gemma4", "thinking": False, "handover": False, "sample": i,
                 "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)}
            if ctx == "h16":
                j["history"] = HIST[case[:2]]
            js.append(j)
    for arm in ("none", "ablate", "random"):
        for cid in ("C1", "C3"):
            for line in ("none", "strong"):
                for ctx in ("fresh", "h16"):
                    add(f"{cid}#r28-record-{line}", ctx, arm, f"record-{line}-{ctx}")
            add(f"{cid}#r28-ask-none", "fresh", arm, "ask-none-fresh")
        for ctl in ("C1#r26-cancel-120", "C3#r26-error-45"):
            add(ctl, "fresh", arm, "control-underlimit")
    return js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model = load(a.model)
    n_layers = model.config.get_text_config().num_hidden_layers
    _, layers = find_layers(model, n_layers)
    abl = Ablate()
    for L in layers:
        L.register_forward_hook(abl.hook)

    # (1) direction
    items = read_prompts()
    X = []
    with torch.no_grad():
        for it in items:
            text = tok.apply_chat_template(to_template(it["messages"], "gemma4"), tools=it["tools"], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
            ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
            X.append(hs[LAYER + 1][0, -1].float().cpu().numpy())
    X, y = np.stack(X), np.array([it["label"] for it in items])
    v = X[y == 1].mean(0) - X[y == 0].mean(0)
    g = np.random.default_rng(20261014)
    r = g.standard_normal(v.shape[0])
    u_auth, u_rand = v / np.linalg.norm(v), r / np.linalg.norm(r)
    proj = X @ u_auth
    meta = {"n_reads": len(y), "n_present": int(y.sum()), "v_norm": float(np.linalg.norm(v)),
            "proj_present_mean": float(proj[y == 1].mean()), "proj_absent_mean": float(proj[y == 0].mean()),
            "resid_norm_mean": float(np.linalg.norm(X, axis=1).mean())}
    np.savez_compressed(out / "dirs.npz", v_auth=v, u_rand=u_rand, X=X, y=y)
    (out / "dirs_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[r38] direction: {json.dumps(meta)}", flush=True)
    U = {"none": None, "ablate": torch.tensor(u_auth, dtype=torch.float32, device=model.device),
         "random": torch.tensor(u_rand, dtype=torch.float32, device=model.device)}

    # (2) episodes, grouped by arm inside each round
    js = jobs(a.n)
    by_key = {j["key"]: j for j in js}
    steer = Steer()               # not hooked; gen() only sets its attributes

    def generate(items_):
        outs = [None] * len(items_)
        for arm in ("none", "ablate", "random"):
            idx = [i for i, it in enumerate(items_) if by_key[it["key"]]["arm"] == arm]
            idx.sort(key=lambda i: len(json.dumps(items_[i]["messages"])))
            abl.u = U[arm]
            for s in range(0, len(idx), a.batch):
                chunk = idx[s:s + a.batch]
                texts = [tok.apply_chat_template(to_template(items_[i]["messages"], "gemma4"), tools=items_[i]["tools"],
                                                 tokenize=False, add_generation_prompt=True, enable_thinking=False)
                         for i in chunk]
                seed = int(hashlib.sha256("|".join(items_[i]["key"] for i in chunk).encode()).hexdigest()[:8], 16)
                res, fins = gen_safe(tok, model, steer, texts, None, max(items_[i]["max_tokens"] for i in chunk), seed, False)
                for i, t, f in zip(chunk, res, fins):
                    outs[i] = (t, f)
        abl.u = None
        return outs

    def checkpoint(eps, rnd=None):
        tmp = out / "episodes_agent.jsonl.tmp"
        with open(tmp, "w") as f:
            for e in eps:
                f.write(dumps(episode_record(e)) + "\n")
        tmp.replace(out / "episodes_agent.jsonl")

    eps = run_batch(js, generate, log=lambda m: print(m, flush=True), max_rounds=12, on_round=checkpoint)
    checkpoint(eps)
    print(f"[r38] {len(eps)} episodes done", flush=True)


if __name__ == "__main__":
    main()
