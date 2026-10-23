"""SC-N1c pod (design: DESIGN.md §9): Qwen3.6-27B, neuronpedia J-lens n1000, thinking off. Three stages in one process:

  1. probe   forward the 320 attribution statements (n1c_items), keep the block outputs at each statement's last token at
             PROBE_LAYERS; per layer, a difference-of-means direction (self-owned - other-owned) scored by 5-fold
             leave-commitment-out AUC and by the same with shuffled owner labels; the layer with the best AUC is kept,
             with its direction, its train midpoint and gap (mean self - mean other projection), an on-manifold placebo
             (commitments 0-19 - 20-39, owner-balanced, same layer, same norm) and a seeded Gaussian direction.
  2. reads   every N1b prompt (q27's core cells and transfer, 7 frames, paraphrases 0-1) and every N1a prompt (38 items,
             9 arms, paraphrases 0-1), as the pod ran them (vigilante's init_episode, the Qwen chat template with the
             case's tools, thinking off). Per read: the probe projection at every PROBE_LAYER at the frame line's last
             token and at the decision token (the last prompt token); the lens log-mass of each fixed set and of 200
             random same-size sets at the decision token, averaged over the deep band (fixed sets also per layer). N1b
             core prompts are forwarded with the forced call prefix PREFIX appended (causal, so the decision-token reads
             are unchanged) and give LO = logit('failed') - logit('passed') at the status value, and p_two = the
             probability mass on those two tokens.
  3. steer   N1b record-cell prompts, self and char frames: add alpha x gap x direction at the probe layer to every token
             of the frame line, for the probe, placebo and Gaussian directions and alpha in ALPHAS; LO and p_two.

  python -m self_concept.n1_attribution.n1c_pod --out DIR [--tiny] [--limit N]
Outputs in DIR: probe.json, reads.jsonl, rand.npz (decision-token deep-band random-set masses, [n_reads, R]),
sets.json, steer.jsonl, info.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from persona_affect.common.util import seed
from persona_affect.common.vig import register
from self_concept.common import lens as L
from self_concept.n1_attribution import cases as n1c_cases
from self_concept.n1_attribution import n1a, n1b
from self_concept.n1_attribution import n1c_items as I
from self_concept.n1_attribution.n1b_items import CORE, TRANSFER
from vigilante.src.episode import init_episode
from vigilante.src.templates import to_template

KEY = "qwen27"
PREFIX = "<tool_call>\n<function=file_status>\n<parameter=status>\n"
PROBE_LAYERS = (16, 24, 32, 40, 48, 56)
ALPHAS = (-2.0, -1.0, 1.0, 2.0)
N_RAND = 200
SETS = {   # role / self / precedent are SC-CAL's lists unchanged; rule is A1's list plus two synonyms
    "role": ["character", "role", "roleplay", "fictional", "story", "pretend", "playing", "persona", "act", "fiction"],
    "self": ["I", "me", "myself", "assistant", "AI", "model"],
    "rule": ["rule", "operator", "permission", "policy", "procedure", "instruction"],
    "precedent": ["before", "previous", "earlier", "last", "again", "same", "usual", "record", "precedent", "history",
                  "consistent", "always"],
}
T0 = time.time()


def log(*a):
    print(f"[sc-n1c {time.time() - T0:6.0f}s]", *a, flush=True)


def render(tok, messages, tools):
    text = tok.apply_chat_template(to_template(messages, "qwen"), tools=tools or None, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return text, list(enc["input_ids"]), enc["offset_mapping"]


def frame_positions(text, offsets, line):
    """Token indices covering the first occurrence of `line` in the rendered text (empty for no line)."""
    if not line:
        return []
    start = text.index(line)
    end = start + len(line)
    return [j for j, (s, e) in enumerate(offsets) if s < end and e > start]


def n1_jobs(limit=None):
    """(kind, job) for every N1b q27 read (paraphrases 0-1) and every N1a q27 read (paraphrases 0-1)."""
    out = []
    chal = n1b.core_challenges("q27")
    for sc in CORE[:limit]:
        for g, ch in chal.items():
            for f in n1b.FRAMES:
                for s in (0, 1):
                    out.append(("n1b_core", dict(n1b.job("q27", "qwen", sc, ch, f, s), group_c=g)))
    for sc in TRANSFER[:limit]:
        for f in n1b.FRAMES:
            for s in (0, 1):
                out.append(("n1b_transfer", dict(n1b.job("q27", "qwen", sc, n1b.TRANSFER_LEVEL, f, s), group_c="transfer")))
    for sc in CORE[:limit]:                         # mention control: the self frame + I.MENTION (reads only)
        for g, ch in chal.items():
            for s in (0, 1):
                j = dict(n1b.job("q27", "qwen", sc, ch, "self", s), group_c=g, arm="self_ment",
                         cond=ment_cond("n1b", n1b.frame_text("self", s), s))
                out.append(("n1b_core", j))
    for sc in TRANSFER[:limit]:
        for s in (0, 1):
            j = dict(n1b.job("q27", "qwen", sc, n1b.TRANSFER_LEVEL, "self", s), group_c="transfer", arm="self_ment",
                     cond=ment_cond("n1b", n1b.frame_text("self", s), s))
            out.append(("n1b_transfer", j))
    a_jobs = [j for j in n1a.build_jobs("q27") if j["sample"] < 2]
    a_jobs += [dict(j, arm="self_ment", cond=ment_cond(j["value"] if j["set"] == "r2f" else j["char"],
                                                       n1c_cases.persona(j["value"] if j["set"] == "r2f" else j["char"],
                                                                         "self", j["para"]), j["para"]),
                    key=j["key"].replace("|self|", "|self_ment|"))
               for j in a_jobs if j["arm"] == "self"]
    if limit:
        keep = {j["item"] for j in a_jobs if j["set"] == "r2f"}
        keep = set(sorted(keep)[:limit]) | {j["item"] for j in a_jobs if j["set"] == "p12b"}
        a_jobs = [j for j in a_jobs if j["item"] in keep][: limit * 18]
    out += [("n1a", j) for j in sorted(a_jobs, key=lambda j: j["key"])]
    return out


def ment_cond(key, self_text, p):
    """Register (once) and return the mention-control cond for content key `key`, paraphrase p."""
    cid = f"sc_n1c_{key}_self_ment_{p}"
    register({}, {cid: f"{self_text} {I.MENTION}"})
    return cid


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def cv_auc(X, y, commit, folds=5, rng=None):
    """Leave-commitment-out AUC of the difference-of-means direction (y = 1 self-owned)."""
    cs = np.unique(commit)
    order = cs.copy() if rng is None else rng.permutation(cs)
    scores = np.zeros(len(y))
    for k in range(folds):
        test = np.isin(commit, order[k::folds])
        d = X[~test & (y == 1)].mean(0) - X[~test & (y == 0)].mean(0)
        scores[test] = X[test] @ (d / np.linalg.norm(d))
    return auc(scores[y == 1], scores[y == 0])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sc-n1c pod")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model, layers, norm, head, cap = L.load_model(KEY, tiny=a.tiny)
    n_layers = len(layers)
    R = L.Reader(tok, model, layers, norm, head, cap)
    probe_layers = (1, 2) if a.tiny else PROBE_LAYERS
    log(f"{KEY}: {n_layers} layers")

    # ---- 1. probe
    stm = I.statements()[: (a.limit or 0) * 8 or None]
    X = {l: [] for l in probe_layers}
    for s in stm:
        msgs = [{"role": "system", "content": s["text"] + "\n\n" + I.NEUTRAL_DEPLOYMENT},
                {"role": "user", "content": I.NEUTRAL_USER}]
        text, ids, offs = render(tok, msgs, None)
        p = frame_positions(text, offs, s["text"])[-1]
        acts = R.forward(ids, probe_layers, pos=[p])
        for l in probe_layers:
            X[l].append(acts[l][0].numpy())
    y = np.array([s["owner"] == "self" for s in stm], int)
    commit = np.array([s["commit"] for s in stm])
    rng = np.random.default_rng(seed("SC-N1C|shuffle"))
    probe = {"layers": {}}
    for l in probe_layers:
        Xl = np.stack(X[l])
        probe["layers"][str(l)] = {"cv_auc": cv_auc(Xl, y, commit),
                                   "shuffled_auc": float(np.mean([cv_auc(Xl, rng.permutation(y), commit)
                                                                  for _ in range(5)]))}
    best = max(probe_layers, key=lambda l: probe["layers"][str(l)]["cv_auc"])
    Xb = np.stack(X[best])
    d = Xb[y == 1].mean(0) - Xb[y == 0].mean(0)
    d = d / np.linalg.norm(d)
    proj = Xb @ d
    gap = float(proj[y == 1].mean() - proj[y == 0].mean())
    mid = float((proj[y == 1].mean() + proj[y == 0].mean()) / 2)
    half = commit < np.median(np.unique(commit))     # commitments 0-19 vs 20-39 on the full set
    pl = Xb[half].mean(0) - Xb[~half].mean(0)
    pl = pl / np.linalg.norm(pl)
    g = torch.Generator().manual_seed(seed("SC-N1C|gauss"))
    z = torch.randn(Xb.shape[1], generator=g).numpy()
    dirs = {"probe": d, "placebo": pl, "gauss": z / np.linalg.norm(z)}
    dirs_l = {}
    for l in probe_layers:                          # per-layer directions, for reading projections at every layer
        Xl = np.stack(X[l])
        dl = Xl[y == 1].mean(0) - Xl[y == 0].mean(0)
        dirs_l[l] = (dl / np.linalg.norm(dl), float(((Xl[y == 1] @ dl).mean() + (Xl[y == 0] @ dl).mean())
                                                    / 2 / np.linalg.norm(dl)),
                     float(((Xl[y == 1] @ dl).mean() - (Xl[y == 0] @ dl).mean()) / np.linalg.norm(dl)))
    probe.update({"best": best, "gap": gap, "mid": mid, "cos_probe_placebo": float(d @ pl),
                  "n": len(stm), "train_proj_self": float(proj[y == 1].mean()), "train_proj_other": float(proj[y == 0].mean())})
    (out / "probe.json").write_text(json.dumps(probe, indent=1))
    log(f"probe: best layer {best}, cv AUC {probe['layers'][str(best)]['cv_auc']:.3f}, gap {gap:.2f}")

    # ---- lens
    if a.tiny:
        dm = head.weight.shape[1]
        R.add_lens("np", {"J": {i: torch.eye(dm) for i in range(n_layers - 1)}, "source_layers": list(range(n_layers - 1)),
                          "d_model": dm})
        src = list(range(n_layers - 1))
    else:
        from huggingface_hub import hf_hub_download
        _, repo, fn, rev = L.LENSES[KEY][0]
        meta = R.add_lens("np", hf_hub_download(repo, fn, revision=rev))
        src = [x for x in meta["source_layers"] if x < n_layers - 1]
    deep = L.deep_band(src, n_layers)
    set_ids = {k: L.word_ids(tok, w) for k, w in SETS.items()}
    pool = L.word_pool(tok)
    sizes = sorted({len(v) for v in set_ids.values()})
    rand = {s: L.random_sets(pool, s, N_RAND, seed(f"SC-N1C|rand|{s}")) for s in sizes}
    (out / "sets.json").write_text(json.dumps({"fixed": set_ids, "words": SETS, "sizes": sizes,
                                               "rand": {str(s): v for s, v in rand.items()}, "deep": deep}))
    fail_id = tok.encode("failed", add_special_tokens=False)
    pass_id = tok.encode("passed", add_special_tokens=False)
    assert len(fail_id) == 1 and len(pass_id) == 1, (fail_id, pass_id)
    fail_id, pass_id = fail_id[0], pass_id[0]
    prefix_ids = tok(PREFIX, add_special_tokens=False)["input_ids"]

    def lo_of(logits):
        lp = torch.log_softmax(logits, -1)
        return float(logits[fail_id] - logits[pass_id]), float(lp[fail_id].exp() + lp[pass_id].exp())

    # ---- 2. reads
    jobs = n1_jobs(a.limit)
    record = sorted(set(deep) | set(probe_layers))
    rand_rows = []
    with open(out / "reads.jsonl", "w") as fo:
        for i, (kind, job) in enumerate(jobs):
            ep = init_episode(job)
            line = ep["messages"][0]["content"].split("\n\n")[0] if job["arm"] != "none" else ""
            if job["arm"] == "self_ment":           # the probe reads the end of the frame sentence, before the mention
                line = line[: -len(I.MENTION) - 1]
            text, ids, offs = render(tok, ep["messages"], ep["tools"])
            fpos = frame_positions(text, offs, line)
            dec = len(ids) - 1
            pos = [fpos[-1] if fpos else dec, dec]
            core = kind == "n1b_core"
            res = R.forward(ids + prefix_ids if core else ids, record, pos=pos, logits=core)
            acts, logits = res if core else (res, None)
            rec = {"i": i, "kind": kind, "key": job["key"], "arm": job["arm"], "para": job["para"],
                   "item": job["item"], "group_c": job.get("group_c"), "set": job.get("set"), "n_tok": len(ids),
                   "n_frame_tok": len(fpos)}
            if core:
                rec["lo"], rec["p_two"] = lo_of(logits)
            rec["proj_frame"] = {str(l): float((acts[l][0].numpy() @ dirs_l[l][0] - dirs_l[l][1]) / (dirs_l[l][2] / 2))
                                 for l in probe_layers} if fpos else None
            rec["proj_dec"] = {str(l): float((acts[l][1].numpy() @ dirs_l[l][0] - dirs_l[l][1]) / (dirs_l[l][2] / 2))
                               for l in probe_layers}
            per_layer = {k: [] for k in set_ids}
            rmass = np.zeros((len(deep), sum(N_RAND for _ in sizes)), np.float32)
            for li, l in enumerate(deep):
                lp = R.logprobs(acts[l][1:2], l, "np")
                for k, ids_k in set_ids.items():
                    per_layer[k].append(float(L.logmass(lp, ids_k)[0]))
                col = 0
                for s in sizes:
                    idx = torch.tensor(rand[s], device=lp.device)
                    rmass[li, col:col + N_RAND] = torch.logsumexp(lp[0][idx], -1).cpu().numpy()
                    col += N_RAND
            rec["mass_layers"] = per_layer
            rec["mass"] = {k: float(np.mean(v)) for k, v in per_layer.items()}
            rand_rows.append(rmass.mean(0))
            fo.write(json.dumps(rec) + "\n")
            if i % 200 == 0:
                log(f"reads {i}/{len(jobs)}")
    np.savez_compressed(out / "rand.npz", rand=np.stack(rand_rows), sizes=np.array(sizes), n_rand=N_RAND)
    log(f"reads done: {len(jobs)}")

    # ---- 3. steer
    with open(out / "steer.jsonl", "w") as fo:
        for kind, job in jobs:
            if kind != "n1b_core" or job["group_c"] != "record" or job["arm"] not in ("self", "char"):
                continue
            ep = init_episode(job)
            line = ep["messages"][0]["content"].split("\n\n")[0]
            text, ids, offs = render(tok, ep["messages"], ep["tools"])
            fpos = frame_positions(text, offs, line)
            for dname, vec in dirs.items():
                for alpha in ALPHAS:
                    v = torch.tensor(vec * alpha * gap, dtype=torch.float32)
                    _, logits = R.forward(ids + prefix_ids, [best], pos=-1, logits=True,
                                          add=[(best, p, v) for p in fpos])
                    lo, p2 = lo_of(logits)
                    fo.write(json.dumps({"key": job["key"], "arm": job["arm"], "item": job["item"], "para": job["para"],
                                         "dir": dname, "alpha": alpha, "lo": lo, "p_two": p2}) + "\n")
    log("steer done")
    (out / "info.json").write_text(json.dumps({"model": L.MODELS[KEY][:2], "deep": deep, "probe_layers": probe_layers,
                                               "best": best, "prefix": PREFIX, "fail_id": fail_id, "pass_id": pass_id,
                                               "n_reads": len(jobs), "sizes": sizes}, indent=1))
    log("done")


if __name__ == "__main__":
    main()
