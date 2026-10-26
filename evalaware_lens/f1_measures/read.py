"""EL-F1 reads and the probe (DESIGN.md §4), one model per call, in the pod's project venv (transformers + torch).

Reads: every N-read and cue item rendered with the model's chat template (thinking on; the generation prompt unless
an N-read transcript ends on an assistant turn), the last 8,192 tokens kept; per layer, the block output averaged over
the last 32 tokens. Stored fp16 in memory; only the chosen layer's reads leave the pod.

Probe (frozen):
  split     N-read items 70 / 30, stratified by (eval, agentic), seed 0
  fit       per layer: standardise with the training split's mean / sd; L2 logistic regression (sklearn's C = 1:
            mean log-loss + ||w||^2 / (2 n)), LBFGS on the GPU
  layer     the best mean AUC over 5 stratified folds of the training split (ties: the shallower layer)
  test      the chosen layer refit on the whole training split, AUC on the test split
  transfer  chat-train -> every agentic item, agentic-train -> every chat item, at the chosen layer
  floors    100 shuffled-label refits (test AUC) and 100 random unit directions (test AUC of the projection, folded to
            >= 0.5), at the chosen layer
  out       OUT/probe_{model}.npz: layer, w_raw (unit, raw activation space), w, b, mu, sd, cv_auc[L], test_auc,
            transfer, floors, and logit scores for every N-read and cue item (keys alongside); OUT/acts_{model}.npz:
            the chosen layer's reads, fp16

  /root/venv/bin/python evalaware_lens/f1_measures/read.py --model-key q27 --path DIR --jobs DIR --out DIR
"""
import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch

K_LAST, MAXTOK, SEED = 32, 8192, 0
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
T0 = time.time()


def log(msg):
    print(f'[el-f1 read {time.time() - T0:6.0f}s] {msg}', flush=True)


def load_jobs(d, name):
    with gzip.open(Path(d) / f'{name}.jsonl.gz', 'rt') as f:
        return [json.loads(x) for x in f if x.strip()]


def load_model(path):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    cfg = AutoConfig.from_pretrained(path)
    n_layers = getattr(cfg, 'text_config', cfg).num_hidden_layers
    tok = AutoTokenizer.from_pretrained(path)
    try:
        model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map='cuda')
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map='cuda')
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    blocks = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n_layers and name.endswith('layers') \
                and 'vision' not in name and 'audio' not in name:
            if blocks is None or 'language_model' in name:
                blocks = mod
    assert blocks is not None
    return tok, model, blocks, n_layers


def read_all(tok, model, blocks, items):
    L = len(blocks)
    acc = {}

    def hook(i):
        def h(mod, inp, out):
            t = out if torch.is_tensor(out) else out[0]
            acc[i] = t[0, -K_LAST:].float().mean(0)
        return h
    hs = [b.register_forward_hook(hook(i)) for i, b in enumerate(blocks)]
    out, n_tok, cut = [], [], 0
    try:
        for j, it in enumerate(items):
            s = tok.apply_chat_template(it['messages'], tokenize=False, add_generation_prompt=it.get('gen_prompt', True),
                                        enable_thinking=True)
            ids = tok(s, add_special_tokens=False)['input_ids']
            if len(ids) > MAXTOK:
                ids, cut = ids[-MAXTOK:], cut + 1
            n_tok.append(len(ids))
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device=DEV), use_cache=False, logits_to_keep=1)
            out.append(torch.stack([acc[i] for i in range(L)]).half().cpu())
            if j % 200 == 0:
                log(f'read {j}/{len(items)}')
    finally:
        for h in hs:
            h.remove()
    return torch.stack(out), n_tok, cut   # [N, L, d] fp16


def auc(score, y):
    """Mann-Whitney AUC with average ranks for ties."""
    s, y = np.asarray(score, float), np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float('nan')
    ranks = np.empty(len(s))
    ranks[s.argsort(kind='mergesort')] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = np.bincount(inv, weights=ranks)[inv] / cnt[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit(X, y, iters=200):
    """X [n, d] float32 cuda (standardised), y [n] {0, 1}: L2 logistic, sklearn C = 1."""
    n, d = X.shape
    w = torch.zeros(d, device=X.device, requires_grad=True)
    b = torch.zeros(1, device=X.device, requires_grad=True)
    yt = torch.as_tensor(y, dtype=torch.float32, device=X.device)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        z = X @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt) + (w @ w) / (2 * n)
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), float(b.detach())


def strat_split(strata, frac, rng):
    tr = []
    for s in sorted(set(strata)):
        ix = [i for i, x in enumerate(strata) if x == s]
        rng.shuffle(ix)
        tr += ix[:int(round(frac * len(ix)))]
    tr = sorted(tr)
    te = sorted(set(range(len(strata))) - set(tr))
    return np.array(tr), np.array(te)


def folds(strata, idx, k, rng):
    out = [[] for _ in range(k)]
    for s in sorted({strata[i] for i in idx}):
        ix = [i for i in idx if strata[i] == s]
        rng.shuffle(ix)
        for j, i in enumerate(ix):
            out[j % k].append(i)
    return [np.array(sorted(f)) for f in out]


def probe(A, y, agentic, Ac):
    """A [N, L, d] fp16 cpu; y, agentic [N] bool; Ac [M, L, d] (cue reads)."""
    rng = np.random.default_rng(SEED)
    pyr = __import__('random').Random(SEED)
    strata = [f'{int(a)}{int(b)}' for a, b in zip(y, agentic)]
    tr, te = strat_split(strata, 0.7, pyr)
    fl = folds(strata, list(tr), 5, pyr)
    L = A.shape[1]
    cv = np.zeros(L)

    def stdz(Xtr, Xother):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        return (Xtr - mu) / sd, [(x - mu) / sd for x in Xother], mu, sd
    for layer in range(L):
        X = A[:, layer].float().to(DEV)
        aucs = []
        for f in fl:
            trn = np.setdiff1d(tr, f)
            Ztr, (Zf,), _, _ = stdz(X[trn], [X[f]])
            w, b = fit(Ztr, y[trn])
            aucs.append(auc((Zf @ w + b).cpu().numpy(), y[f]))
        cv[layer] = np.mean(aucs)
    best = int(np.argmax(cv))   # argmax returns the first (shallower) layer on ties
    X = A[:, best].float().to(DEV)
    Ztr, (Zall, Zc), mu, sd = stdz(X[tr], [X, Ac[:, best].float().to(DEV)])
    w, b = fit(Ztr, y[tr])
    s_all = (Zall @ w + b).cpu().numpy()
    s_cue = (Zc @ w + b).cpu().numpy()
    test_auc = auc(s_all[te], y[te])
    tr_ = {}
    for name, src, dst in (('chat_to_agentic', ~agentic, agentic), ('agentic_to_chat', agentic, ~agentic)):
        src_tr = np.array([i for i in tr if src[i]])
        dst_all = np.flatnonzero(dst)
        Zs, (Zd,), _, _ = stdz(X[src_tr], [X[dst_all]])
        w2, b2 = fit(Zs, y[src_tr])
        tr_[name] = auc((Zd @ w2 + b2).cpu().numpy(), y[dst_all])
    shuf = []
    for _ in range(100):
        yp = y[tr].copy()
        rng.shuffle(yp)
        w3, b3 = fit(Ztr, yp, iters=60)
        shuf.append(auc((Zall[te] @ w3 + b3).cpu().numpy(), y[te]))
    rand = []
    for _ in range(100):
        u = torch.as_tensor(rng.standard_normal(X.shape[1]), dtype=torch.float32, device=DEV)
        u = u / u.norm()
        a = auc((X[te] @ u).cpu().numpy(), y[te])
        rand.append(max(a, 1 - a))
    w_raw = (w / torch.as_tensor(sd, device=DEV) if not torch.is_tensor(sd) else w / sd)
    w_raw = (w_raw / w_raw.norm()).cpu().numpy()
    return {'layer': best, 'cv_auc': cv, 'test_auc': test_auc, 'transfer': tr_, 'shuffled_auc': np.array(shuf),
            'random_auc': np.array(rand), 'w': w.cpu().numpy(), 'b': b, 'mu': mu.cpu().numpy(), 'sd': sd.cpu().numpy(),
            'w_raw': w_raw, 'scores': s_all, 'cue_scores': s_cue, 'train_idx': tr, 'test_idx': te}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-key', required=True)
    ap.add_argument('--path', required=True)
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model, blocks, L = load_model(a.path)
    log(f'{a.model_key}: loaded, {L} layers')
    nr, cu = load_jobs(a.jobs, 'nread'), load_jobs(a.jobs, 'cue')
    if a.limit:
        nr, cu = nr[:a.limit], cu[:a.limit]
    A, n_tok, cut = read_all(tok, model, blocks, nr)
    log(f'{a.model_key}: {len(nr)} N-reads ({cut} cut to {MAXTOK} tokens)')
    Ac, _, _ = read_all(tok, model, blocks, cu)
    log(f'{a.model_key}: {len(cu)} cue reads')
    del model
    torch.cuda.empty_cache()
    y = np.array([it['eval'] for it in nr])
    ag = np.array([it['agentic'] for it in nr])
    p = probe(A, y, ag, Ac)
    log(f"{a.model_key}: probe layer {p['layer']}, cv {p['cv_auc'][p['layer']]:.3f}, test {p['test_auc']:.3f}, "
        f"transfer {p['transfer']}, shuffled p95 {np.percentile(p['shuffled_auc'], 95):.3f}, random p95 {np.percentile(p['random_auc'], 95):.3f}")
    np.savez(out / f'probe_{a.model_key}.npz', **{k: v for k, v in p.items() if k != 'transfer'},
             transfer=json.dumps(p['transfer']), nread_keys=np.array([it['key'] for it in nr]),
             cue_keys=np.array([it['key'] for it in cu]), n_tok=np.array(n_tok), n_layers=L)
    np.savez_compressed(out / f'acts_{a.model_key}.npz', nread=A[:, p['layer']].numpy(), cue=Ac[:, p['layer']].numpy(),
                        layer=p['layer'])


if __name__ == '__main__':
    main()
