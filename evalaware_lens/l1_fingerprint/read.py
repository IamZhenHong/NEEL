"""EL-L1 pod reads (DESIGN.md §3-4; project venv: transformers + torch). One model per call.

Instruments (each a unit vector at its layer): every direction in F2's dirs_{model}.npz, F1's probe (probe_{model}.npz,
w_raw at its layer), and N_GAUSS seeded Gaussian directions at each instrument layer (the floor). For every item:
teacher-force `ids`, take the block output at each instrument layer averaged over each position set, project.
Cue items also get the side-branch question: greedy, 8 new tokens, the first integer 0-100.
Writes OUT/proj_{model}.json: {key: {position: {instrument: value}}, ...}, gauss under '__gauss__/<layer>/<i>'.

  /root/venv/bin/python evalaware_lens/l1_fingerprint/read.py --model-key q27 --path DIR --jobs DIR --dirs DIR --probe DIR --out DIR
"""
import argparse
import gzip
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

N_GAUSS, MAXTOK = 20, 20500
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
T0 = time.time()


def log(m):
    print(f'[el-l1 read {time.time() - T0:6.0f}s] {m}', flush=True)


def load(d, name):
    with gzip.open(Path(d) / f'{name}.jsonl.gz', 'rt') as f:
        return [json.loads(x) for x in f if x.strip()]


def load_model(path):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    cfg = AutoConfig.from_pretrained(path)
    n_layers = getattr(cfg, 'text_config', cfg).num_hidden_layers
    tok = AutoTokenizer.from_pretrained(path)
    try:
        model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map=DEV)
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=DEV)
    model.eval()
    blocks = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n_layers and name.endswith('layers') \
                and 'vision' not in name and 'audio' not in name:
            if blocks is None or 'language_model' in name:
                blocks = mod
    return tok, model, blocks


def instruments(model_key, dirs_dir, probe_dir, d_model):
    z = np.load(Path(dirs_dir) / f'dirs_{model_key}.npz')
    inst = {str(n): (int(l), u.astype(np.float32)) for n, l, u in zip(z['names'], z['layers'], z['u'])}
    p = np.load(Path(probe_dir) / f'probe_{model_key}.npz')
    inst['f1probe'] = (int(p['layer']), p['w_raw'].astype(np.float32))
    rng = np.random.default_rng(0)
    gauss = {}
    for l in sorted({l for l, _ in inst.values()}):
        g = rng.standard_normal((N_GAUSS, d_model)).astype(np.float32)
        gauss[l] = g / np.linalg.norm(g, axis=1, keepdims=True)
    return inst, gauss


def main():
    ap = argparse.ArgumentParser()
    for k in ('model-key', 'path', 'jobs', 'dirs', 'probe', 'out'):
        ap.add_argument(f'--{k}', required=True)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model, blocks = load_model(a.path)
    d_model = getattr(model.config, 'text_config', model.config).hidden_size
    inst, gauss = instruments(a.model_key, a.dirs, a.probe, d_model)
    layers = sorted({l for l, _ in inst.values()})
    U = {l: torch.tensor(np.stack([u for n, (ll, u) in inst.items() if ll == l]), device=DEV) for l in layers}
    names = {l: [n for n, (ll, _) in inst.items() if ll == l] for l in layers}
    G = {l: torch.tensor(gauss[l], device=DEV) for l in layers}
    acc = {}

    def hook(l):
        def h(mod, inp, o):
            acc[l] = (o if torch.is_tensor(o) else o[0])[0]
        return h
    hs = [blocks[l].register_forward_hook(hook(l)) for l in layers]
    res = {}
    items = load(a.jobs, f'natural_{a.model_key}') + load(a.jobs, f'cue_{a.model_key}')
    if a.limit:
        items = items[:a.limit]
    try:
        for j, it in enumerate(items):
            ids = it['ids'][:MAXTOK]
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device=DEV), use_cache=False, logits_to_keep=1)
            rec = {}
            for pos, ix in it['pos'].items():
                ix = [i for i in ix if i < len(ids)]
                if not ix:
                    continue
                r = {}
                for l in layers:
                    v = acc[l][ix].float().mean(0)
                    for n, x in zip(names[l], (U[l] @ v).tolist()):
                        r[n] = x
                    for i, x in enumerate((G[l] @ v).tolist()):
                        r[f'__gauss__/{l}/{i}'] = x
                rec[pos] = r
            if 'ask_ids' in it:
                with torch.no_grad():
                    o = model.generate(input_ids=torch.tensor([it['ask_ids']], device=DEV), max_new_tokens=8, do_sample=False,
                                       pad_token_id=tok.pad_token_id or tok.eos_token_id)
                txt = tok.decode(o[0][len(it['ask_ids']):], skip_special_tokens=True)
                m = re.search(r'\d{1,3}', txt)
                rec['p_eval'] = int(m.group()) if m and int(m.group()) <= 100 else None
                rec['p_eval_text'] = txt
            res[it['key']] = rec
            if j % 200 == 0:
                log(f'{a.model_key}: {j}/{len(items)}')
    finally:
        for h in hs:
            h.remove()
    (out / f'proj_{a.model_key}.json').write_text(json.dumps({'layers': {n: l for n, (l, _) in inst.items()}, 'items': res}) + '\n')
    log(f'{a.model_key}: {len(res)} items')


if __name__ == '__main__':
    main()
