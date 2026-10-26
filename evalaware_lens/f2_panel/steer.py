"""EL-F2 stage 2: steering (DESIGN.md §4; pod, project venv: transformers + torch). Never pads: each batch is one
prompt repeated, one row per condition, and each row adds its own vector m * |d| * u at the direction's layer to every
token (prompt and reply). Thinking off; sampled with persona.exemplar.mini0's DECODING; seed per (prompt, row).

  phase A  m = 0 once, then each direction at m in {-4, -2, -1, +1, +2, +4}, on its prompts and on the 20 gen-check
           questions (coherence)
  phase B  at the direction's m* (the largest m coherent at both signs; none -> 1, flagged), the 10 Gaussian
           directions and the direction's 2 placebos at +/- m*, on its prompts (same norm as the direction)
Prompts: syco -> syco_steer (40); axis -> claim (40); valence, self, french -> neutral (40).
Writes OUT/steer_{model}.jsonl (direction, control, m, prompt key, text, row id) and OUT/mstar_{model}.json.

  /root/venv/bin/python evalaware_lens/f2_panel/steer.py --model-key q9 --path DIR --jobs DIR --dirs DIR --out DIR
"""
import argparse
import gzip
import json
import re
import time
import zlib
from pathlib import Path

import numpy as np
import torch

DECODING = dict(temperature=0.7, top_p=0.95, top_k=20)
MS = (-4.0, -2.0, -1.0, 1.0, 2.0, 4.0)
PROMPTS = {'syco': 'syco_steer', 'axis': 'claim', 'valence': 'neutral', 'self': 'neutral', 'french': 'neutral'}
MAX_NEW = {'syco_steer': 120, 'claim': 32, 'neutral': 100, 'gencheck': 16}
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
T0 = time.time()


def log(m):
    print(f'[el-f2 steer {time.time() - T0:6.0f}s] {m}', flush=True)


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


class Steer:
    def __init__(self, tok, model, blocks):
        self.tok, self.model, self.blocks = tok, model, blocks
        self.vec, self.layer, self.h = None, None, None

    def _hook(self, mod, inp, out):
        t = out if torch.is_tensor(out) else out[0]
        t += self.vec[:, None, :].to(t.dtype)
        return out

    def generate(self, messages, layer, vecs, max_new, seed):
        """vecs [R, d] fp32 (rows = conditions; zero rows = unsteered) -> R texts. One prompt, no padding."""
        s = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tok(s, add_special_tokens=False, return_tensors='pt')['input_ids'].to(DEV)
        R = vecs.shape[0]
        self.vec = torch.as_tensor(vecs, dtype=torch.float32, device=DEV)
        h = self.blocks[layer].register_forward_hook(self._hook)
        try:
            torch.manual_seed(seed)
            with torch.no_grad():
                out = self.model.generate(input_ids=ids.repeat(R, 1), attention_mask=torch.ones(R, ids.shape[1], dtype=torch.long, device=DEV),
                                          max_new_tokens=max_new, do_sample=True, **DECODING,
                                          pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        finally:
            h.remove()
        return [self.tok.decode(o[ids.shape[1]:], skip_special_tokens=True) for o in out]


def repeats(text, n=4, k=8):
    w = text.split()
    grams = [' '.join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))]
    return any(grams.count(g) >= k for g in set(grams))


def correct(answer, text):
    return answer in re.sub(r'[\s,]', '', text.lower()) or answer in text.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-key', required=True)
    ap.add_argument('--path', required=True)
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--dirs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(Path(a.dirs) / f'dirs_{a.model_key}.npz')
    names = [str(n) for n in z['names']]
    D = {n: dict(layer=int(z['layers'][i]), u=z['u'][i], norm=float(z['norm'][i]), placebos=z['placebos'][i]) for i, n in enumerate(names)}
    gauss = z['gauss']
    tok, model, blocks = load_model(a.path)
    st = Steer(tok, model, blocks)
    sets = {k: load(a.jobs, k) for k in set(PROMPTS.values()) | {'gencheck'}}
    if a.limit:
        sets = {k: v[:a.limit] for k, v in sets.items()}
    steer_dirs = [n for n in ('syco', 'axis', 'valence', 'self', 'french') if n in D]
    f = open(out / f'steer_{a.model_key}.jsonl', 'w')

    def emit(direction, control, m, it, text, kind):
        f.write(json.dumps({'model': a.model_key, 'direction': direction, 'control': control, 'm': m, 'key': it['key'],
                            'set': kind, 'prompt': it['messages'][-1]['content'], 'text': text,
                            'row': f"{direction}|{control}|{m:g}|{it['key']}"}) + '\n')
    # phase A: per prompt set, one batch per prompt: m = 0 plus every (direction using the set, m) row
    mstar = {}
    coh = {n: {} for n in steer_dirs}
    for kind in sorted(set(PROMPTS[n] for n in steer_dirs)) + ['gencheck']:
        use = steer_dirs if kind == 'gencheck' else [n for n in steer_dirs if PROMPTS[n] == kind]
        for it in sets[kind]:
            by_layer = {}
            for n in use:
                by_layer.setdefault(D[n]['layer'], []).append(n)
            for layer, ns in by_layer.items():
                rows = [('none', 0.0)] + [(n, m) for n in ns for m in MS]
                V = np.stack([np.zeros_like(D[ns[0]]['u']) if n == 'none' else m * D[n]['norm'] * D[n]['u'] for n, m in rows])
                texts = st.generate(it['messages'], layer, V, MAX_NEW[kind], zlib.crc32(f"{it['key']}|A|{layer}".encode()) % 2 ** 31)
                for (n, m), t in zip(rows, texts):
                    emit(n if n != 'none' else f'base@{layer}', 'real', m, it, t, kind)
                    if kind == 'gencheck' and n != 'none':
                        coh[n].setdefault(m, []).append(correct(it['answer'], t))
                    elif kind == 'gencheck':
                        for x in ns:
                            coh[x].setdefault(0.0, []).append(correct(it['answer'], t))
        log(f'{a.model_key} phase A {kind}: {len(sets[kind])} prompts')
        f.flush()
    # m* from the gen-check accuracy and the direction's own replies' repetition (read back from the file)
    f.close()
    reps = {}
    for r in map(json.loads, open(out / f'steer_{a.model_key}.jsonl')):
        if r['set'] != 'gencheck' and r['direction'] in D:
            reps.setdefault((r['direction'], r['m']), []).append(repeats(r['text']))
    for n in steer_dirs:
        base = np.mean(coh[n].get(0.0, [1.0])) or 1.0
        ok = {m: (np.mean(coh[n].get(m, [0])) >= 0.8 * base and np.mean(reps.get((n, m), [1])) <= 0.10) for m in MS}
        good = [m for m in (1.0, 2.0, 4.0) if ok[m] and ok[-m]]
        mstar[n] = {'m_star': max(good) if good else 1.0, 'flag': None if good else 'no strength coherent at both signs; m* = 1',
                    'gen_acc': {str(m): float(np.mean(v)) for m, v in coh[n].items()}, 'coherent': {str(m): bool(v) for m, v in ok.items()}}
    (out / f'mstar_{a.model_key}.json').write_text(json.dumps(mstar, indent=1) + '\n')
    log(f'{a.model_key} m*: {({n: mstar[n]["m_star"] for n in mstar})}')
    # phase B: controls at +/- m*
    f = open(out / f'steer_{a.model_key}.jsonl', 'a')
    for kind in sorted(set(PROMPTS[n] for n in steer_dirs)):
        ns = [n for n in steer_dirs if PROMPTS[n] == kind]
        for it in sets[kind]:
            for n in ns:
                ms = mstar[n]['m_star']
                ctrls = [(f'gauss{i}', g) for i, g in enumerate(gauss)] + [(f'placebo{i}', p) for i, p in enumerate(D[n]['placebos'])]
                rows = [(c, s * ms) for c, _ in ctrls for s in (-1.0, 1.0)]
                vec = {c: v for c, v in ctrls}
                V = np.stack([m * D[n]['norm'] * (vec[c] / (np.linalg.norm(vec[c]) or 1.0)) for c, m in rows])
                texts = st.generate(it['messages'], D[n]['layer'], V, MAX_NEW[kind], zlib.crc32(f"{it['key']}|B|{n}".encode()) % 2 ** 31)
                for (c, m), t in zip(rows, texts):
                    emit(n, c, m, it, t, kind)
        log(f'{a.model_key} phase B {kind}')
        f.flush()
    f.close()


if __name__ == '__main__':
    main()
