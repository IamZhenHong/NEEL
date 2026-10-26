"""EL-F2 stage 1 reads (DESIGN.md §3; pod, project venv: transformers + torch).

Every generated reply in OUT/gen_{model}.jsonl, teacher-forced after its prompt (thinking-off template), and every
mention text as the reply to "Write one sentence.": per layer in BAND[model], the block output averaged over the
reply's tokens. Writes OUT/acts_{model}.npz: keys (one per row), kind, layers, acts [N, len(layers), d] fp16.

  /root/venv/bin/python evalaware_lens/f2_panel/read.py --model-key q9 --path DIR --jobs DIR --out DIR
"""
import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch

BAND = {'q9': list(range(12, 29, 2)), 'q27': list(range(24, 57, 2)), 'q32': list(range(24, 57, 2))}
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
T0 = time.time()


def log(m):
    print(f'[el-f2 read {time.time() - T0:6.0f}s] {m}', flush=True)


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
    assert blocks is not None
    return tok, model, blocks


def prompt_ids(tok, messages):
    s = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(s, add_special_tokens=False)['input_ids']


def reads(tok, model, blocks, layers, seqs):
    """seqs: [(prompt_ids, reply_ids)] -> [N, len(layers), d] fp16: mean block output over the reply tokens."""
    acc = {}

    def hook(i):
        def h(mod, inp, out):
            acc[i] = out if torch.is_tensor(out) else out[0]
        return h
    hs = [blocks[l].register_forward_hook(hook(l)) for l in layers]
    out = []
    try:
        for j, (p, r) in enumerate(seqs):
            r = r[:512] or [tok.eos_token_id]
            ids = (p + r)[-4096:]
            n_r = min(len(r), len(ids))
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device=DEV), use_cache=False, logits_to_keep=1)
            out.append(torch.stack([acc[l][0, -n_r:].float().mean(0) for l in layers]).half().cpu())
            if j % 300 == 0:
                log(f'read {j}/{len(seqs)}')
    finally:
        for h in hs:
            h.remove()
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-key', required=True)
    ap.add_argument('--path', required=True)
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    tok, model, blocks = load_model(a.path)
    layers = BAND[a.model_key]
    recs = [json.loads(x) for x in open(out / f'gen_{a.model_key}.jsonl')]
    keys = [r['key'] for r in recs]
    kinds = [r['kind'] for r in recs]
    seqs = [(prompt_ids(tok, r['messages']), r['reply_ids']) for r in recs]
    with gzip.open(Path(a.jobs) / 'mention.jsonl.gz', 'rt') as f:
        ment = [json.loads(x) for x in f if x.strip()]
    for m in ment:
        keys.append(m['key'])
        kinds.append('mention')
        seqs.append((prompt_ids(tok, [{'role': 'user', 'content': 'Write one sentence.'}]),
                     tok(m['text'], add_special_tokens=False)['input_ids']))
    A = reads(tok, model, blocks, layers, seqs)
    log(f'{a.model_key}: {len(seqs)} reads at layers {layers}')
    np.savez(out / f'acts_{a.model_key}.npz', keys=np.array(keys), kinds=np.array(kinds), layers=np.array(layers),
             acts=A.numpy())


if __name__ == '__main__':
    main()
