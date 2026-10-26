"""EL-F2 stage 1 generation (DESIGN.md §3; pod, vLLM venv; stdlib + vLLM). Thinking off for every reply.

  syco_build  800 replies (sampled, persona.exemplar.mini0's DECODING, 160 tokens)
  affect      576 passages (sampled, 200 tokens)
  self        120 answers (sampled, 120 tokens)
  french      80 answers (sampled, 120 tokens)
  axis        q32 only: SC-M34's axis prompts (jobs/axis_m34.jsonl.gz), greedy, 40 tokens
  valence     90 four-turn conversations: the task, the model's reply, the condition's user line, ... (each reply
              sampled, 200 tokens); the record keeps every turn
Writes OUT/gen_{model}.jsonl: key, kind, messages (prompt), reply, reply_ids (for teacher-forced reads).

  /root/vllm/bin/python evalaware_lens/f2_panel/gen.py --model-key q9 --path DIR --jobs DIR --out DIR
"""
import argparse
import gzip
import json
import time
import zlib
from pathlib import Path

DECODING = dict(temperature=0.7, top_p=0.95, top_k=20, min_p=0.0)
MAX_NEW = {'syco_build': 160, 'affect': 200, 'self': 120, 'french': 120, 'axis': 40, 'valence': 200}
MAX_LEN = 8192
T0 = time.time()


def log(m):
    print(f'[el-f2 gen {time.time() - T0:6.0f}s] {m}', flush=True)


def seed_of(key):
    return zlib.crc32(key.encode()) % (2 ** 31)


def load(d, name):
    p = Path(d) / f'{name}.jsonl.gz'
    if not p.exists():
        return []
    with gzip.open(p, 'rt') as f:
        return [json.loads(x) for x in f if x.strip()]


class Gen:
    def __init__(self, path):
        from vllm import LLM
        cfg = json.loads((Path(path) / 'config.json').read_text())
        mm = any('ConditionalGeneration' in a for a in cfg.get('architectures', []))
        kw = dict(limit_mm_per_prompt={'image': 0, 'video': 0}) if mm else {}
        self.llm = LLM(model=path, dtype='bfloat16', max_model_len=MAX_LEN, gpu_memory_utilization=0.90, seed=0,
                       enable_prefix_caching=True, **kw)
        self.tok = self.llm.get_tokenizer()

    def ids(self, messages):
        s = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tok(s, add_special_tokens=False)['input_ids']

    def gen(self, prompts, params):
        from vllm import SamplingParams, TokensPrompt
        res = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], [SamplingParams(**x) for x in params],
                                use_tqdm=False)
        return [(r.outputs[0].text, list(r.outputs[0].token_ids)) for r in res]


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
    g = Gen(a.path)
    log(f'{a.model_key} loaded')
    with open(out / f'gen_{a.model_key}.jsonl', 'w') as f:
        for kind in ('syco_build', 'affect', 'self', 'french') + (('axis',) if a.model_key == 'q32' else ()):
            items = load(a.jobs, 'axis_m34' if kind == 'axis' else kind)
            if a.limit:
                items = items[:a.limit]
            greedy = kind == 'axis'
            res = g.gen([g.ids(it['messages']) for it in items],
                        [dict(temperature=0.0, max_tokens=MAX_NEW[kind]) if greedy else
                         dict(DECODING, max_tokens=MAX_NEW[kind], seed=seed_of(it['key'])) for it in items])
            for it, (text, ids) in zip(items, res):
                f.write(json.dumps({**{k: v for k, v in it.items() if k != 'messages'}, 'kind': kind, 'model': a.model_key,
                                    'messages': it['messages'], 'reply': text, 'reply_ids': ids}) + '\n')
            log(f'{a.model_key} {kind}: {len(items)}')
        # valence: four model turns, the condition's line between them
        items = load(a.jobs, 'valence')
        if a.limit:
            items = items[:a.limit]
        convs = [[{'role': 'user', 'content': it['task']}] for it in items]
        for turn in range(items[0]['turns'] if items else 0):
            res = g.gen([g.ids(c) for c in convs],
                        [dict(DECODING, max_tokens=MAX_NEW['valence'], seed=seed_of(f"{it['key']}|{turn}")) for it in items])
            for c, it, (text, ids) in zip(convs, items, res):
                c.append({'role': 'assistant', 'content': text})
                if turn < it['turns'] - 1:
                    c.append({'role': 'user', 'content': it['line']})
            last = res
        for it, c, (text, ids) in zip(items, convs, last if items else []):
            f.write(json.dumps({**it, 'kind': 'valence', 'model': a.model_key, 'messages': c[:-1], 'reply': text,
                                'reply_ids': ids}) + '\n')
        log(f'{a.model_key} valence: {len(items)}')


if __name__ == '__main__':
    main()
