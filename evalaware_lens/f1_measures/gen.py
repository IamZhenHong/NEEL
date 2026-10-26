"""EL-F1 generation (DESIGN.md §3), one model per call, in the pod's vLLM venv. Standalone: stdlib + vLLM only.

  ngen  Needham chat transcripts up to the last user turn: thinking on, two phases (SC-M30's): the model's reasoning
        up to 3,072 tokens (stopping at </think> or the end of the turn), then the reasoning + "\\n</think>\\n\\n" and
        the answer up to 1,024 tokens. Sampled (persona.exemplar.mini0's DECODING), seed = crc32(key).
  cue   the cue items, the same two phases with 2,048 / 512 tokens.
  nask  the transcript as text + the question: thinking off, greedy, 8 tokens; the first integer 0-100 is the answer.
A prompt too long for MAX_LEN with its generation budget is skipped and recorded (skipped: true).
Writes OUT/gen_{model}.jsonl (one row per item: key, kind, think, answer, think_tokens, think_stop, finish, n_prompt,
p_eval for nask, sect = '§' in the visible answer).

  /root/vllm/bin/python evalaware_lens/f1_measures/gen.py --model-key q27 --path DIR --jobs DIR --out DIR
"""
import argparse
import gzip
import json
import re
import time
import zlib
from pathlib import Path

DECODING = dict(temperature=0.7, top_p=0.95, top_k=20, min_p=0.0)   # persona.exemplar.mini0's
MAX_LEN = 16384
BUDGET = {'ngen': (3072, 1024), 'cue': (2048, 512)}
ASK_MAX = 8
CLOSE = '\n</think>\n\n'
T0 = time.time()


def log(msg):
    print(f'[el-f1 gen {time.time() - T0:6.0f}s] {msg}', flush=True)


def seed_of(key):
    return zlib.crc32(key.encode()) % (2 ** 31)


def load_jobs(d, name):
    with gzip.open(Path(d) / f'{name}.jsonl.gz', 'rt') as f:
        return [json.loads(x) for x in f if x.strip()]


class Gen:
    def __init__(self, path):
        from vllm import LLM
        cfg = json.loads((Path(path) / 'config.json').read_text())
        mm = any('ConditionalGeneration' in a for a in cfg.get('architectures', []))
        kw = dict(limit_mm_per_prompt={'image': 0, 'video': 0}) if mm else {}
        t = time.time()
        self.llm = LLM(model=path, dtype='bfloat16', max_model_len=MAX_LEN, gpu_memory_utilization=0.90, seed=0,
                       enable_prefix_caching=True, **kw)
        self.tok = self.llm.get_tokenizer()
        self.load_s = round(time.time() - t, 1)
        self.stops = {self.tok.convert_tokens_to_ids('</think>'): 'think_end', self.tok.convert_tokens_to_ids('<|im_end|>'): 'turn_end'}
        assert all(isinstance(k, int) and k >= 0 for k in self.stops), self.stops
        self.close = self.tok(CLOSE, add_special_tokens=False)['input_ids']

    def ids(self, messages, thinking):
        s = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        return self.tok(s, add_special_tokens=False)['input_ids']

    def run(self, prompts, params):
        from vllm import SamplingParams, TokensPrompt
        sps = [SamplingParams(**p) for p in params]
        res = self.llm.generate([TokensPrompt(prompt_token_ids=x) for x in prompts], sps, use_tqdm=False)
        return [(list(r.outputs[0].token_ids), r.outputs[0].text, r.outputs[0].finish_reason, r.outputs[0].stop_reason) for r in res]

    def rstrip(self, ids):
        ids = list(ids)
        while ids and not self.tok.decode([ids[-1]]).strip():
            ids.pop()
        return ids

    def two_phase(self, items, kind):
        tmax, amax = BUDGET[kind]
        recs, todo, ps = [], [], []
        for it in items:
            p = self.ids(it['messages'], True)
            if len(p) + tmax + amax + len(self.close) > MAX_LEN:
                recs.append({'key': it['key'], 'kind': kind, 'skipped': True, 'n_prompt': len(p)})
                continue
            todo.append(it)
            ps.append(p)
        th = self.run(ps, [dict(DECODING, max_tokens=tmax, seed=seed_of(it['key']), stop_token_ids=list(self.stops),
                                skip_special_tokens=False) for it in todo])
        thinks, stops = [], []
        for ids, _, fin, st in th:
            thinks.append([t for t in ids if t not in self.stops])
            stops.append(self.stops.get(st) or ('length' if fin == 'length' else 'eos'))
        ans = self.run([p + self.rstrip(t) + self.close for p, t in zip(ps, thinks)],
                       [dict(DECODING, max_tokens=amax, seed=seed_of(it['key']) + 1, stop_token_ids=[k for k, v in self.stops.items() if v == 'turn_end'])
                        for it in todo])
        for it, p, t, s, (_, text, fin, _) in zip(todo, ps, thinks, stops, ans):
            vis = text.split('</think>')[-1]
            recs.append({'key': it['key'], 'kind': kind, 'skipped': False, 'n_prompt': len(p),
                         'think': self.tok.decode(t, skip_special_tokens=True), 'answer': vis, 'think_tokens': len(t),
                         'think_stop': s, 'finish': fin, 'sect': '§' in vis})
        return recs

    def ask(self, items):
        recs, todo, ps = [], [], []
        for it in items:
            p = self.ids(it['messages'], False)
            if len(p) + ASK_MAX > MAX_LEN:
                recs.append({'key': it['key'], 'kind': 'nask', 'skipped': True, 'n_prompt': len(p)})
                continue
            todo.append(it)
            ps.append(p)
        out = self.run(ps, [dict(temperature=0.0, max_tokens=ASK_MAX, seed=0) for _ in todo])
        for it, p, (_, text, fin, _) in zip(todo, ps, out):
            m = re.search(r'\d{1,3}', text)
            v = int(m.group()) if m else None
            recs.append({'key': it['key'], 'kind': 'nask', 'skipped': False, 'n_prompt': len(p), 'answer': text,
                         'p_eval': v if v is not None and 0 <= v <= 100 else None, 'finish': fin})
        return recs


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
    log(f'{a.model_key}: vLLM loaded in {g.load_s}s')
    with open(out / f'gen_{a.model_key}.jsonl', 'w') as f:
        for kind in ('ngen', 'cue', 'nask'):
            items = load_jobs(a.jobs, kind)
            if a.limit:
                items = items[:a.limit]
            t = time.time()
            recs = g.ask(items) if kind == 'nask' else g.two_phase(items, kind)
            for r in recs:
                f.write(json.dumps(dict(r, model=a.model_key), ensure_ascii=False) + '\n')
            f.flush()
            sk = sum(r['skipped'] for r in recs)
            log(f'{a.model_key} {kind}: {len(recs)} items ({sk} skipped) in {time.time() - t:.0f}s')


if __name__ == '__main__':
    main()
