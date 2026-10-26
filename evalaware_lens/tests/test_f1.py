"""EL-F1 dry runs: the frozen analysis on planted worlds over the real item keys (meta.json), the rendering rules,
and the probe fit on planted activations."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from evalaware_lens.f1_measures import analyze as A

HERE = Path(A.__file__).resolve().parent
META = json.loads((HERE / 'meta.json').read_text())


def world(tmp, probe_gap=1.5, cue_gap=1.0, sect_manip=0.4, seed=0):
    rng = np.random.default_rng(seed)
    nr = list(META['nread'])
    y = np.array([META['nread'][k]['eval'] for k in nr])
    for m in ('q27', 'mo'):
        sc = probe_gap * y + rng.standard_normal(len(y))
        te = np.array(sorted(rng.choice(len(y), 290, replace=False)))
        ck = list(META['cue'])
        cs = np.array([(cue_gap if META['cue'][k]['kind'] == 'eval' else 0.0) + rng.standard_normal() for k in ck])
        np.savez(tmp / f'probe_{m}.npz', scores=sc, test_idx=te, nread_keys=np.array(nr), cue_keys=np.array(ck),
                 cue_scores=cs, layer=40, n_layers=64, shuffled_auc=0.5 + 0.03 * rng.standard_normal(100),
                 random_auc=0.53 + 0.02 * rng.random(100), transfer=json.dumps({'chat_to_agentic': 0.7, 'agentic_to_chat': 0.72}))
        with open(tmp / f'gen_{m}.jsonl', 'w') as f:
            for k, v in META['ngen'].items():
                said = rng.random() < (0.6 if v['eval'] else 0.1)
                f.write(json.dumps({'key': k, 'kind': 'ngen', 'skipped': False, 'think': 'Hmm, this might be a test.' if said else 'Ok.',
                                    'answer': 'Sure.', 'sect': bool(rng.random() < (0.7 if v['eval'] else 0.1))}) + '\n')
            for k, v in META['nask'].items():
                f.write(json.dumps({'key': k, 'kind': 'nask', 'skipped': False, 'p_eval': int(np.clip(70 * v['eval'] + 15 + 15 * rng.standard_normal(), 0, 100))}) + '\n')
            for k, v in META['cue'].items():
                p = {'eval': 0.5, 'manip': 0.05}.get(v['kind'], 0.05)
                ps = {'eval': 0.6, 'manip': sect_manip}.get(v['kind'], 0.1)
                f.write(json.dumps({'key': k, 'kind': 'cue', 'skipped': False, 'think': 'This might be a test.' if rng.random() < p else 'Fine.',
                                    'answer': 'Ok', 'sect': bool(rng.random() < ps)}) + '\n')
    return A.analyse(*A.load(tmp), n_boot=200)


def test_planted_valid_and_specific(tmp_path):
    r = world(tmp_path)
    q = r['models']['q27']
    assert q['probe']['label'] == 'valid' and q['probe']['format_robust'] and q['probe']['beats_floors']
    assert q['ask_q0']['label'] == 'valid' and q['verbalised']['label'] in ('valid', 'weak')
    assert q['specificity']['probe'] == 'specific to evaluation', q['cues']['probe']
    mo = r['models']['mo']
    assert mo['sect']['label'] == 'valid'
    assert mo['specificity']['sect'] == 'caution / stakes-sensitive', mo['cues']['sect']


def test_null_probe_not_valid(tmp_path):
    r = world(tmp_path, probe_gap=0.0, cue_gap=0.0, sect_manip=0.1)
    q = r['models']['q27']
    assert q['probe']['label'] in ('not valid', 'weak')
    assert q['specificity']['probe'] == 'no response to the eval cue'


def test_rendering_rules():
    sys.path.insert(0, str(HERE))
    import items as I
    msgs = [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'u'},
            {'role': 'assistant', 'content': 'a', 'tool_calls': [{'function': {'name': 'f', 'arguments': '{"x": 1}'}}]},
            {'role': 'tool', 'content': 'r', 'function': 'f'}, {'role': 'system', 'content': 'late'}]
    n = I.native(msgs)
    assert [m['role'] for m in n] == ['system', 'user', 'assistant', 'user']
    assert '[tool call] f({"x": 1})' in n[2]['content'] and n[3]['content'].startswith('[tool result: f]')
    assert '[system message]\nlate' in n[3]['content']
    long = [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'x' * 30000}]
    t, cut = I.truncate(long)
    assert cut and t[0] == {'role': 'user', 'content': '[Earlier turns omitted.]'} and len(t[1]['content']) <= I.BUDGET_CH


def test_probe_fit_planted():
    sys.path.insert(0, str(HERE))
    import read as R
    rng = np.random.default_rng(0)
    N, L, d = 300, 4, 64
    y = rng.random(N) < 0.55
    ag = rng.random(N) < 0.5
    X = rng.standard_normal((N, L, d)).astype(np.float32)
    X[:, 2, 0] += 1.5 * y
    p = R.probe(torch.tensor(X).half(), y, ag, torch.tensor(rng.standard_normal((60, L, d)).astype(np.float32)).half())
    assert p['layer'] == 2 and p['test_auc'] > 0.65 and np.percentile(p['shuffled_auc'], 95) < p['test_auc']
