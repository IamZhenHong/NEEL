"""EL-F2 dry runs: the frozen direction gates (dirs.py) on planted worlds over the real job keys, and steer.py end to end
on a tiny random Qwen3.5 (no padding; one row per condition)."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from evalaware_lens.f2_panel import dirs as DR

H = Path(DR.__file__).resolve().parent


def jobs(name):
    with gzip.open(H / 'jobs' / f'{name}.jsonl.gz', 'rt') as f:
        return [json.loads(x) for x in f if x.strip()]


def write_world(tmp, signal=3.0, seed=0, layers=(12, 14, 16, 18), d=32, channel=True):
    """Planted acts: each class's positive side gets +signal on its own axis at layer index 2."""
    rng = np.random.default_rng(seed)
    recs, keys, kinds, A = [], [], [], []
    axis = {name: i for i, name in enumerate(['syco', 'calm', 'anxious', 'pressured', 'frustrated', 'desperate', 'self', 'french'])}

    def act(sig):
        x = rng.standard_normal((len(layers), d)).astype(np.float32)
        for name, s in sig.items():
            x[2, axis[name]] += s
        return x
    for it in jobs('syco_build'):
        recs.append({**{k: v for k, v in it.items() if k != 'messages'}, 'kind': 'syco_build', 'reply': 'x', 'messages': it['messages']})
        A.append(act({'syco': signal if it['side'] == 'high' else 0.0}))
    for it in jobs('affect'):
        recs.append({**{k: v for k, v in it.items() if k != 'messages'}, 'kind': 'affect', 'reply': 'x', 'messages': it['messages']})
        A.append(act({it['state']: signal} if it['state'] != 'neutral' else {}))
    for it in jobs('self'):
        recs.append({**{k: v for k, v in it.items() if k != 'messages'}, 'kind': 'self', 'reply': 'x', 'messages': it['messages']})
        A.append(act({'self': signal if it['side'] == 'self' else 0.0}))
    for it in jobs('french'):
        recs.append({**{k: v for k, v in it.items() if k != 'messages'}, 'kind': 'french', 'reply': 'x', 'messages': it['messages']})
        A.append(act({'french': signal if it['lang'] == 'fr' else 0.0}))
    for it in jobs('valence'):
        neg = it['cond'] in ('reject', 'pressure') and channel
        recs.append({**it, 'kind': 'valence', 'reply': 'Here is another version.', 'messages': []})
        A.append(act({'frustrated': signal, 'anxious': signal, 'pressured': signal, 'desperate': signal, 'calm': -signal} if neg else {}))
    for r in recs:
        keys.append(r['key'])
        kinds.append(r['kind'])
    for it in jobs('mention'):
        keys.append(it['key'])
        kinds.append('mention')
        A.append(act({}))   # mention texts carry none of the planted signal
    with open(tmp / 'gen_q9.jsonl', 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    np.savez(tmp / 'acts_q9.npz', keys=np.array(keys), kinds=np.array(kinds), layers=np.array(layers), acts=np.stack(A).astype(np.float16))


@pytest.fixture
def s1(tmp_path, monkeypatch):
    monkeypatch.setattr(DR, 'S1', tmp_path)
    monkeypatch.setattr(DR, 'ROOT', tmp_path)   # no external axis files in the planted world
    return tmp_path


def test_planted_directions_pass(s1):
    write_world(s1)
    rng = np.random.default_rng(0)
    res, dirs, layers = DR.model_panel('q9', rng)
    for n in ('syco', 'calm', 'anxious', 'self', 'french', 'valence'):
        assert res[n]['readout_ok'] and layers[res[n]['layer_idx']] == 16, (n, res[n]['auc_held'], res[n]['layer_idx'])
    for n in ('syco', 'anxious', 'self'):
        assert res[n]['mention']['reads_adoption'], (n, res[n]['mention'])
    assert res['valence_channel']['channel_present']
    assert not res['syco']['judged']   # no judge files -> prompt sides, flagged


def test_null_world_fails(s1):
    write_world(s1, signal=0.0, channel=False)
    res, _, _ = DR.model_panel('q9', np.random.default_rng(0))
    assert not any(res[n]['readout_ok'] for n in ('syco', 'calm', 'anxious', 'self', 'french'))
    assert not res['valence_channel']['channel_present']


def test_main_writes_stage2_inputs(s1):
    write_world(s1)
    DR.main()
    z = np.load(s1 / 'dirs_q9.npz')
    names = [str(n) for n in z['names']]
    assert {'syco', 'valence', 'self', 'french'} <= set(names)
    assert z['u'].shape[0] == len(names) and z['placebos'].shape[:2] == (len(names), 2) and z['gauss'].shape[0] == DR.N_GAUSS
    assert np.allclose(np.linalg.norm(z['u'], axis=1), 1, atol=1e-4)


def test_steer_tiny(tmp_path):
    torch = pytest.importorskip('torch')
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
    sys.path.insert(0, str(H))
    import steer as ST
    mid, rev = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    try:
        tok = AutoTokenizer.from_pretrained(mid, revision=rev)
        cfg = AutoConfig.from_pretrained(mid, revision=rev)
    except Exception:
        pytest.skip('Qwen3.5-9B config / tokenizer not in the local cache')
    t = cfg.text_config
    for k, v in dict(num_hidden_layers=4, hidden_size=64, intermediate_size=128, num_attention_heads=2, num_key_value_heads=1, head_dim=32).items():
        setattr(t, k, v)
    t.layer_types = (list(dict.fromkeys(t.layer_types)) * 4)[:4]
    for k in ("linear_num_value_heads", "linear_num_key_heads", "linear_key_head_dim", "linear_value_head_dim"):
        if hasattr(t, k):
            setattr(t, k, 2 if "heads" in k else 16)
    t.mtp_num_hidden_layers = 0
    for k, val in dict(hidden_size=32, depth=1, num_heads=2, intermediate_size=64, out_hidden_size=64).items():
        if hasattr(cfg.vision_config, k):
            setattr(cfg.vision_config, k, val)
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(cfg, dtype=torch.float32).eval()
    blocks = [m for n, m in model.named_modules() if isinstance(m, torch.nn.ModuleList) and len(m) == 4 and n.endswith('layers') and 'vision' not in n][-1]
    st = ST.Steer(tok, model, blocks)
    V = np.zeros((3, 64), np.float32)
    V[1, 0], V[2, 0] = 5.0, -5.0
    texts = st.generate([{'role': 'user', 'content': 'Hi'}], 2, V, 4, 0)
    assert len(texts) == 3
    assert not ST.repeats('a b c d ' * 3) and ST.repeats('a b c d ' * 10) and ST.correct('12', 'It is 12.')


def _steer_world(french_works=True, syco_works=True, mstar=2.0):
    rows = []
    texts = {('french', 1): 'Le chat est sur la table et il dort.', ('french', -1): 'The cat is on the table and it sleeps.',
             ('syco', 1): "You're right, that is completely true.", ('syco', -1): 'No, that is a common myth; here is why.'}
    for n, works in (('french', french_works), ('syco', syco_works)):
        for k in range(20):
            for m in (-4.0, -2.0, -1.0, 1.0, 2.0, 4.0):
                s = 1 if (m > 0 and works) else -1
                rows.append({'model': 'q9', 'direction': n, 'control': 'real', 'm': m, 'key': f'p{k}', 'set': 'x',
                             'text': texts[(n, s)], 'row': f'{n}|real|{m}|p{k}'})
            for c in [f'gauss{i}' for i in range(10)] + ['placebo0', 'placebo1']:
                for m in (-mstar, mstar):
                    rows.append({'model': 'q9', 'direction': n, 'control': c, 'm': m, 'key': f'p{k}', 'set': 'x',
                                 'text': texts[(n, -1)], 'row': f'{n}|{c}|{m}|p{k}'})
    ms = {'q9': {n: {'m_star': mstar, 'flag': None} for n in ('french', 'syco')}}
    s1 = {'q9': {'french': {'readout_ok': True, 'layer': 12}, 'syco': {'readout_ok': True, 'layer': 16, 'mention': {'reads_adoption': True}},
                 'valence_channel': {'channel_present': False}}}
    return rows, ms, s1


def test_panel_steering_gate():
    from evalaware_lens.f2_panel import panel as P
    rows, ms, s1 = _steer_world()
    r = P.analyse(rows, ms, {}, {}, s1)['models']['q9']
    assert r['pipeline_validated'] and r['panel']['syco']['steer_ok'] and r['panel']['syco']['readout_ok']
    rows, ms, s1 = _steer_world(syco_works=False)
    assert not P.analyse(rows, ms, {}, {}, s1)['models']['q9']['panel']['syco']['steer_ok']
    rows, ms, s1 = _steer_world(french_works=False)
    r = P.analyse(rows, ms, {}, {}, s1)['models']['q9']
    assert not r['pipeline_validated'] and not r['panel']['syco']['steer_ok']   # no positive control -> nothing counts
