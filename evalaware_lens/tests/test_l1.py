"""EL-L1 dry runs: the frozen fingerprint rules on planted worlds (natural + cue contrasts, Gaussian floors)."""
import numpy as np

from evalaware_lens.l1_fingerprint import analyze as A


def world(planted=('syco',), cue_gap=1.0, nat_gap=1.0, manip_too=False, seed=0, n_prompts=60, n_tasks=60):
    rng = np.random.default_rng(seed)
    insts = {'syco': 32, 'axis': 50, 'f1probe': 33, 'french': 24}
    items, meta = {}, {'natural_q27': {}, 'cue_q27': {}}

    def rec(shift):
        r = {}
        for n, L in insts.items():
            r[n] = rng.standard_normal() + (shift if n in planted else 0.0)
        for L in set(insts.values()):
            for i in range(20):
                r[f'__gauss__/{L}/{i}'] = rng.standard_normal()
        return r
    for p in range(n_prompts):
        for s in range(4):
            aw = s < 2
            k = f'nat|{p}|{s}'
            items[k] = {'at': rec(nat_gap if aw else 0.0), 'summary': rec(nat_gap if aw else 0.0)}
            meta['natural_q27'][k] = {'prompt': f'P{p}', 'aware': aw}
    for t in range(n_tasks):
        for kind in ('eval', 'manip', 'neutral'):
            g = cue_gap if kind == 'eval' or (manip_too and kind == 'manip') else 0.0
            k = f'cue|{t}|{kind}'
            items[k] = {'prompt_end': rec(g), 'summary': rec(g), 'p_eval': 70 if kind == 'eval' else 30}
            meta['cue_q27'][k] = {'task': t, 'kind': kind}
    proj = {'q27': {'layers': insts, 'items': items}}
    panel = {'models': {'q27': {'panel': {'syco': {'readout_ok': True}, 'axis': {'readout_ok': True}}}}}
    return proj, meta, panel


def test_planted_fingerprint():
    r = A.analyse(*world(), n_boot=400)
    assert r['models']['q27']['fingerprint'] == ['syco'], r['models']['q27']['fingerprint']
    assert r['two_model_fingerprint'] == ['syco'] and 'A' in r['models']['q27']['families']
    assert r['models']['q27']['p_eval']['eval']['mean'] == 70


def test_caution_not_fingerprint():
    r = A.analyse(*world(manip_too=True), n_boot=400)   # moves with eval and with manip alike: not evaluation-specific
    assert r['models']['q27']['fingerprint'] == []


def test_null_world():
    r = A.analyse(*world(planted=()), n_boot=400)
    assert r['models']['q27']['fingerprint'] == [] and r['headline'].startswith('nothing beyond the controls')


def test_french_flags_generic_shift():
    r = A.analyse(*world(planted=('syco', 'french')), n_boot=400)
    assert r['models']['q27']['generic_shift_flag']
