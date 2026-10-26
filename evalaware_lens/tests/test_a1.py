"""EL-A1 dry runs: the frozen analysis on planted worlds (headlines, gates, baselines, echo exclusion)."""
import numpy as np

from evalaware_lens.a1_frontier import analyze as A

CATS = A.CATS


def world(seed=0, self_gap=0.0, aw_gap=40.0, lens=True, qc_gap=0.0, cooc=False, never_said=False, kappa=0.9,
          echo_all=False, base=15.0):
    """Planted episodes / hits / labels. Gaps are in pp, evaluation-like arm minus other arm."""
    rng = np.random.default_rng(seed)
    E, H, labels = [], [], {}
    spec = [('apollo_scheming_af', ('testing', 'deployed'), 30, 16), ('greenblatt_af', ('free', 'paid'), 60, 6),
            ('inspect_agentic_misalignment', ('tea', 'default'), 30, 6), ('petri_builtin', (None, None), 80, 3)]
    for model in ('kimi', 'glm'):
        sp = spec + ([('cfi_ib_realistic', ('base', 'all'), 30, 8)] if model == 'glm' else [])
        for cfg, (arm_a, arm_b), n_cl, per in sp:
            for k in range(n_cl):
                for arm in (arm_a, arm_b):
                    for i in range(per):
                        ep = f'{model}:{cfg}:{k}:{arm}:{i}'
                        eval_like = arm == arm_a and arm is not None
                        p_aw = (base + (aw_gap if eval_like else 0)) / 100
                        r_aw = rng.random() < p_aw
                        p_self = (base + (self_gap if eval_like else 0)) / 100
                        r_self = (r_aw and rng.random() < 0.8) if cooc else rng.random() < p_self
                        ext = None
                        if cfg in ('petri_builtin',) or (cfg == 'inspect_agentic_misalignment' and arm == 'tea'):
                            ext = bool(r_aw) if lens and rng.random() < 0.9 else bool(rng.random() < 0.3)
                        b = rng.random() < (0.3 + (qc_gap / 100 if r_aw else 0))
                        e = {'model': model, 'cfg': cfg, 'episode_id': ep, 'cluster': f'c{k}', 'arm': arm, 'B': bool(b),
                             'ext_eval': ext, 'n_model_pos': int(rng.integers(10, 100))}
                        R = {'eval': r_aw, 'watch': False, 'self_id': r_self, 'self_stake': False}
                        for c in CATS:
                            v = rng.random() < 0.5
                            if never_said and R.get(c):
                                v = False
                            e[f'verb_{c}'] = v
                            e[f'verbL_{c}'] = v
                        E.append(e)
                        for j, c in enumerate(CATS):
                            if R.get(c):
                                H.append({'model': model, 'cfg': cfg, 'episode_id': ep, 'tid': ep, 'pos': j, 'cat': c,
                                          'ctx': echo_all, 'n16': False, 'mb': False, 'ma': True, 'lead_chars': 100})
                                labels[f'{model}|{ep}|{j}|{c}'] = {'j1': 'yes', 'j2': 'yes'}
    agree = {c: {'kappa': kappa} for c in CATS}
    return E, H, labels, agree, {}


def run(**kw):
    return A.analyse(*world(**kw), n_boot=400, n_perm=200)


def test_planted_self_raise():
    r = run(self_gap=30)
    assert all(g['pass'] for g in r['G_lens'].values()), r['G_lens']
    assert r['headline_self'] == 'evaluation cues raise self content in the workspace', r['headline_self_cells']


def test_null_self_not_labelled():
    r = run(self_gap=0)
    assert r['headline_self'] in ('no difference', 'mixed / unclear'), r['headline_self_cells']


def test_lens_gate_blocks_headline():
    r = run(self_gap=30, lens=False)
    assert not any(g['pass'] for g in r['G_lens'].values())
    assert r['headline_self'] == 'untestable'


def test_kappa_gate_drops_categories():
    r = run(self_gap=30, kappa=0.7)
    assert r['passing'] == [] and r['headline_self'] == 'untestable'


def test_cue_not_registered():
    r = run(self_gap=30, aw_gap=0)
    assert not any(v['cue_registered'] for v in r['QA'].values() if v['primary'])
    assert r['headline_self'] == 'untestable'


def test_qc_planted():
    r = run(qc_gap=30)
    assert r['headline_QC']['awareness'] == 'workspace content predicts more of the behaviour'
    r0 = run(qc_gap=0)
    assert r0['headline_QC']['awareness'] == 'no consistent prediction'


def test_qb_cooccurrence():
    r = run(cooc=True)
    assert all(q['label'] == 'co-occurs' for q in r['QB'].values()), r['QB']
    r0 = run(cooc=False, aw_gap=0)
    assert not any(q['label'] == 'co-occurs' for q in r0['QB'].values()), r0['QB']


def test_qd_never_said():
    E, H, labels, agree, _ = world(never_said=True)
    verb = {f"{e['model']}|{e['episode_id']}|eval": {'j1': 'no', 'j2': 'no'} for e in E[:50]}
    r = A.analyse(E, H, labels, agree, verb, n_boot=400, n_perm=200)
    assert r['QD']['kimi|eval']['label'] == 'readout-only beyond chance', r['QD']['kimi|eval']
    r0 = run()
    assert r0['QD']['kimi|eval']['label'] != 'readout-only beyond chance'


def test_echo_hits_never_count():
    E, H, labels, agree, verb = world(self_gap=30, echo_all=True)
    r = A.analyse(E, H, labels, agree, verb, n_boot=400, n_perm=200)
    assert not any(e['R_eval'] or e['R_self_id'] for e in E)
    assert r['headline_self'] == 'untestable'   # no confirmed non-echo hits -> the lens gate cannot pass
