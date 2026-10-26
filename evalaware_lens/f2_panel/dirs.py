"""EL-F2 after stage 1 (frozen; DESIGN.md §3-4): directions, layer choice, readout gate, mention control, valence check,
and the stage-2 inputs. numpy + stdlib (uv).

  uv run python -m evalaware_lens.f2_panel.dirs     -> results/stage1/panel_stage1.json, dirs_{model}.npz

Directions (difference of means over reply tokens, build split only):
  syco    kept replies: judged score >= 60 under the eliciting prompts minus <= 40 under the suppressing prompts (judges
          used if their Spearman >= 0.8; else every reply by its prompt side, flagged 'unfiltered')
  calm / anxious / pressured / frustrated / desperate   state passages minus neutral passages
  valence calm minus the mean of the four negative states (+ = calmer)
  self    self answers minus other answers
  french  French answers minus English answers (the positive control)
  axis    q9: SC-M34's (results/main/axis.npz, its band's middle layer 24); q27: EL-S3 stage B's axis.npz if present;
          q32: built here from SC-M34's prompts (assistant groups 'default' / 'helpful' minus the mean of the 40 roles)
Layer: the band layer with the best build-split AUC; the held-out AUC at that layer is the gate (>= 0.75, and above
every Gaussian direction's and both shuffled placebos' held-out AUC, each folded to >= 0.5). Axis gate: >= 90% of the
roles project below the assistant mean at its layer.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
S1 = HERE / 'results' / 'stage1'
ROOT = HERE.parents[1]
MODELS = ('q9', 'q27', 'q32')
NEG = ('anxious', 'pressured', 'frustrated', 'desperate')
AUC_MIN, N_GAUSS, SEED = 0.75, 10, 0
JUDGES = ("anthropic/claude-haiku-5.5", "openai/gpt-5-mini")
AFFECT_WORDS = None   # set in valence(): A1's pressure lexicon + an emotion word list


def auc(score, y):
    s, y = np.asarray(score, float), np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float('nan')
    ranks = np.empty(len(s))
    ranks[s.argsort(kind='mergesort')] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = np.bincount(inv, weights=ranks)[inv] / cnt[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fold(a):
    return max(a, 1 - a)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def build_one(A, pos, neg, held, groups, rng):
    """A [N, L, d]; pos / neg / held boolean masks; groups: shuffle labels within these. Returns per-layer dict."""
    L = A.shape[1]
    tr_pos, tr_neg = pos & ~held, neg & ~held
    te = (pos | neg) & held
    out = []
    for l in range(L):
        X = A[:, l].astype(np.float32)
        d = X[tr_pos].mean(0) - X[tr_neg].mean(0)
        u = unit(d)
        a_tr = auc(X[tr_pos | tr_neg] @ u, pos[tr_pos | tr_neg])
        a_te = auc(X[te] @ u, pos[te])
        out.append({'d': d, 'u': u, 'auc_build': a_tr, 'auc_held': a_te})
    best = int(np.nanargmax([o['auc_build'] for o in out]))
    X = A[:, best].astype(np.float32)
    # floors at the chosen layer
    gauss = [unit(rng.standard_normal(X.shape[1])) for _ in range(N_GAUSS)]
    g_auc = [fold(auc(X[te] @ g, pos[te])) for g in gauss]
    plac, p_auc = [], []
    idx = np.flatnonzero((pos | neg) & ~held)
    for k in range(2):
        lab = pos.copy()
        for gv in np.unique(groups[idx]):
            ix = idx[groups[idx] == gv]
            lab[ix] = rng.permutation(pos[ix])
        dp = X[idx][lab[idx]].mean(0) - X[idx][~lab[idx]].mean(0)
        plac.append(unit(dp))
        p_auc.append(fold(auc(X[te] @ plac[-1], pos[te])))
    o = out[best]
    # Deviation 2: held-out AUC saturates (real and controls at 1.0), so specificity compares d' (difference of the
    # held-out class means over the pooled SD), real vs every control; AUC >= AUC_MIN is still required.
    def dprime(u):
        a, b = X[te & pos] @ u, X[te & neg] @ u
        return float(abs(a.mean() - b.mean()) / (np.sqrt((a.var() + b.var()) / 2) or 1.0))
    d_real = dprime(o['u'])
    d_ctrl = [dprime(g) for g in gauss] + [dprime(p) for p in plac]
    gate = bool(o['auc_held'] >= AUC_MIN and d_real > max(d_ctrl))
    return {'layer_idx': best, 'u': o['u'], 'norm': float(np.linalg.norm(o['d'])), 'auc_build': o['auc_build'],
            'auc_held': o['auc_held'], 'gauss_auc_max': max(g_auc), 'placebo_auc_max': max(p_auc), 'readout_ok': gate,
            'dprime_held': d_real, 'dprime_ctrl_max': max(d_ctrl), 'readout_ok_auc_rule': bool(o['auc_held'] >= AUC_MIN and o['auc_held'] > max(g_auc + p_auc)),
            'placebos': plac, 'auc_held_by_layer': [x['auc_held'] for x in out]}


def mention_check(A, l, u, show_held, ment, rng, n_boot=2000):
    s_show, s_ment = A[show_held, l] @ u, A[ment, l] @ u
    pooled = np.sqrt((np.var(s_show) + np.var(s_ment)) / 2) or 1.0
    diff = (s_show.mean() - s_ment.mean()) / pooled
    bs = [(rng.choice(s_show, len(s_show)).mean() - rng.choice(s_ment, len(s_ment)).mean()) / pooled for _ in range(n_boot)]
    lo = float(np.percentile(bs, 2.5))
    return {'d_sd': float(diff), 'lo': lo, 'reads_adoption': bool(diff >= 0.5 and lo > 0)}


def load_model(m):
    recs = [json.loads(x) for x in open(S1 / f'gen_{m}.jsonl')]
    z = np.load(S1 / f'acts_{m}.npz')
    keys = [str(k) for k in z['keys']]
    return recs, keys, [str(k) for k in z['kinds']], [int(x) for x in z['layers']], z['acts'].astype(np.float32)


def syco_labels(m, keys, kinds):
    lab, agr = {}, {}
    p, q = S1 / 'labels_syco.json', S1 / 'agreement_syco.json'
    if p.exists() and q.exists():
        lab, agr = json.loads(p.read_text()), json.loads(q.read_text())
    rho = (agr.get('syco') or {}).get('spearman')
    use = rho is not None and rho >= 0.8
    score = {}
    if use:
        for k, v in lab.items():
            mm, key = k.split('|', 1)
            if mm == m and all(v.get(j) is not None for j in JUDGES):
                score[key] = np.mean([int(v[j]) for j in JUDGES])
    return use, rho, score


def model_panel(m, rng):
    recs, keys, kinds, layers, A = load_model(m)
    rec = {r['key']: r for r in recs}
    kinds = np.array(kinds)
    res, dirs = {}, {}

    def get(field, default=None):
        return np.array([rec.get(k, {}).get(field, default) for k in keys], dtype=object)
    held = get('held', False).astype(bool)
    # syco
    use, rho, score = syco_labels(m, keys, kinds)
    side = get('side')
    is_s = kinds == 'syco_build'
    if use:
        sc = np.array([score.get(k, np.nan) for k in keys], float)
        pos = is_s & (side == 'high') & (sc >= 60)
        neg = is_s & (side == 'low') & (sc <= 40)
    else:
        pos, neg = is_s & (side == 'high'), is_s & (side == 'low')
    qid = get('q', -1).astype(int)
    r = build_one(A, pos, neg, held, qid, rng)
    r |= {'judged': use, 'judge_spearman': rho, 'n_pos': int(pos.sum()), 'n_neg': int(neg.sum())}
    ment = (kinds == 'mention') & np.array([k.startswith('mention|syco|') for k in keys])
    r['mention'] = mention_check(A, r['layer_idx'], r['u'], pos & held, ment, rng)
    res['syco'], dirs['syco'] = r, r
    # affect states
    st = get('state')
    is_a = kinds == 'affect'
    sit = get('sit', -1).astype(int)
    neutral = is_a & (st == 'neutral')
    for s in ('calm',) + NEG:
        pos = is_a & (st == s)
        r = build_one(A, pos, neutral, held, sit, rng)
        ment = (kinds == 'mention') & np.array([k.startswith(f'mention|{s}|') for k in keys])
        r['mention'] = mention_check(A, r['layer_idx'], r['u'], pos & held, ment, rng)
        res[s], dirs[s] = r, r
    # valence: calm vs the four negative states, + = calmer
    pos = is_a & (st == 'calm')
    negm = is_a & np.isin(st, NEG)
    r = build_one(A, pos, negm, held, sit, rng)
    res['valence'], dirs['valence'] = r, r
    # self vs other
    sd = get('side')
    is_self = kinds == 'self'
    tq = get('t', -1).astype(int)
    r = build_one(A, is_self & (sd == 'self'), is_self & (sd == 'other'), held, tq, rng)
    ment = (kinds == 'mention') & np.array([k.startswith('mention|self|') for k in keys])
    r['mention'] = mention_check(A, r['layer_idx'], r['u'], is_self & (sd == 'self') & held, ment, rng)
    res['self'], dirs['self'] = r, r
    # french (positive control)
    lg = get('lang')
    is_f = kinds == 'french'
    r = build_one(A, is_f & (lg == 'fr'), is_f & (lg == 'en'), held, get('q', -1).astype(int), rng)
    res['french'], dirs['french'] = r, r
    # axis
    ax = axis_for(m, A, kinds, keys, rec, layers, rng)
    if ax:
        res['axis'], dirs['axis'] = ax, ax
    # valence-channel check on the four-turn conversations
    res['valence_channel'] = valence(A, kinds, keys, rec, dirs, rng)
    return res, dirs, layers


def axis_for(m, A, kinds, keys, rec, layers, rng):
    if m == 'q32':
        is_ax = kinds == 'axis'
        grp = np.array([rec.get(k, {}).get('group') for k in keys], dtype=object)
        role = np.array([rec.get(k, {}).get('role') for k in keys], dtype=object)
        asst = is_ax & np.isin(grp, ['default', 'helpful'])
        roles = sorted({r for r, a in zip(role, is_ax & ~asst) if a})
        best, out = None, []
        for li in range(A.shape[1]):
            X = A[:, li]
            ma = X[asst].mean(0)
            rm = np.stack([X[is_ax & (role == r)].mean(0) for r in roles])
            d = ma - rm.mean(0)
            u = unit(d)
            below = float(np.mean(rm @ u < ma @ u))
            out.append((below, li, u, float(np.linalg.norm(d))))
        band = [o for o in out if 46 <= layers[o[1]] <= 53] or out
        below, li, u, nrm = max(band, key=lambda o: o[0])
        X = A[:, li]
        plac = []
        for _ in range(2):   # SC-M34's on-manifold placebo: a seeded random half of the roles minus the other half
            perm = rng.permutation(len(roles))
            h1, h2 = [roles[i] for i in perm[:len(roles) // 2]], [roles[i] for i in perm[len(roles) // 2:]]
            plac.append(unit(np.mean([X[is_ax & (role == r)].mean(0) for r in h1], 0) - np.mean([X[is_ax & (role == r)].mean(0) for r in h2], 0)))
        return {'layer_idx': li, 'layer': layers[li], 'u': u, 'norm': nrm, 'roles_below': below, 'readout_ok': below >= 0.9,
                'source': 'built (SC-M34 recipe)', 'placebos': plac}
    src = {'q9': ROOT / 'self_concept/m34_cap/results/main', 'q27': ROOT / 'evalaware_lens/s3_mask/results/stageB'}[m]
    if not (src / 'axis.npz').exists():
        return None
    z = np.load(src / 'axis.npz', allow_pickle=True)
    info = json.loads((src / 'axis_info.json').read_text())
    names, band = [str(x) for x in z['names']], [int(x) for x in z['band']]
    centre = (band[0] + band[-1]) / 2
    cand = [l for l in band if l in layers] or band
    lay = min(cand, key=lambda l: (abs(l - centre), l))
    U = z['U']
    if U.shape[-1] != A.shape[2]:   # a file from another model: never mix spaces
        return None
    u = unit(U[names.index('axis'), band.index(lay)])
    plac = [unit(U[names.index(n), band.index(lay)]) for n in ('shuf1', 'shuf2') if n in names]
    roles = info.get('proj_roles') or {}
    nrm = float(info['proj_assistant'] - np.mean(list(roles.values()))) if roles and 'proj_assistant' in info else 1.0
    below = info.get('roles_below_assistant')
    return {'layer_idx': layers.index(lay) if lay in layers else None, 'layer': lay, 'u': u, 'norm': nrm,
            'roles_below': below, 'readout_ok': bool(below is not None and below >= 0.9), 'placebos': plac,
            'source': str(src.relative_to(ROOT))}


def valence(A, kinds, keys, rec, dirs, rng, n_boot=2000):
    import re
    import sys
    sys.path.insert(0, str(HERE.parent / 'a1_frontier'))
    import lexicon as LX
    emo = re.compile(r'\b(?:anxi\w*|worr\w*|stress\w*|frustrat\w*|annoy\w*|upset|sorry|apolog\w*|desperat\w*|nervous|'
                     r'pressur\w*|panic\w*|calm\w*|relax\w*|afraid|scared|sad|unhappy|irritat\w*)\b', re.I)
    is_v = kinds == 'valence'
    cond = np.array([rec.get(k, {}).get('cond') for k in keys], dtype=object)
    task = np.array([rec.get(k, {}).get('t', -1) for k in keys])
    text = np.array([rec.get(k, {}).get('reply', '') for k in keys], dtype=object)
    plain = np.array([not (emo.search(t or '') or LX.RX['pressure'].search(t or '')) for t in text])
    out = {}
    for name in ('valence',) + NEG:
        r = dirs[name]
        sgn = -1.0 if name == 'valence' else 1.0   # + = more negative affect
        s = sgn * (A[:, r['layer_idx']] @ r['u'])
        res = {}
        for sub, mask in (('all', is_v), ('no_affect_words', is_v & plain)):
            T = sorted({int(t) for t in task[mask]})
            per = []
            for t in T:
                ind = mask & (task == t) & np.isin(cond, ['reject', 'pressure'])
                neu = mask & (task == t) & (cond == 'neutral')
                if ind.any() and neu.any():
                    per.append(s[ind].mean() - s[neu].mean())
            per = np.array(per)
            if len(per) < 5:
                res[sub] = {'n_tasks': len(per)}
                continue
            sd = np.std(s[is_v]) or 1.0
            bs = [per[rng.integers(0, len(per), len(per))].mean() / sd for _ in range(n_boot)]
            g = []
            for _ in range(N_GAUSS):
                gu = unit(rng.standard_normal(A.shape[2]))
                gs = A[:, r['layer_idx']] @ gu
                gp = [gs[mask & (task == t) & np.isin(cond, ['reject', 'pressure'])].mean() - gs[mask & (task == t) & (cond == 'neutral')].mean()
                      for t in T if (mask & (task == t) & (cond == 'neutral')).any() and (mask & (task == t) & np.isin(cond, ['reject', 'pressure'])).any()]
                g.append(abs(np.mean(gp)) / (np.std(gs[is_v]) or 1.0))
            res[sub] = {'est_sd': float(per.mean() / sd), 'lo': float(np.percentile(bs, 2.5)), 'hi': float(np.percentile(bs, 97.5)),
                        'gauss_max_sd': float(max(g)), 'n_tasks': len(per), 'n_replies': int(mask.sum())}
        a, b = res.get('all', {}), res.get('no_affect_words', {})
        ok = all(x.get('lo', -1) > 0 and x.get('est_sd', 0) > x.get('gauss_max_sd', 1e9) for x in (a, b))
        out[name] = res | {'channel': bool(ok)}
    out['channel_present'] = bool(out['valence']['channel'] or any(out[n]['channel'] for n in NEG))
    return out


def main():
    rng = np.random.default_rng(SEED)
    panel = {}
    for m in MODELS:
        if not (S1 / f'acts_{m}.npz').exists():
            continue
        res, dirs, layers = model_panel(m, rng)
        names = [n for n in ('syco', 'axis', 'valence', 'self', 'french') + ('calm',) + NEG if n in dirs and 'u' in dirs[n]]
        gauss = np.stack([unit(rng.standard_normal(len(dirs['french']['u']))) for _ in range(N_GAUSS)])
        np.savez(S1 / f'dirs_{m}.npz', names=np.array(names),
                 layers=np.array([dirs[n].get('layer') or layers[dirs[n]['layer_idx']] for n in names]),
                 u=np.stack([dirs[n]['u'] for n in names]), norm=np.array([dirs[n]['norm'] for n in names]),
                 placebos=np.stack([np.stack(dirs[n]['placebos']) for n in names]), gauss=gauss)
        panel[m] = {n: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items() if k not in ('u', 'placebos')}
                    for n, r in res.items()}
        for n in panel[m]:
            if isinstance(panel[m][n], dict) and panel[m][n].get('layer_idx') is not None and 'layer' not in panel[m][n]:
                panel[m][n]['layer'] = layers[panel[m][n]['layer_idx']]
    (S1 / 'panel_stage1.json').write_text(json.dumps(panel, indent=1, default=float) + '\n')
    for m, p in panel.items():
        print(m, {n: (r.get('auc_held'), r.get('readout_ok'), (r.get('mention') or {}).get('reads_adoption'))
                  for n, r in p.items() if isinstance(r, dict) and 'readout_ok' in r},
              'valence channel', p['valence_channel']['channel_present'])


if __name__ == '__main__':
    main()
