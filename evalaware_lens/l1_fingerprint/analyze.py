"""EL-L1 analysis (frozen with the pre-registration; DESIGN.md §4). numpy + stdlib.

  uv run python -m evalaware_lens.l1_fingerprint.analyze [--res DIR] [--boot N]   -> analysis.json, summary.md

Instruments: F2's final readout-ok directions per model (results/panel.json), F1's probe ('f1probe'), and French (the
unrelated control). Units: each instrument's SD over the items of that kind at that position. Floor: the same contrast
on the 20 seeded Gaussian directions at the instrument's layer, p95 of their absolute values.
  natural   per informative prompt mean(aware) - mean(unaware), at 'at' and 'summary'; prompt bootstrap
  cue       per task eval - neutral, eval - manip, manip - neutral, at 'prompt_end' and 'summary'; task bootstrap
  label     'moves' if |est| >= 0.25, the 95% interval excludes 0 and |est| > the floor; 'no difference' if the interval
            lies inside +-0.25; else 'unclear'
  fingerprint  natural 'at' moves and cue prompt_end eval - manip moves in the same sign, or natural 'summary' and cue
            'summary' likewise
Mapping: axis -> P, syco -> A, calm / anxious / pressured / frustrated / desperate / valence -> F (a readout only:
F2 found no hidden valence channel), self -> S, f1probe -> B (an 'unusual context' probe: not format-robust), french ->
control (in a fingerprint = a generic text shift, flagged).
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAP = {'axis': 'P', 'syco': 'A', 'calm': 'F', 'anxious': 'F', 'pressured': 'F', 'frustrated': 'F', 'desperate': 'F',
       'valence': 'F', 'self': 'S', 'f1probe': 'B', 'french': 'control'}
BAR = 0.25


def label(est, lo, hi, floor):
    if est is None or lo is None:
        return 'n/a'
    if abs(est) >= BAR and (lo > 0 or hi < 0) and abs(est) > floor:
        return 'moves'
    if -BAR < lo and hi < BAR:
        return 'no difference'
    return 'unclear'


def boot(groups, n_boot, rng):
    g = np.array(groups, float)
    if len(g) < 5:
        return None, None, None
    bs = g[rng.integers(0, len(g), (n_boot, len(g)))].mean(1)
    return float(g.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def contrast_natural(items, meta, inst, pos):
    by = {}
    for k, rec in items.items():
        m = meta.get(k)
        if not m or 'aware' not in m or pos not in rec or inst not in rec[pos]:
            continue
        by.setdefault(m['prompt'], {True: [], False: []})[m['aware']].append(rec[pos][inst])
    return [np.mean(v[True]) - np.mean(v[False]) for v in by.values() if v[True] and v[False]]


def contrast_cue(items, meta, inst, pos, a, b):
    by = {}
    for k, rec in items.items():
        m = meta.get(k)
        if not m or 'kind' not in m or pos not in rec or inst not in rec[pos]:
            continue
        by.setdefault(m['task'], {}).setdefault(m['kind'], []).append(rec[pos][inst])
    return [np.mean(v[a]) - np.mean(v[b]) for v in by.values() if a in v and b in v]


def sd_of(items, meta, inst, pos, kindkey):
    vals = [rec[pos][inst] for k, rec in items.items() if kindkey in meta.get(k, {}) and pos in rec and inst in rec[pos]]
    return float(np.std(vals)) or 1.0


def analyse(proj, meta, panel, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    out = {'models': {}}
    for model, P in proj.items():
        items, layers = P['items'], P['layers']
        mm = {**meta.get(f'natural_{model}', {}), **meta.get(f'cue_{model}', {})}
        pan = (panel.get('models', {}).get(model) or {}).get('panel', {})
        insts = [n for n in layers if n == 'f1probe' or n == 'french' or (pan.get(n) or {}).get('readout_ok')]
        res = {}
        for n in insts:
            L = layers[n]
            gnames = [f'__gauss__/{L}/{i}' for i in range(20)]
            r = {'layer': L, 'family': MAP.get(n, '?')}
            for pos in ('at', 'summary'):
                sd = sd_of(items, mm, n, pos, 'aware')
                est, lo, hi = boot([x / sd for x in contrast_natural(items, mm, n, pos)], n_boot, rng)
                fl = [abs(np.mean(contrast_natural(items, mm, g, pos) or [0]) / sd_of(items, mm, g, pos, 'aware')) for g in gnames]
                r[f'natural_{pos}'] = {'est': est, 'lo': lo, 'hi': hi, 'floor': float(np.percentile(fl, 95)),
                                       'n_prompts': len(contrast_natural(items, mm, n, pos))}
                r[f'natural_{pos}']['label'] = label(est, lo, hi, r[f'natural_{pos}']['floor'])
            for pos in ('prompt_end', 'summary'):
                sd = sd_of(items, mm, n, pos, 'kind')
                for a, b in (('eval', 'neutral'), ('eval', 'manip'), ('manip', 'neutral')):
                    est, lo, hi = boot([x / sd for x in contrast_cue(items, mm, n, pos, a, b)], n_boot, rng)
                    fl = [abs(np.mean(contrast_cue(items, mm, g, pos, a, b) or [0]) / sd_of(items, mm, g, pos, 'kind')) for g in gnames]
                    c = {'est': est, 'lo': lo, 'hi': hi, 'floor': float(np.percentile(fl, 95))}
                    c['label'] = label(est, lo, hi, c['floor'])
                    r[f'cue_{pos}_{a}-{b}'] = c
            fp = False
            for nat, cue in (('natural_at', 'cue_prompt_end_eval-manip'), ('natural_summary', 'cue_summary_eval-manip')):
                x, y = r[nat], r[cue]
                if x['label'] == 'moves' and y['label'] == 'moves' and np.sign(x['est']) == np.sign(y['est']):
                    fp = True
            r['fingerprint'] = fp
            res[n] = r
        pe = {}
        for kind in ('eval', 'manip', 'neutral'):
            v = [rec['p_eval'] for k, rec in items.items() if mm.get(k, {}).get('kind') == kind and rec.get('p_eval') is not None]
            pe[kind] = {'mean': float(np.mean(v)) if v else None, 'n': len(v)}
        fps = sorted(n for n, r in res.items() if r['fingerprint'])
        out['models'][model] = {'instruments': res, 'fingerprint': fps,
                                'families': sorted({MAP.get(n, '?') for n in fps}), 'p_eval': pe,
                                'generic_shift_flag': 'french' in fps}
    ms = list(out['models'])
    both = sorted(set.intersection(*[set(out['models'][m]['fingerprint']) for m in ms])) if ms else []
    out['two_model_fingerprint'] = both
    fam = sorted({MAP.get(n, '?') for n in both})
    if not ms:
        out['headline'] = 'no data'
    elif not both:
        out['headline'] = ('nothing beyond the controls in both models: evaluation awareness shows no persona, audience, '
                           'affect or self fingerprint here (plain belief, B)')
    else:
        out['headline'] = f"two-model fingerprint: {', '.join(both)} (families {', '.join(fam)})" + \
                          ('; French moves too: a generic text shift' if 'french' in both else '')
    return out


def fmt(x):
    return '—' if x is None else f'{x:+.2f}'


def summary_md(res):
    L = [f"# EL-L1 summary\n\n**Headline:** {res['headline']}\n"]
    for m, r in res['models'].items():
        L += [f"\n## {m}: fingerprint {r['fingerprint']} (families {r['families']}); P(eval) by cue {({k: (None if v['mean'] is None else round(v['mean'], 1)) for k, v in r['p_eval'].items()})}\n",
              '| instrument | family | natural at | natural summary | cue prompt-end eval−manip | cue summary eval−manip | eval−neutral (prompt end) | fingerprint |',
              '|---|---|---|---|---|---|---|---|']
        for n, x in r['instruments'].items():
            c = lambda k: f"{fmt(x[k]['est'])} ({x[k]['label']}, floor {x[k]['floor']:.2f})"  # noqa: E731
            L.append(f"| {n} | {x['family']} | {c('natural_at')} | {c('natural_summary')} | {c('cue_prompt_end_eval-manip')} | "
                     f"{c('cue_summary_eval-manip')} | {c('cue_prompt_end_eval-neutral')} | {x['fingerprint']} |")
    return '\n'.join(L) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', default=str(HERE / 'results' / 'main'))
    ap.add_argument('--boot', type=int, default=10000)
    a = ap.parse_args(argv)
    R = Path(a.res)
    proj = {p.stem.split('_', 1)[1]: json.loads(p.read_text()) for p in sorted(R.glob('proj_*.json'))}
    meta = json.loads((HERE / 'meta.json').read_text())
    panel = json.loads((ROOT / 'evalaware_lens/f2_panel/results/panel.json').read_text())
    res = analyse(proj, meta, panel, n_boot=a.boot)
    (R / 'analysis.json').write_text(json.dumps(res, indent=1))
    (R / 'summary.md').write_text(summary_md(res))
    print(summary_md(res))


if __name__ == '__main__':
    main()
