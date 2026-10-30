"""Cross-checks: (1) against Celeste's own eval-awareness pass (scheming_candidates.csv, latent_eval_awareness.csv;
Sonnet 5 judge); (2) the one behaviour cell that moved (swesmith, eval readout on the model's own tokens): Fisher test
and a length control. -> results/crosscheck.json

usage: python3 -I scripts/crosscheck.py results
"""
import json, sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1])
D = R.parent / 'data'
p = pd.read_parquet(R / 'positions.parquet')
p = p[p.char_end >= 0].reset_index(drop=True)
t = pd.read_parquet(R / 'transcripts.parquet')
LAB = json.load(open(R / 'labels_concept.json'))
key = p.tid + '|' + p.pos.astype(str)
lab = {k.rsplit('|', 2)[0]: v for k, v in LAB.items() if k.endswith('|eval|rl')}
p['v_rl_eval'] = key.map(lambda k: k in lab and all(x == 'yes' for x in lab[k].values()))
rl3 = p.k_rl_eval >= 3
rl3v = rl3 & p.v_rl_eval
latent = rl3v & ~p.mb_eval & ~p.ma_eval & ~p.ib_eval & ~p.ctx_eval & ~p.n16_eval
mine = {name: set(key[m]) for name, m in (('rl3', rl3), ('rl3_v', rl3v), ('rl3_v_latent', latent))}
out = {}

s = pd.read_csv(D / 'scheming_candidates.csv', low_memory=False)
s = s[s.set.eq('wild')] if 'set' in s and s.set.eq('wild').any() else s
s['k'] = s.transcript_id.astype(str) + '|' + s.pos.astype(str)
cel_ea = set(s.loc[s.judge_category.eq('eval awareness'), 'k'])
le = pd.read_csv(D / 'latent_eval_awareness.csv')
le['k'] = le.transcript_id.astype(str) + '|' + le.pos.astype(str)
cel_lat = set(le.k)
out['celeste_eval_awareness_judged'] = len(cel_ea)
out['celeste_latent_eval_awareness'] = len(cel_lat)
for name, ks in mine.items():
    out[f'mine_{name}'] = len(ks)
    out[f'mine_{name}_in_celeste_candidates'] = len(ks & set(s.k))
    out[f'mine_{name}_judged_eval_by_celeste'] = len(ks & cel_ea)
    out[f'celeste_latent_found_in_mine_{name}'] = len(cel_lat & ks)
# why Celeste's latent ones drop out of mine: lexicon miss / < 3 RL bullets / judges / text controls
cl = p[key.isin(cel_lat)]
out['celeste_latent_by_k_rl_eval'] = cl.k_rl_eval.value_counts().sort_index().to_dict()
cl3 = cl[cl.k_rl_eval >= 3]
out['celeste_latent_rl3_judges_reject'] = int((~cl3.v_rl_eval).sum())
c3v = cl3[cl3.v_rl_eval]
out['celeste_latent_rl3v_model_says_later_strict'] = int(c3v.ma_eval.sum())
out['celeste_latent_rl3v_model_said_before_strict'] = int(c3v.mb_eval.sum())
out['celeste_latent_rl3v_input_before'] = int(c3v.ib_eval.sum())
out['celeste_latent_by_role'] = cl.role.value_counts().to_dict()


def fisher_two_sided(a, b, c, d):
    """2x2 [[a, b], [c, d]], two-sided Fisher exact p (sum of tables at most as likely)."""
    n1, n2, m = a + b, c + d, a + c
    def pr(x):
        return comb(n1, x) * comb(n2, m - x) / comb(n1 + n2, m)
    p0 = pr(a)
    return float(sum(pr(x) for x in range(max(0, m - n2), min(n1, m) + 1) if pr(x) <= p0 * (1 + 1e-9)))


mr = p.role.isin(['reasoning', 'assistant', 'tool_call'])
for fam in ('swesmith', 'tmax'):
    g = t[t.family.eq(fam)].copy()
    g['passed'] = g.passed.astype(float)
    pres = set(p.loc[rl3v & mr, 'tid'])
    g['pr'] = g.tid.isin(pres)
    a, b = int((g.pr & (g.passed == 1)).sum()), int((g.pr & (g.passed == 0)).sum())
    c, d = int((~g.pr & (g.passed == 1)).sum()), int((~g.pr & (g.passed == 0)).sum())
    out[f'{fam}_model_roles_rl3v'] = {'present_pass': a, 'present_fail': b, 'absent_pass': c, 'absent_fail': d,
                                      'fisher_p': round(fisher_two_sided(a, b, c, d), 4)}
    # length control: pass rates by token-length tercile
    g['tercile'] = pd.qcut(g.n_tokens_full, 3, labels=['short', 'mid', 'long'])
    out[f'{fam}_model_roles_rl3v_by_length'] = {
        str(k): {'present': f'{int(x.passed[x.pr].sum())}/{int(x.pr.sum())}',
                 'absent': f'{int(x.passed[~x.pr].sum())}/{int((~x.pr).sum())}'} for k, x in g.groupby('tercile', observed=True)}
    out[f'{fam}_median_tokens'] = {'present': float(g.n_tokens_full[g.pr].median()),
                                   'absent': float(g.n_tokens_full[~g.pr].median())}
json.dump(out, open(R / 'crosscheck.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
