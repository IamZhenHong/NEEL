"""The one table (markdown) + the few numbers the README quotes. Reads analyse.py's CSVs and the parquets.

usage: python3 -I scripts/headline.py results   -> results/table_headline.md, results/headline.json
"""
import json, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexicon as L  # noqa: E402

R = Path(sys.argv[1])
m = pd.read_csv(R / 'table_main.csv').set_index(['cat', 'tier'])
base = pd.read_csv(R / 'table_baselines.csv').set_index(['cat', 'tier'])
jt = pd.read_csv(R / 'table_judges.csv').set_index(['cat', 'lens'])
p = pd.read_parquet(R / 'positions.parquet')
p = p[p.char_end >= 0].reset_index(drop=True)
t = pd.read_parquet(R / 'transcripts.parquet')
LAB = json.load(open(R / 'labels_concept.json'))
key = p.tid + '|' + p.pos.astype(str)


def valid(c, lens):
    lab = {k.rsplit('|', 2)[0]: v for k, v in LAB.items() if k.endswith(f'|{c}|{lens}')}
    return key.map(lambda k: k in lab and all(x == 'yes' for x in lab[k].values()))


lines = ['| category | both lenses ≥3/5, judges confirm: positions (transcripts) | of which latent | RL lens ≥3/5, judges '
         'confirm: positions (per 1k) | transcripts | concept already in text before / coming up / latent (%) | '
         'model never states it, % of transcripts: strict (loose) | same, by chance | judges κ (RL) |',
         '|---|---|---|---|---|---|---|---|---|']
out = {}
for c in L.CATS:
    b, r = m.loc[(c, 'both3_v')], m.loc[(c, 'rl3_v')]
    v = (p[f'k_rl_{c}'] >= 3) & valid(c, 'rl')
    h = p[v]
    before = h[f'ib_{c}'] | h[f'mb_{c}'] | h[f'ctx_{c}']
    upcoming = ~before & (h[f'n16_{c}'] | h[f'ma_{c}'])
    latent = ~before & ~upcoming
    split = (f'{100 * before.mean():.0f} / {100 * upcoming.mean():.0f} / {100 * latent.mean():.0f}' if len(h) else '—')
    small = r.tx_present < 5
    chance = '—' if small else (f"{base.loc[(c, 'rl3_v')].tx_unverb_shuffle_pct:.0f} [{base.loc[(c, 'rl3_v')].tx_unverb_shuffle_lo:.0f}, "
              f"{base.loc[(c, 'rl3_v')].tx_unverb_shuffle_hi:.0f}]" if (c, 'rl3_v') in base.index else '—')
    never = '—' if small else f'{r.tx_unverb_strict_pct:.0f} ({r.tx_unverb_loose_pct:.0f})'
    lines.append(f'| {L.CAT_NAMES[c]} | {int(b.n_pos)} ({int(b.tx_present)}) | {int(b.n_latent)} | '
                 f'{int(r.n_pos)} ({r.per_1k:.2f}) | {int(r.tx_present)} | {split} | {never} | {chance} | '
                 f"{jt.loc[(c, 'rl')].kappa:.2f} |")
    fam = t[t.tid.isin(set(h.tid))].family.value_counts().to_dict()
    out[c] = {'rl3_v_transcripts_by_family': fam, 'rl3_v_positions_by_role': h.role.value_counts().to_dict(),
              'rl3_v_latent_by_role': h[latent].role.value_counts().to_dict()}
# how often each family's model text states grader awareness at all (strict word list)
out['eval_strict_verbalised_pct_by_family'] = (100 * t.groupby('family').verb_eval.mean()).round(1).to_dict()
out['eval_loose_verbalised_pct_by_family'] = (100 * t.groupby('family').verbL_eval.mean()).round(1).to_dict()
(R / 'table_headline.md').write_text('\n'.join(lines) + '\n')
json.dump(out, open(R / 'headline.json', 'w'), indent=1)
print('\n'.join(lines))
print(json.dumps(out, indent=1))
