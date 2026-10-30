"""Stage 3: the one figure (exploratory). Reads results/positions.parquet + labels via analyse.py's tiers.

(a) where the concept of each robust, judge-confirmed readout sits in the text: already in the text before the read
    token (input or the model's own earlier text), coming up (next 16 tokens or the model says it later), or nowhere
    (latent: never in the text before, not in the next 16 tokens, never said by the model).
(b) per transcript: % of transcripts with an internal readout whose model never states the concept (strict word list;
    the bar runs to the loose word list), against the same rate in transcripts with no readout.

usage: python3 -I scripts/figure.py results
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexicon as L  # noqa: E402

R = Path(sys.argv[1])
p = pd.read_parquet(R / 'positions.parquet')
p = p[p.char_end >= 0].reset_index(drop=True)
t = pd.read_parquet(R / 'transcripts.parquet')
LAB = json.load(open(R / 'labels_concept.json'))
key = p.tid + '|' + p.pos.astype(str)
for c in L.CATS:
    for lens in ('rl', 'sft'):
        lab = {k.rsplit('|', 2)[0]: v for k, v in LAB.items() if k.endswith(f'|{c}|{lens}')}
        p[f'v_{lens}_{c}'] = key.map(lambda k: k in lab and all(x == 'yes' for x in lab[k].values()))

TIER = {'RL lens only (≥3/5, judges confirm)': lambda c: (p[f'k_rl_{c}'] >= 3) & p[f'v_rl_{c}'],
        'Both lenses (≥3/5 each, judges confirm)': lambda c: (p[f'k_rl_{c}'] >= 3) & (p[f'k_sft_{c}'] >= 3)
        & p[f'v_rl_{c}'] & p[f'v_sft_{c}']}
COL = {'before': '#2a78d6', 'upcoming': '#eb6834', 'latent': '#1baf7a'}  # reference categorical slots 1-3
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={'width_ratios': [1.25, 1]})
labels, ypos, y = [], [], 0
for c in L.CATS:
    for ti, (tname, f) in enumerate(TIER.items()):
        h = p[f(c)]
        before = h[f'ib_{c}'] | h[f'mb_{c}'] | h[f'ctx_{c}']
        upcoming = ~before & (h[f'n16_{c}'] | h[f'ma_{c}'])
        latent = ~before & ~upcoming
        n = len(h)
        left = 0
        for k, part in (('before', before), ('upcoming', upcoming), ('latent', latent)):
            w = 100 * part.mean() if n else 0
            ax.barh(y, w, left=left, height=0.72, color=COL[k], edgecolor=SURF, linewidth=2,
                    hatch='////' if ti == 1 else None)
            left += w
        ax.text(101, y, f'n={n}' + (f'  latent {int(latent.sum())}' if n else ''), va='center', fontsize=8, color=INK2)
        labels.append(f'{L.CAT_NAMES[c]}' if ti == 0 else '  both lenses')
        ypos.append(y)
        y += 1
    y += 0.5
ax.set_yticks(ypos, labels)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel('% of robust readouts')
ax.set_title('(a) Where the readout\'s concept is in the text', loc='left', fontsize=10)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.xaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color=COL['before'], label='already in text before'),
                   Patch(color=COL['upcoming'], label='coming up (next 16 / model says later)'),
                   Patch(color=COL['latent'], label='nowhere (latent)'),
                   Patch(facecolor='white', edgecolor=INK2, hatch='////', label='both-lens tier')],
          loc='upper center', bbox_to_anchor=(0.45, -0.13), ncol=2, frameon=False, fontsize=8)

# (b) transcript level, RL tier; comparator = presence shuffled within family x thinking (analyse.py, 1,000 draws)
f = TIER['RL lens only (≥3/5, judges confirm)']
base = pd.read_csv(R / 'table_baselines.csv').set_index(['cat', 'tier'])
cats_b = []
for c in L.CATS:
    pres = set(p.loc[f(c), 'tid'])
    a = t[t.tid.isin(pres)]
    if len(a) < 5:  # training awareness: one transcript
        continue
    i = len(cats_b)
    cats_b.append(c)
    s_, lo = 100 * (~a[f'verb_{c}']).mean(), 100 * (~a[f'verbL_{c}']).mean()
    nb = base.loc[(c, 'rl3_v')]
    bx.plot([nb.tx_unverb_shuffle_lo, nb.tx_unverb_shuffle_hi], [i + 0.22] * 2, color=INK2, lw=1.5)
    bx.plot(nb.tx_unverb_shuffle_pct, i + 0.22, 'D', ms=6, color=INK2, mec=SURF, mew=1.5)
    bx.plot([lo, s_], [i, i], color=COL['latent'], lw=2, solid_capstyle='round')
    bx.plot(s_, i, 'o', ms=8, color=COL['latent'], mec=SURF, mew=2)
    bx.text(103, i + 0.1, f'{len(a)} tx', va='center', fontsize=8, color=INK2)
yb = np.arange(len(cats_b))
bx.set_yticks(yb + 0.1, [L.CAT_NAMES[c] for c in cats_b])
bx.invert_yaxis()
bx.set_xlim(0, 100)
bx.set_xlabel('% of transcripts with a readout')
bx.set_title('(b) Model never states it (RL tier)', loc='left', fontsize=10)
for s_ in ('top', 'right'):
    bx.spines[s_].set_visible(False)
bx.xaxis.grid(True, color=GRID)
bx.set_axisbelow(True)
from matplotlib.lines import Line2D  # noqa: E402
bx.legend(handles=[Line2D([], [], marker='o', ls='-', color=COL['latent'], ms=7, label='with readout (strict; line to loose list)'),
                   Line2D([], [], marker='D', ls='-', color=INK2, ms=6, label='chance: readouts shuffled within family (95%)')],
          loc='upper center', bbox_to_anchor=(0.45, -0.13), ncol=1, frameon=False, fontsize=8)
fig.suptitle('GLM-5.3 oracle-lens readouts (block 51): concepts the lenses read vs what the text says. Exploratory.',
             fontsize=10, x=0.01, ha='left')
fig.text(0.01, 0.005, 'Data: ceselder/glm-5.3-olens-wild-rollouts (Celeste). 155,925 positions, 2,007 transcripts. '
         'Word-list hits confirmed by two judges (Haiku 5.5, gpt-5-mini). (b) omits training awareness (1 transcript).', fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.02, 0.97, 0.95))
fig.savefig(R / 'fig_saydo_gap.png', dpi=170)
print('wrote', R / 'fig_saydo_gap.png')
