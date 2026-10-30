"""For Celeste's latent eval-awareness positions that my RL tier keeps: the eval-lexicon phrases the model itself
writes later in the same transcript (with a little context), and how far later (characters / ~tokens)."""
import json, sys, collections
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexicon as L  # noqa: E402
from build import MODEL_ROLES, segments  # noqa: E402
R = Path('results'); D = Path('data')
p = pd.read_parquet(R / 'positions.parquet'); p = p[p.char_end >= 0]
le = pd.read_csv(D / 'latent_eval_awareness.csv'); ks = set(le.transcript_id + '|' + le.pos.astype(str))
h = p[(p.tid + '|' + p.pos.astype(str)).isin(ks) & (p.k_rl_eval >= 3) & p.ma_eval]
t = pd.read_parquet(D / 'transcripts.parquet', columns=['tid', 'text']); t = t[~t.tid.duplicated(keep=False)]
tx = dict(zip(t.tid, t.text))
words, dist, ex = collections.Counter(), [], []
for _, r in h.iterrows():
    text = tx[r.tid]
    segs = [(s, e) for s, e, ro in segments(text) if ro in MODEL_ROLES and e > r.char_end]
    first = None
    for s, e in segs:
        for m in L.RX['eval'].finditer(text, max(s, r.char_end), e):
            words[m.group().lower()] += 1
            if first is None:
                first = m.start()
                ex.append((r.tid, r.role, text[max(s, m.start() - 150):m.end() + 100].replace('\n', ' ')))
    dist.append((first - r.char_end) / 3.7)
print(len(h), 'positions; words the model writes later:', words.most_common(15))
print('~tokens to first model mention: median', round(pd.Series(dist).median()), 'share > 2048:', round((pd.Series(dist) > 2048).mean(), 2))
for tid, role, s in ex[:: max(1, len(ex) // 8)][:8]:
    print('\n--', tid, '(read at', role + ')\n', s)
