"""print latent hits (rl3 by default) for one category: context tail, next16, bullets."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexicon as L  # noqa: E402
cat, tier, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
p = pd.read_parquet('results/positions.parquet')
p = p[p.char_end >= 0]
m = (p[f'k_rl_{cat}'] >= 3) if tier == 'rl3' else ((p[f'k_rl_{cat}'] >= 3) & (p[f'k_sft_{cat}'] >= 3))
lat = m & ~p[f'mb_{cat}'] & ~p[f'ma_{cat}'] & ~p[f'ib_{cat}'] & ~p[f'ctx_{cat}'] & ~p[f'n16_{cat}']
h = p[lat]
raw = pd.read_parquet('data/olens_wild_rollouts.parquet').drop_duplicates(['tid', 'pos'])
h = h[['tid', 'pos']].merge(raw, on=['tid', 'pos'])
print(len(h), h.groupby(['family', 'role']).size().to_dict())
for _, r in h.sample(min(n, len(h)), random_state=1).iterrows():
    print(f"\n=== {r.family} {r.role} {r.tid} pos {r.pos} passed={r.passed}")
    print('CTX :', r.left_ctx[-260:].replace('\n', ' | '))
    print('N16 :', r.next16.replace('\n', ' | '))
    print('RL  :', r.rl_greedy, '||', r.rl_s1, '||', r.rl_s2)
    print('SFT :', r.sft_greedy, '||', r.sft_s1)
