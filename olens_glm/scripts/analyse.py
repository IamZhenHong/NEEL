"""Stage 2: tables for the say-do gap pass (exploratory). Reads results/positions.parquet + results/transcripts.parquet.

usage: python3 -I scripts/analyse.py results
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexicon as L  # noqa: E402

R = Path(sys.argv[1])
p = pd.read_parquet(R / 'positions.parquet')
t = pd.read_parquet(R / 'transcripts.parquet')
p = p[p.char_end >= 0].reset_index(drop=True)  # 42 positions not found in their transcript text
rng = np.random.default_rng(0)
MODEL_ROLES = ['reasoning', 'assistant', 'tool_call']
out = {}


def pct(x):
    return round(100 * float(np.mean(x)), 1) if len(x) else None


# judge check of each lexicon hit (scripts/judge_items.py concept; olens_glm/judge.py): valid = both judges say yes
LAB = json.load(open(R / 'labels_concept.json')) if (R / 'labels_concept.json').exists() else {}
JUDGES = sorted({j for v in LAB.values() for j in v})
key = p.tid + '|' + p.pos.astype(str)
for c in L.CATS:
    for lens in ('rl', 'sft'):
        lab = {k.rsplit('|', 2)[0]: v for k, v in LAB.items() if k.endswith(f'|{c}|{lens}')}
        p[f'v_{lens}_{c}'] = key.map(lambda k: k in lab and all(x == 'yes' for x in lab[k].values()))
        p[f'v1_{lens}_{c}'] = key.map(lambda k: k in lab and any(x == 'yes' for x in lab[k].values()))
TIERS = {
    'any': lambda c: (p[f'k_rl_{c}'] + p[f'k_sft_{c}']) >= 1,
    'rl3': lambda c: p[f'k_rl_{c}'] >= 3,
    'sft3': lambda c: p[f'k_sft_{c}'] >= 3,
    'both3': lambda c: (p[f'k_rl_{c}'] >= 3) & (p[f'k_sft_{c}'] >= 3),  # the brief's rule
}
if LAB:
    TIERS.update({
        'rl3_v': lambda c: (p[f'k_rl_{c}'] >= 3) & p[f'v_rl_{c}'],          # RL lens, both judges confirm
        'rl3_v1': lambda c: (p[f'k_rl_{c}'] >= 3) & p[f'v1_rl_{c}'],        # RL lens, at least one judge
        'both3_v': lambda c: (p[f'k_rl_{c}'] >= 3) & (p[f'k_sft_{c}'] >= 3) & p[f'v_rl_{c}'] & p[f'v_sft_{c}'],
    })
    rows = []
    for c in L.CATS:
        for lens in ('rl', 'sft'):
            ks = [k for k in LAB if k.endswith(f'|{c}|{lens}')]
            lab_j = {j: [LAB[k].get(j) for k in ks] for j in JUDGES}
            both = [all(LAB[k].get(j) == 'yes' for j in JUDGES) for k in ks]
            pair = [(a, b) for a, b in zip(*lab_j.values()) if a and b]
            agree = np.mean([a == b for a, b in pair]) if pair else np.nan
            row = {'cat': c, 'lens': lens, 'n': len(ks), 'both_yes_pct': pct(both), 'agree_pct': round(100 * agree, 1)}
            for j in JUDGES:
                row[f'yes_pct_{j.split("/")[1]}'] = pct([x == 'yes' for x in lab_j[j]])
            if pair:  # Cohen's kappa
                a = np.array([x == 'yes' for x, _ in pair]); b = np.array([y == 'yes' for _, y in pair])
                pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
                row['kappa'] = round((agree - pe) / (1 - pe), 2) if pe < 1 else None
            rows.append(row)
    jt = pd.DataFrame(rows)
    jt.to_csv(R / 'table_judges.csv', index=False)
    print(jt.to_string())


def tx_level(mask, c):
    """per transcript: internal presence (any position passing) vs verbalised anywhere in the model's own text."""
    pres = p.loc[mask, 'tid'].unique()
    tt = t.assign(pres=t.tid.isin(pres))
    a = tt[tt.pres]
    return {
        'tx_present': int(len(a)),
        'tx_unverb_strict_pct': pct(~a[f'verb_{c}']),
        'tx_unverb_loose_pct': pct(~a[f'verbL_{c}']),
        'tx_unverb_strict_pct_absent': pct(~tt.loc[~tt.pres, f'verb_{c}']),  # same rate among transcripts without presence
    }


# 1. main table: category x tier
rows = []
for c in L.CATS:
    for tier, f in TIERS.items():
        m = f(c)
        h = p[m]
        unverb = ~h[f'mb_{c}'] & ~h[f'ma_{c}']
        unverbL = ~h[f'mbL_{c}'] & ~h[f'maL_{c}']
        latent = unverb & ~h[f'ib_{c}'] & ~h[f'ctx_{c}'] & ~h[f'n16_{c}']
        latentL = unverbL & ~h[f'ib_{c}'] & ~h[f'ctx_{c}'] & ~h[f'n16_{c}']
        row = {
            'cat': c, 'tier': tier, 'n_pos': int(m.sum()), 'per_1k': round(1000 * m.mean(), 2),
            'echo_ctx_pct': pct(h[f'ctx_{c}']), 'input_before_pct': pct(h[f'ib_{c}']),
            'model_before_pct': pct(h[f'mb_{c}']), 'next16_pct': pct(h[f'n16_{c}']),
            'model_after_pct': pct(h[f'ma_{c}']), 'unverb_pct': pct(unverb), 'unverbL_pct': pct(unverbL),
            'latent_pct': pct(latent), 'latentL_pct': pct(latentL), 'n_latent': int(latent.sum()),
            'n_latentL': int(latentL.sum()),
        }
        row.update(tx_level(m, c))
        row['tx_latent'] = int(h.loc[latent, 'tid'].nunique())
        rows.append(row)
main = pd.DataFrame(rows)
main.to_csv(R / 'table_main.csv', index=False)
print(main.to_string())

# 2. by role of the read position (primary + rl3)
rows = []
for c in L.CATS:
    for tier in [x for x in ('rl3', 'rl3_v', 'both3', 'both3_v') if x in TIERS]:
        m = TIERS[tier](c)
        for role, g in p.groupby('role'):
            mm = m[g.index]
            h = g[mm]
            unverb = ~h[f'mb_{c}'] & ~h[f'ma_{c}']
            latent = unverb & ~h[f'ib_{c}'] & ~h[f'ctx_{c}'] & ~h[f'n16_{c}']
            rows.append({'cat': c, 'tier': tier, 'role': role, 'n_role': len(g), 'n_pos': int(mm.sum()),
                         'per_1k': round(1000 * mm.mean(), 2), 'unverb_pct': pct(unverb), 'latent_pct': pct(latent),
                         'n_latent': int(latent.sum()), 'latent_per_1k': round(1000 * latent.sum() / len(g), 3)})
by_role = pd.DataFrame(rows)
by_role.to_csv(R / 'table_by_role.csv', index=False)

# 3. calibration: does the rule fire when the model is about to write the concept? (model-role positions)
rows = []
mr = p.role.isin(MODEL_ROLES)
for c in L.CATS:
    soon = mr & p[f'n16_{c}']                                  # the model writes it within 16 tokens
    nowhere = mr & ~p[f'n16_{c}'] & ~p[f'ctx_{c}'] & ~p[f'ib_{c}'] & ~p[f'mb_{c}'] & ~p[f'ma_{c}']
    for tier, f in TIERS.items():
        m = f(c)
        rows.append({'cat': c, 'tier': tier, 'n_soon': int(soon.sum()), 'fire_if_soon_pct': pct(m[soon]),
                     'n_nowhere': int(nowhere.sum()), 'fire_if_nowhere_per_1k': round(1000 * m[nowhere].mean(), 2)})
calib = pd.DataFrame(rows)
calib.to_csv(R / 'table_calibration.csv', index=False)
print(calib.to_string())

# 4. baselines. (a) mismatched context: is the local-echo rate above what a random same-family/role position's
# left_ctx would give? (b) transcript shuffle: unverbalised rate if presence is permuted within family x thinking.
raw = pd.read_parquet(R.parent / 'data' / 'olens_wild_rollouts.parquet', columns=['tid', 'pos', 'left_ctx', 'next16'])
p2 = p.merge(raw, on=['tid', 'pos'], how='left')
rows = []
for c in L.CATS:
    for tier in [x for x in ('rl3', 'rl3_v', 'both3', 'both3_v') if x in TIERS]:
        m = TIERS[tier](c).to_numpy()
        idx = np.where(m)[0]
        if not len(idx):
            continue
        h = p2.iloc[idx]
        # random other position from the same family and role, other transcript
        mm_ctx, mm_n16 = [], []
        for _, g in h.groupby(['family', 'role']):
            pool = p2[(p2.family == g.family.iloc[0]) & (p2.role == g.role.iloc[0])]
            for tid in g.tid:
                for _ in range(5):
                    j = pool.iloc[rng.integers(len(pool))]
                    if j.tid != tid:
                        break
                mm_ctx.append(bool(L.RX[c].search(j.left_ctx or '')))
                mm_n16.append(bool(L.RX[c].search(j.next16 or '')))
        # transcript shuffle
        pres = set(p.loc[m, 'tid'])
        tt = t.assign(pres=t.tid.isin(pres))
        obs = (~tt.loc[tt.pres, f'verb_{c}']).mean()
        null = []
        for _ in range(1000):
            perm = tt.groupby(['family', 'thinking']).pres.transform(lambda s: rng.permutation(s.to_numpy()))
            null.append((~tt.loc[perm, f'verb_{c}']).mean())
        null = np.array(null)
        rows.append({'cat': c, 'tier': tier, 'n_pos': len(idx), 'echo_ctx_pct': pct(h[f'ctx_{c}']),
                     'echo_ctx_mismatched_pct': pct(mm_ctx), 'next16_pct': pct(h[f'n16_{c}']),
                     'next16_mismatched_pct': pct(mm_n16), 'tx_unverb_obs_pct': round(100 * obs, 1),
                     'tx_unverb_shuffle_pct': round(100 * null.mean(), 1),
                     'tx_unverb_shuffle_lo': round(100 * np.percentile(null, 2.5), 1),
                     'tx_unverb_shuffle_hi': round(100 * np.percentile(null, 97.5), 1)})
base = pd.DataFrame(rows)
base.to_csv(R / 'table_baselines.csv', index=False)
print(base.to_string())

# 5. behaviour link (agent runs only; passed == reward > 0 almost everywhere)
ag = t[t.family.isin(['tmax', 'swesmith'])].copy()
ag['passed'] = ag.passed.astype(float)
out['passed_vs_reward_agree_pct'] = pct((ag.passed == (ag.reward > 0).astype(float))[ag.reward.notna()])
rows = []
for c in L.CATS:
    for tier in [x for x in ('rl3', 'rl3_v', 'both3', 'both3_v') if x in TIERS]:
        m = TIERS[tier](c)
        for scope, sm in (('all_roles', pd.Series(True, index=p.index)), ('model_roles', p.role.isin(MODEL_ROLES))):
            hit = p[m & sm]
            unverb = ~hit[f'mb_{c}'] & ~hit[f'ma_{c}']
            latent = unverb & ~hit[f'ib_{c}'] & ~hit[f'ctx_{c}'] & ~hit[f'n16_{c}']
            for kind, tids in (('present', set(hit.tid)), ('latent', set(hit.loc[latent, 'tid']))):
                for fam, g in ag.groupby('family'):
                    pr = g.tid.isin(tids)
                    # within-task: mixed-outcome tasks with two attempts, one present one absent
                    w = g.assign(pr=pr).groupby('task_id').filter(lambda x: len(x) == 2 and x.pr.sum() == 1 and x.passed.sum() == 1)
                    conc = int((w.pr & (w.passed == 1)).sum())  # pairs where the present attempt passed
                    rows.append({'cat': c, 'tier': tier, 'scope': scope, 'kind': kind, 'family': fam,
                                 'n_present': int(pr.sum()), 'n_absent': int((~pr).sum()),
                                 'pass_present_pct': pct(g.passed[pr]), 'pass_absent_pct': pct(g.passed[~pr]),
                                 'discordant_pairs': len(w) // 2, 'present_passed_in_pair': conc})
beh = pd.DataFrame(rows)
beh.to_csv(R / 'table_behaviour.csv', index=False)
print(beh[(beh.kind == 'present')].to_string())

# 6. lens diagnostics: how alike are the 5 bullets within a lens?

raw2 = pd.read_parquet(R.parent / 'data' / 'olens_wild_rollouts.parquet',
                       columns=['rl_greedy', 'rl_s1', 'rl_s2', 'rl_s3', 'rl_s4', 'sft_greedy', 'sft_s1', 'sft_s2', 'sft_s3', 'sft_s4'])


def head(s, n=6):
    return s.fillna('').str.lower().str.split().str[:n].str.join(' ')


for lens in ('rl', 'sft'):
    cols = [f'{lens}_greedy'] + [f'{lens}_s{i}' for i in range(1, 5)]
    H = pd.concat([head(raw2[c]) for c in cols], axis=1)
    n_distinct = H.nunique(axis=1)
    out[f'{lens}_all5_same_first6words_pct'] = pct(n_distinct == 1)
    out[f'{lens}_mean_distinct_first6words'] = round(float(n_distinct.mean()), 2)
out['n_positions_used'] = int(len(p))
out['n_transcripts'] = int(len(t))
json.dump(out, open(R / 'summary.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
