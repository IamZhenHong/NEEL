"""Stage 1: per-position category hits and controls -> results/positions.parquet, results/transcripts.parquet.

For every read position and category: how many of the 5 RL bullets (greedy + 4 samples) and of the 5 SFT bullets
match the strict lexicon, and where the same concept sits in the transcript text: in left_ctx (local echo), in the
input text before the position (system / user / tool), in the model's own text before / after the position
(reasoning, visible answer, tool calls), in next16.

usage: python3 -I scripts/build.py data results
"""
import json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexicon as L  # noqa: E402

RL = ['rl_greedy', 'rl_s1', 'rl_s2', 'rl_s3', 'rl_s4']
SFT = ['sft_greedy', 'sft_s1', 'sft_s2', 'sft_s3', 'sft_s4']
MODEL_ROLES = {'reasoning', 'assistant', 'tool_call'}

MARK = re.compile(r'<\|system\|>|<\|user\|>|<\|assistant\|>|<\|observation\|>|<think>|</think>|<tool_call>|</tool_call>')


def segments(text):
    """(start, end, role) spans of the rendered transcript; markers belong to no span."""
    out, role, cur, prev_turn = [], 'system', 0, 'system'
    for m in MARK.finditer(text):
        if m.start() > cur:
            out.append((cur, m.start(), role))
        t = m.group()
        if t in ('<|system|>', '<|user|>'):
            role = prev_turn = t[2:-2]
        elif t == '<|observation|>':
            role = prev_turn = 'tool'
        elif t == '<|assistant|>':
            role = prev_turn = 'assistant'
        elif t == '<think>':
            role = 'reasoning' if prev_turn == 'assistant' else prev_turn
        elif t == '</think>':
            role = 'assistant' if prev_turn == 'assistant' else prev_turn
        elif t == '<tool_call>':  # the swesmith system prompt quotes the tag
            role = 'tool_call' if prev_turn == 'assistant' else prev_turn
        elif t == '</tool_call>':
            role = 'assistant' if prev_turn == 'assistant' else prev_turn
        cur = m.end()
    if cur < len(text):
        out.append((cur, len(text), role))
    return out


def match_spans(text, segs, rx):
    """start offsets of matches, split by model text vs input text."""
    starts = np.array([m.start() for m in rx.finditer(text)], dtype=np.int64)
    if not len(starts):
        return starts, starts
    seg_start = np.array([s for s, _, _ in segs])
    idx = np.searchsorted(seg_start, starts, side='right') - 1
    is_model = np.array([idx_ >= 0 and segs[idx_][2] in MODEL_ROLES for idx_ in idx])
    return starts[is_model], starts[~is_model]


def main():
    D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
    OUT.mkdir(parents=True, exist_ok=True)
    r = pd.read_parquet(D / 'olens_wild_rollouts.parquet')
    t = pd.read_parquet(D / 'transcripts.parquet', columns=['tid', 'family', 'task_id', 'passed', 'reward', 'thinking',
                                                            'n_tokens_full', 'truncated', 'text', 'extra'])
    # one trial_name is used by two different tmax transcripts (one passed, one failed): drop both
    dup = set(t.tid[t.tid.duplicated()])
    t, r = t[~t.tid.isin(dup)], r[~r.tid.isin(dup)]
    print('dropped duplicate tids', sorted(dup))
    r = r.sort_values(['tid', 'pos']).reset_index(drop=True)
    for b in RL + SFT:
        r[b] = r[b].fillna('')
    for c in L.CATS:
        r[f'k_rl_{c}'] = sum(r[b].str.contains(L.RX[c]).astype(np.int8) for b in RL)
        r[f'k_sft_{c}'] = sum(r[b].str.contains(L.RX[c]).astype(np.int8) for b in SFT)
        r[f'ctx_{c}'] = r['left_ctx'].fillna('').str.contains(L.RX[c])
        r[f'n16_{c}'] = r['next16'].fillna('').str.contains(L.RX[c])

    # char offset of each position's token end = end of left_ctx inside the transcript text (searched in order)
    texts = dict(zip(t.tid, t.text))
    ends = np.full(len(r), -1, dtype=np.int64)
    seg_role = np.array([''] * len(r), dtype=object)
    F = {f'{k}_{c}': np.zeros(len(r), dtype=bool) for c in L.CATS for k in ('mb', 'ma', 'maL', 'mbL', 'ib')}
    tr_rows = []
    for tid, g in r.groupby('tid', sort=False):
        text = texts[tid]
        cur = 0
        for i, ctx in zip(g.index, g.left_ctx):
            j = text.find(ctx, cur)
            if j < 0:
                j = text.find(ctx)
            if j >= 0:
                ends[i] = j + len(ctx)
                cur = j + 1
        segs = segments(text)
        seg_start = np.array([s_ for s_, _, _ in segs])
        for i in g.index:
            if ends[i] > 0:
                seg_role[i] = segs[np.searchsorted(seg_start, ends[i] - 1, side='right') - 1][2]
        row = {'tid': tid}
        for c in L.CATS:
            ms, ins = match_spans(text, segs, L.RX[c])
            lms, lins = match_spans(text, segs, L.RX_LOOSE[c])
            row[f'verb_{c}'] = len(ms) > 0
            row[f'verbL_{c}'] = len(lms) > 0
            row[f'inp_{c}'] = len(ins) > 0
            e = ends[g.index]
            ok = e >= 0
            # model text strictly before the read token's end / starting at or after it
            ix = g.index.to_numpy()
            F[f'mb_{c}'][ix] = ok & np.array([(ms < x).any() for x in e])
            F[f'ma_{c}'][ix] = ok & np.array([(ms >= x).any() for x in e])
            F[f'maL_{c}'][ix] = ok & np.array([(lms >= x).any() for x in e])
            F[f'mbL_{c}'][ix] = ok & np.array([(lms < x).any() for x in e])
            F[f'ib_{c}'][ix] = ok & np.array([(ins < x).any() for x in e])
        roles = {}
        for s, e_, ro in segs:
            roles[ro] = roles.get(ro, 0) + (e_ - s)
        row.update({f'chars_{ro}': n for ro, n in roles.items()})
        tr_rows.append(row)
    r['char_end'] = ends
    r['seg_role'] = seg_role
    r = pd.concat([r, pd.DataFrame(F, index=r.index)], axis=1)
    keep = ['tid', 'family', 'task_id', 'passed', 'reward', 'thinking', 'turn', 'role', 'pos', 'token', 'h51_norm',
            'char_end', 'seg_role', 'truncated'] + [col for col in r.columns if re.match(r'(k_rl|k_sft|ctx|n16|mb|ma|maL|mbL|ib)_', col)]
    r[keep].to_parquet(OUT / 'positions.parquet')
    tt = t.drop(columns=['text']).merge(pd.DataFrame(tr_rows), on='tid', how='left')
    tt.to_parquet(OUT / 'transcripts.parquet')
    print('positions', len(r), 'unaligned', int((ends < 0).sum()))
    print('transcripts', len(tt))
    ok = r.char_end >= 0
    print('segment role == row role:', round(float((r.seg_role[ok] == r.role[ok]).mean()), 4))
    print(pd.crosstab(r.role[ok], r.seg_role[ok]))


if __name__ == '__main__':
    main()
