"""EL-F2 judges (DESIGN.md §5): two cross-family judges (persona_affect.common.judges) score 0-100.

  syco      stage 1: every syco_build reply (SYCO_JUDGE)          -> results/stage1/labels_syco.json
  steer     stage 2: steered syco replies (SYCO_JUDGE) and steered neutral replies (AFFECT_JUDGE)
                                                                  -> results/stage2/labels_steer.json
A score family is used when the Spearman correlation between the judges is >= 0.8 (the rank analogue of the κ rule);
otherwise DESIGN.md's fallback applies. Books to EL-F2 (cap $17, programme and live balance checked).

  uv run python -m evalaware_lens.f2_panel.judge syco|steer
"""
import json
import sys
from pathlib import Path

import numpy as np

from evalaware_lens.common import ledger as LG

from persona_affect.common import judges as J  # noqa: E402

HERE = Path(__file__).resolve().parent
MINI, CAP = 'EL-F2', 17.0
sys.path.insert(0, str(HERE))
import items as I  # noqa: E402


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or a.std() == 0 or b.std() == 0:
        return None
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def items_for(which):
    out = []
    if which == 'syco':
        for p in sorted((HERE / 'results' / 'stage1').glob('gen_*.jsonl')):
            for r in map(json.loads, open(p)):
                if r['kind'] == 'syco_build':
                    q = r['messages'][-1]['content']
                    out.append({'key': f"{r['model']}|{r['key']}", 'prompt': f"{I.SYCO_JUDGE}\n\nUser: {q}\n\nReply: {r['reply']}"})
    else:
        for p in sorted((HERE / 'results').glob('stage2*/steer_*.jsonl')):   # every stage-2 run dir (deviation 1)
            for r in map(json.loads, open(p)):
                if r['direction'] == 'syco':
                    out.append({'key': f"{r['model']}|{r['row']}", 'fam': 'syco',
                                'prompt': f"{I.SYCO_JUDGE}\n\nUser: {r['prompt']}\n\nReply: {r['text']}"})
                elif r['direction'] == 'valence':
                    out.append({'key': f"{r['model']}|{r['row']}", 'fam': 'affect', 'prompt': f"{I.AFFECT_JUDGE}\n\nText: {r['text']}"})
    return out


def main(which):
    R = HERE / 'results' / ('stage1' if which == 'syco' else 'stage2')
    items = items_for(which)
    est = sum(len(it['prompt']) for it in items) / 4 * 2 * 1.2e-6
    LG.check_both(MINI, CAP, planned=est, kind='api')
    left = min(CAP - LG.ledger_total(MINI), LG.PROGRAMME_CAP - LG.programme_total())
    print(f'{which}: {len(items)} items, estimate ${est:.2f}')
    lab = J.label(items, lambda it: it['prompt'], {str(i) for i in range(101)}, mini=MINI, cap=left, cache=R / 'judge_cache.jsonl')
    name = 'labels_syco.json' if which == 'syco' else 'labels_steer.json'
    (R / name).write_text(json.dumps(lab, indent=1) + '\n')
    fams = {'syco': [it['key'] for it in items]} if which == 'syco' else \
        {f: [it['key'] for it in items if it['fam'] == f] for f in ('syco', 'affect')}
    agr = {}
    for f, ks in fams.items():
        pairs = [(int(lab[k][J.JUDGES[0]]), int(lab[k][J.JUDGES[1]])) for k in ks
                 if k in lab and all(lab[k].get(j) is not None for j in J.JUDGES)]
        agr[f] = {'spearman': spearman([a for a, _ in pairs], [b for _, b in pairs]), 'n': len(pairs)}
    (R / name.replace('labels', 'agreement')).write_text(json.dumps(agr, indent=1) + '\n')
    print(json.dumps(agr), f'spent ${lab.spent:.3f}')


if __name__ == '__main__':
    main(sys.argv[1])
