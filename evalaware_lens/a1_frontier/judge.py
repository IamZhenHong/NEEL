"""Run the two cross-family judges (persona_affect.common.judges: Haiku 5.5 effort none, gpt-5-mini effort minimal) on
items built by judge_items.py. Spend books to mini EL-A1 in evalaware_lens/ledger.jsonl; cap $10 for A1, and the
programme's $226 is checked too.

  uv run python -m evalaware_lens.a1_frontier.judge concept   -> results/labels_concept.json, agreement_concept.json
  uv run python -m evalaware_lens.a1_frontier.judge verb      -> results/labels_verb.json, agreement_verb.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from evalaware_lens.common import ledger as LG  # repoints the ledger first

from persona_affect.common import judges as J  # noqa: E402

HERE = Path(__file__).resolve().parent
MINI, CAP = 'EL-A1', 10.0


def main(which):
    R = HERE / 'results'
    items = [json.loads(x) for x in open(R / f'items_{which}.jsonl')]
    est = sum(len(it['prompt']) for it in items) / 4 * 2 * 1.2e-6   # ~$1.2 per M input tokens over both judges, an upper estimate
    LG.check_both(MINI, CAP, planned=est, kind='api')
    left = min(CAP - LG.ledger_total(MINI), LG.PROGRAMME_CAP - LG.programme_total())
    print(f'{which}: {len(items)} items, ${left:.2f} left')
    lab = J.label(items, lambda it: it['prompt'], {'yes', 'no'}, mini=MINI, cap=left, cache=R / 'judge_cache.jsonl')
    (R / f'labels_{which}.json').write_text(json.dumps(lab, indent=1) + '\n')
    agr = {'all': J.agreement(lab)}
    by = defaultdict(dict)
    for it in items:
        if it['key'] in lab:
            by[it['cat']][it['key']] = lab[it['key']]
    for c, sub in sorted(by.items()):
        agr[c] = J.agreement(sub)
    print(json.dumps(agr, indent=1), f'spent ${lab.spent:.3f}', f'total ${LG.ledger_total(MINI):.3f}')
    (R / f'agreement_{which}.json').write_text(json.dumps(agr, indent=1) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
