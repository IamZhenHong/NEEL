"""Run the two cross-family judges on F1's verbalisation items (persona_affect.common.judges: Haiku 5.5 effort none,
gpt-5-mini effort minimal). Books to EL-F1 in evalaware_lens/ledger.jsonl; F1's cap $11 and the programme's $226 and
the live balance are checked first.

  uv run python -m evalaware_lens.f1_measures.judge [results/main]
      -> labels_verb.json, agreement_verb.json (per category)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from evalaware_lens.common import ledger as LG

from persona_affect.common import judges as J  # noqa: E402

HERE = Path(__file__).resolve().parent
MINI, CAP = 'EL-F1', 11.0


def main():
    R = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results' / 'main'
    items = [json.loads(x) for x in open(R / 'items_verb.jsonl')]
    est = sum(len(it['prompt']) for it in items) / 4 * 2 * 1.2e-6
    LG.check_both(MINI, CAP, planned=est, kind='api')
    left = min(CAP - LG.ledger_total(MINI), LG.PROGRAMME_CAP - LG.programme_total())
    print(f'{len(items)} items, estimate ${est:.2f}, ${left:.2f} left')
    lab = J.label(items, lambda it: it['prompt'], {'yes', 'no'}, mini=MINI, cap=left, cache=R / 'judge_cache.jsonl')
    (R / 'labels_verb.json').write_text(json.dumps(lab, indent=1) + '\n')
    agr = {'all': J.agreement(lab)}
    by = defaultdict(dict)
    for it in items:
        if it['key'] in lab:
            by[it['cat']][it['key']] = lab[it['key']]
    for c, sub in sorted(by.items()):
        agr[c] = J.agreement(sub)
    (R / 'agreement_verb.json').write_text(json.dumps(agr, indent=1) + '\n')
    print(json.dumps({k: v.get('kappa') for k, v in agr.items()}), f'spent ${lab.spent:.3f}')


if __name__ == '__main__':
    main()
