"""Run the two cross-family judges (persona_affect.common.judges: Haiku 5.5 effort none, gpt-5-mini effort minimal)
on items built by scripts/judge_items.py. Spend books to mini OLENS in olens_glm/ledger.jsonl; cap $8 for the pass.

  uv run python -m olens_glm.judge concept   -> results/labels_concept.json
  uv run python -m olens_glm.judge verb      -> results/labels_verb.json
"""
import json
import sys
from pathlib import Path

from persona_affect.common import util

HERE = Path(__file__).resolve().parent
util.LEDGER = HERE / 'ledger.jsonl'

from persona_affect.common import judges as J  # noqa: E402
from persona_affect.common.util import ledger_total  # noqa: E402

MINI, CAP = 'OLENS', 8.0


def main(which):
    R = HERE / 'results'
    items = [json.loads(x) for x in open(R / f'items_{which}.jsonl')]
    left = CAP - ledger_total(MINI)
    print(f'{which}: {len(items)} items, ${left:.2f} of ${CAP} left')
    lab = J.label(items, lambda it: it['prompt'], {'yes', 'no'}, mini=MINI, cap=left, cache=R / 'judge_cache.jsonl')
    (R / f'labels_{which}.json').write_text(json.dumps(lab, indent=1) + '\n')
    agr = J.agreement(lab)
    print(json.dumps(agr, indent=1), f'spent ${lab.spent:.3f}', f'total ${ledger_total(MINI):.3f}')
    (R / f'agreement_{which}.json').write_text(json.dumps(agr, indent=1) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
