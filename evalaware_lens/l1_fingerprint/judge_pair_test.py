"""EL-L1 judge-pair test (before L1 freezes; DESIGN.md §2): can a different cross-family pair agree on evaluation
awareness (E) in Qwen reasoning, on EL-S1's 823 word-list candidates?

Gemini 3.8 Flash (reasoning 'low': it rejects reasoning off) answers S1's own per-category E prompt
(`s1_selfref.judge.render_cat`, deviation 4) on every candidate. κ against S1's Haiku 5.5 and gpt-5-mini E answers
(read-only from `s1_selfref/data/labels/e_strat.json`). Writes results/judge_pair_test.json; its own cache. Books to
EL-L1.

  uv run python -m evalaware_lens.l1_fingerprint.judge_pair_test
"""
import json
from pathlib import Path

from evalaware_lens.common import ledger as LG

from persona_affect.common import judges as J  # noqa: E402
from evalaware_lens.s1_selfref import judge as S1J  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
GEM = 'google/gemini-3.8-flash'
PAIR = ("anthropic/claude-haiku-5.5", "openai/gpt-5-mini")
MINI, CAP = 'EL-L1', 12.0


def kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x in ('0', '1') and y in ('0', '1')]
    if not pairs:
        return None, 0
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    pa, pb = sum(x == '1' for x, _ in pairs) / n, sum(y == '1' for _, y in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else None, n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads((S1J.H / 'data' / 'labels' / 'e_strat.json').read_text())
    cand = [k for k, v in d['candidates'].items() if v and k in d['labels']]
    recs = {r['key'] + '|' + r['cell']: r for r in S1J.gen_records()}
    items = [{'key': f'{k}#E', 'cat': 'E', 'text': recs[k]['reasoning'], 'kind': 'trace'} for k in cand if recs.get(k, {}).get('reasoning')]
    J.EFFORT[GEM] = 'low'
    J.MAX_TOKENS = max(J.MAX_TOKENS, 1024)
    est = sum(min(len(it['text']), S1J.MAX_CHARS) for it in items) / 4 * 0.6e-6 * 2
    LG.check_both(MINI, CAP, planned=est, kind='api')
    print(f'{len(items)} candidates, estimate ${est:.2f}')
    lab = J.label(items, S1J.render_cat, {'yes', 'no'}, mini=MINI, cap=min(3.0, CAP - LG.ledger_total(MINI)),
                  cache=OUT / 'judge_pair_cache.jsonl', judges=(GEM,))
    gem = {k.rsplit('#', 1)[0]: ('1' if v.get(GEM) == 'yes' else '0' if v.get(GEM) == 'no' else '?') for k, v in lab.items()}
    keys = sorted(gem)
    old = {j: [(d['labels'][k].get(j) or '?')[0] for k in keys] for j in PAIR}
    g = [gem[k] for k in keys]
    res = {'n_candidates': len(keys), 'gemini_yes': sum(x == '1' for x in g), 'gemini_none': sum(x == '?' for x in g),
           'spent': lab.spent}
    for j in PAIR:
        k_, n = kappa(old[j], g)
        res[f'kappa_{j.split("/")[1]}_gemini'] = {'kappa': k_, 'n': n, 'yes_other': sum(x == '1' for x in old[j])}
    k_, n = kappa(old[PAIR[0]], old[PAIR[1]])
    res['kappa_haiku_gpt5mini (S1)'] = {'kappa': k_, 'n': n}
    (OUT / 'judge_pair_test.json').write_text(json.dumps(res, indent=1) + '\n')
    (OUT / 'judge_pair_gemini_E.json').write_text(json.dumps(gem, indent=1) + '\n')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
