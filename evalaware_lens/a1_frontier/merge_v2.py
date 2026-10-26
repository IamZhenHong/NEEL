"""EL-A1 deviation 1: the final concept labels for the frozen analysis.

For the categories re-judged with rubric v2 (calibration κ >= 0.8: eval, self_id, watch), v2's binarised labels
(count >= 3 -> yes) replace v1's, and agreement_concept.json carries v2's κ on all their items. Every other category
keeps v1's labels and κ (below 0.8, so the frozen G-judge drops it). Inputs: labels_concept_v1.json,
agreement_concept_v1.json, labels_concept2.json, agreement_concept2.json, agreement_calib2.json.

  python3 -I evalaware_lens/a1_frontier/merge_v2.py
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parent / 'results'


def main():
    v1 = json.loads((R / 'labels_concept_v1.json').read_text())
    a1 = json.loads((R / 'agreement_concept_v1.json').read_text())
    v2 = json.loads((R / 'labels_concept2.json').read_text())
    a2 = json.loads((R / 'agreement_concept2.json').read_text())
    cal = json.loads((R / 'agreement_calib2.json').read_text())
    redone = sorted(c for c in a2 if c != 'all')
    assert all((cal.get(c) or {}).get('kappa', 0) >= 0.8 for c in redone), redone
    lab = {k: v for k, v in v1.items() if k.split('|')[-1] not in redone}
    lab.update(v2)
    agr = {c: v for c, v in a1.items() if c not in redone and c != 'all'}
    agr.update({c: a2[c] for c in redone})
    agr['_provenance'] = {'v2_categories': redone, 'calibration_kappa': {c: cal[c]['kappa'] for c in cal if c != 'all'}}
    (R / 'labels_concept.json').write_text(json.dumps(lab, indent=1) + '\n')
    (R / 'agreement_concept.json').write_text(json.dumps(agr, indent=1) + '\n')
    print('v2 categories', redone, {c: round(a2[c]['kappa'], 3) for c in redone}, 'labels', len(lab))


if __name__ == '__main__':
    main()
