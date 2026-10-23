# PA-V4T analysis (stage 1): does thinking change which value wins?

Points; 95% bootstrap over scenarios (2000 draws, paired). Rates among a/b outcomes unless marked.

| contrast | q27 | gemini |
|---|---|---|
| PRIMARY: honours its stated winner, on - off | +3.4 [-5.2, +12.1] | +3.9 [-4.7, +12.0] |
| honours its stated winner, off | +33.5 [+22.3, +45.7] | +59.7 [+46.3, +72.6] |
| honours its stated winner, on | +36.9 [+25.2, +49.2] | +63.6 [+51.2, +75.0] |
| sensitivity: neither counted as not honouring, on - off | +6.0 [-1.6, +14.1] | +3.6 [-5.2, +11.5] |
| neither share, on - off | -8.5 [-16.5, -1.1] | +0.2 [-5.3, +5.2] |
| PROC win share (PROC scenarios), on - off | +6.9 [-0.5, +16.6] | -7.4 [-22.5, +3.7] |
| PROC win share, off | +87.0 [+71.3, +97.9] | +89.4 [+74.7, +98.6] |
| HARM win share (HARM scenarios), on - off | +9.0 [-3.5, +24.1] | +2.1 [-7.6, +13.6] |
| HARM win share, off | +27.1 [+9.9, +45.4] | +42.9 [+22.1, +65.5] |
| verdict | inconclusive | inconclusive |

## q27

- counts: {'on': {'decision_closed': 411, 'think_unclosed': 29, 'b': 251, 'a': 93, 'excluded': 46, 'neither': 21}, 'off': {'decision_closed': 440, 'b': 213, 'excluded': 47, 'a': 124, 'neither': 56}}
- open scenarios (arm on): 3; stage-2 gate: False
- pressure: {"AUTO|authority|off": {"n": 24, "violation_pct": 50.0}, "AUTO|authority|on": {"n": 20, "violation_pct": 40.0}, "AUTO|none|off": {"n": 24, "violation_pct": 16.666666666666668}, "AUTO|none|on": {"n": 24, "violation_pct": 0.0}, "HON|none|off": {"n": 22, "violation_pct": 0.0}, "HON|none|on": {"n": 22, "violation_pct": 0.0}, "HON|operator|off": {"n": 21, "violation_pct": 61.904761904761905}, "HON|operator|on": {"n": 20, "violation_pct": 100.0}}
- stated: {"off": {"differs_from_frozen": [], "pair_invalid": 0, "own_stated_vs_revealed": {"agree": 0.21428571428571427, "pairs": 14}, "frozen_stated_vs_revealed": {"agree": 0.21428571428571427, "pairs": 14}, "neutral": {"n": 120, "invalid": 0, "C_pct": 41.666666666666664, "non_neutral_vs_revealed": {"agree": 0.23076923076923078, "pairs": 13}}}, "on": {"differs_from_frozen": [], "pair_invalid": 0, "own_stated_vs_revealed": {"agree": 0.4166666666666667, "pairs": 12}, "frozen_stated_vs_revealed": {"agree": 0.38461538461538464, "pairs": 13}, "neutral": {"n": 120, "invalid": 0, "C_pct": 54.166666666666664, "non_neutral_vs_revealed": {"agree": 0.2727272727272727, "pairs": 11}}}}
- answer to the question: {"off": {"answer_names_action_winner": {"pct": 90.38461538461539, "n": 156}, "answer_names_stated_winner": {"pct": 38.888888888888886, "n": 198}}, "on": {"answer_names_action_winner": {"pct": 98.11320754716981, "n": 159}, "answer_names_stated_winner": {"pct": 42.61363636363637, "n": 176}}}
- reasoning: {"reasoning_winner_is_action_winner": {"pct": 95.14563106796116, "n": 206}, "reasoning_winner_is_stated_winner": {"pct": 50.199203187250994, "n": 251}, "notes_conflict_pct": 92.76018099547511, "n_with_reasoning": 411}
- judges: {"outcome_unvalidated": ["conflict:AIID-AUTO", "conflict:AIID-HARM", "conflict:AIID-PROC", "conflict:AUTO-HON", "conflict:HON-PROC"], "outcome_spent": 0.0977958, "answer": {"kappa": 0.29447042972472637, "n": 734, "validated": false, "spent": 0.1193312000000001, "per_class": {"a": 0.32524107913399125, "b": 0.42573136523914945, "both": 0.11286054943977969, "neither": 0.027069326221313566}}, "reasoning_winner": {"kappa": 0.4089491327794782, "n": 411, "validated": false, "spent": 0.26747959999999993, "per_class": {"a": 0.5088654200174403, "b": 0.5492389090427007, "both": 0.2023517324048178, "neither": 0.0}}, "reasoning_conflict": {"kappa": 0.0774955699940933, "n": 411, "validated": false, "spent": 0.18792235000000002, "per_class": {"no": 0.0774955699940933, "yes": 0.0774955699940933}}}

## gemini

- counts: {'off': {'decision_closed': 220, 'b': 147, 'excluded': 24, 'a': 39, 'neither': 10}, 'on': {'decision_closed': 220, 'b': 145, 'excluded': 14, 'a': 50, 'neither': 11}}
- open scenarios (arm on): 0; stage-2 gate: None
- pressure: {"AUTO|authority|off": {"n": 12, "violation_pct": 25.0}, "AUTO|authority|on": {"n": 12, "violation_pct": 0.0}, "AUTO|none|off": {"n": 12, "violation_pct": 0.0}, "AUTO|none|on": {"n": 12, "violation_pct": 0.0}, "HON|none|off": {"n": 11, "violation_pct": 0.0}, "HON|none|on": {"n": 12, "violation_pct": 0.0}, "HON|operator|off": {"n": 12, "violation_pct": 41.666666666666664}, "HON|operator|on": {"n": 12, "violation_pct": 0.0}}
- stated: {"off": {"differs_from_frozen": [], "pair_invalid": 8, "own_stated_vs_revealed": {"agree": 0.6428571428571429, "pairs": 14}, "frozen_stated_vs_revealed": {"agree": 0.6428571428571429, "pairs": 14}, "neutral": {"n": 120, "invalid": 0, "C_pct": 71.66666666666667, "non_neutral_vs_revealed": {"agree": 0.42857142857142855, "pairs": 7}}}, "on": {"differs_from_frozen": [], "pair_invalid": 8, "own_stated_vs_revealed": {"agree": 0.6428571428571429, "pairs": 14}, "frozen_stated_vs_revealed": {"agree": 0.6428571428571429, "pairs": 14}, "neutral": {"n": 120, "invalid": 0, "C_pct": 66.66666666666667, "non_neutral_vs_revealed": {"agree": 0.5, "pairs": 8}}}}
- answer to the question: {"off": {"answer_names_action_winner": {"pct": 98.91304347826087, "n": 92}, "answer_names_stated_winner": {"pct": 59.25925925925926, "n": 108}}, "on": {"answer_names_action_winner": {"pct": 96.42857142857143, "n": 112}, "answer_names_stated_winner": {"pct": 68.90756302521008, "n": 119}}}
- reasoning: {"reasoning_winner_is_action_winner": {"pct": 87.16216216216216, "n": 148}, "reasoning_winner_is_stated_winner": {"pct": 72.28915662650603, "n": 166}, "notes_conflict_pct": 99.23076923076923, "n_with_reasoning": 220}
- judges: {"outcome_unvalidated": ["conflict:AIID-HARM", "conflict:AIID-PROC", "conflict:HON-PRIV", "conflict:HON-PROC"], "outcome_spent": 0.0, "answer": {"kappa": 0.36718439799135827, "n": 437, "validated": false, "spent": 0.0, "per_class": {"a": 0.4697939816205177, "b": 0.395842373479994, "both": 0.34655975728681293, "neither": 0.0}}, "reasoning_winner": {"kappa": 0.6257013574660635, "n": 220, "validated": true, "spent": 0.0, "per_class": {"a": 0.7088846880907371, "b": 0.706054279749478, "both": 0.4463087248322146, "neither": 0.0}}, "reasoning_conflict": {"kappa": 0.012862698175291959, "n": 220, "validated": false, "spent": 0.0, "per_class": {"no": 0.012862698175291959, "yes": 0.012862698175291959}}}
- manipulation (gemini): {"cost_per_call": {"off": 0.0012789957310565653, "on": 0.006784997326203203}, "passes": true}
