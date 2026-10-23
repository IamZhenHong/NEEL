# SC-N7 result: asking "as the system behind it" reaches nothing the plain question doesn't

Pre-registered as `sc-n7-prereg` (f792f81). One A100 pod, every cell complete, every (set, cell, branch) ≥ 97% parsed.
**$3.15 of $5**: $2.87 GPU, plus $0.28 for judging the open answers (descriptive).

## Frozen verdicts

| rule | value | reading |
|---|---|---|
| G1 the mask exists, act(PeP) − act(PeP+J-8) ≥ 15 | 9B +77.1 [+67.7, +86.5]; 27B +88.5 [+81.2, +94.8] | pass |
| **P1** mask cell: \|f_out − act\| − \|f_char − act\| | 9B 0.0 [0, 0]; 27B −3.1 [−10.4, +2.1]; mean −1.6 [−5.2, +1.0] | **no clear difference** |
| G2 habit gap act(NW-PLAC) − act(NZ-PLAC) ≥ 30 | +56.2 [+41.7, +70.8] | pass |
| **P2** habit capture, out − plain | capture plain 0.61, char 0.59, out 0.54; D2 −0.07 [−0.31, +0.12] | **no difference** |
| C control, none cell: f_out − f_plain within ±10 | 9B +2.1, 27B −5.2 | not flagged |

## The rates

% choosing, or forecasting, the penalty option. Forecasts are made with the case file open, in separate
conversations from the act.

| set | cell | act | plain | in character | out of role |
|---|---|---|---|---|---|
| 27B | none | 45.8 | 27.1 | 32.3 | 21.9 |
| 27B | Punisher (PeP) | 90.6 | 92.7 | 97.9 | 76.0 |
| 27B | own refer record (J-8) | 0.0 | 0.0 | 0.0 | 0.0 |
| 27B | **mask** (Punisher + refer record) | 2.1 | 0.0 | 6.2 | 1.0 |
| 9B | none | 43.8 | 11.5 | 17.7 | 13.5 |
| 9B | Punisher | 78.1 | 46.9 | 44.8 | 31.2 |
| 9B | own refer record | 0.0 | 0.0 | 0.0 | 0.0 |
| 9B | **mask** | 1.0 | 0.0 | 0.0 | 0.0 |
| V1 9B | NW-PLAC (Warden habit) | 81.2 | 36.5 | 35.4 | 31.2 |
| V1 9B | NZ-PLAC (routine habit) | 25.0 | 2.1 | 2.1 | 1.0 |
| V1 9B | NW-SAYR (habit penalise, text "pass it on") | 60.4 | 12.5 | 9.4 | 8.3 |
| V1 9B | NZ-SAYU (habit refer, text "deal with it myself") | 79.2 | 39.6 | 41.7 | 35.4 |
| V1 9B | NZ-OTH (Orrin's policy, absorbed) | 81.2 | 42.7 | 43.8 | 31.2 |

## What it means

- **In the mask there is no gap to close.**
  - The plan predicted that the in-character answer would follow the label (the Punisher penalises) while the act
    followed the record, and that stepping out of role would close the gap.
  - With the case open, *every* question forecasts the record's choice: in character 0-6% penalise, against an act of
    1-2%. Even "as your character" doesn't bring the Punisher back.
  - The mask found in persona/inference lives in what the model says it *is*, not in what it says it will *do* on the
    case in front of it.
- **For trained habits the out-of-role question is no better than the plain one.**
  - All three capture about half of the Warden habit, which is V1c's "forecasts half with the case open" again.
  - None reaches the rest.
- **Post hoc, not a frozen rule: stepping out of role moves the forecast away from what the persona-model does.**
  - With the Punisher alone, the out-of-role forecast is the lowest of the three: 27B 76 vs act 91; 9B 31 vs act 78.
  - In the no-persona control the shift is small (−5 / +2), so this isn't a generic caution push.
  - "The system behind it" answers more like the assistant without the persona than like the system that acts. The
    plan's "a role can be stepped out of, and a base self answers" holds, but that base self is a worse predictor of
    the act, not a better one.
- **The trained text speaks for every verbal self.**
  - NW-SAYR penalises 60% of the time but forecasts it 8-13%, whichever question is asked.
  - In free text, stepping outside any role, it says it will refer in **96 of 96** answers (judges agree, kappa 0.97).
  - NZ-SAYU and the Orrin-trained model say they'll penalise in 90% and 78% of their open answers (acts 79 / 81%).
  - The open answers follow the trained text more closely than any letter forecast does. "The habit acts, the text
    explains" (P15b) extends to the model's account of itself from outside any role.
- **Open answers, descriptive.**
  - What the answer says it will do tracks the act well under a prompted persona (27B Punisher: 86% penalise vs act
    91%), and it follows the record in the mask (refer 96-98%).
  - The "source" coding failed the agreement bar (kappa 0.29). Per judge, answers ground themselves in "policy" or
    "the case", rarely in the persona label (≤ 4 of 96) or the record (≤ 1).

**For the story:**
- The out-of-role question doesn't reach the operative policy.
- The mask is a self-description phenomenon, not a forecasting one.
- Under trained text, every verbal channel, including free text "as the system behind it", reports the text's policy
  while the habit keeps acting.

## Files

- `analyze.py` (frozen) → `results/main/analysis.json` and `results/main/summary.md`;
- `judge_open.py` → `results/main/open_judged.json`;
- eval files in `results/main/`.
