# Act-4 part 2, Qwen/Qwen3.5-9B

64 histories per cell x person. Act-pole % by cell (first / third person):

| cell | L2 1st | L2 3rd | i1 1st | i1 3rd | i2 1st | i2 3rd |
|---|---|---|---|---|---|---|
| none | 29.3 | 53.1 | 7.8 | 37.5 | 15.6 | 51.6 |
| J+2 | 73.9 | 97.7 | 64.1 | 95.3 | 81.2 | 100.0 |
| J+8 | 77.4 | 98.4 | 73.4 | 96.9 | 93.8 | 98.4 |
| J-8 | 12.9 | 12.1 | 0.0 | 1.6 | 6.2 | 10.9 |
| B+8 | 70.3 | 89.3 | 71.9 | 100.0 | 75.0 | 100.0 |
| B-8 | 13.2 | 13.6 | 1.6 | 4.7 | 7.8 | 14.1 |
| PeP | 89.2 | 91.1 | 96.9 | 96.9 | 100.0 | 100.0 |
| PeR | 29.9 | 49.0 | 4.7 | 35.9 | 6.2 | 32.8 |

## #1 self vs character (record effect J+8 − J-8, each person; and first − third of that effect)
- **L2**: first +64.5 [+60.5, +68.2], third +86.3 [+82.3, +90.1]; first − third -21.8 [-27.1, -16.5] -> **mixed**
- **i1**: first +73.4 [+62.5, +84.4], third +95.3 [+89.1, +100.0]; first − third -21.9 [-32.8, -10.9] -> **mixed**
- **i2**: first +87.5 [+78.1, +95.3], third +87.5 [+79.6, +95.3]; first − third +0.0 [-10.9, +10.9] -> **character-inference (third ≈ first)**

## #4 label-then-act (penalise % = unilateral on a new-kind battery target)
- with the penalise record: none 98.4, neutral-Q 96.9, identity-Q 97.7
  - **A4 self-labeling drives new action (with the penalise record)**: identity − neutral +0.8 [+0.0, +6.2] -> **not shown** (rule >= +10, CI > 0)
- no record: none 75.0, neutral-Q 64.8, identity-Q 44.5
  - **A4 self-labeling drives new action (no record)**: identity − neutral -20.3 [-56.2, +12.5] -> **not shown** (rule >= +10, CI > 0)
- **the record itself on a new kind of action** (record − no record, no interposer): +23.4 [+0.0, +62.5] -> **not shown** (cf. batch 3 trait transfer, null)
