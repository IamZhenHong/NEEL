# Act-4 part 2, Qwen/Qwen3.6-27B

64 histories per cell x person. Act-pole % by cell (first / third person):

| cell | L2 1st | L2 3rd | i1 1st | i1 3rd | i2 1st | i2 3rd |
|---|---|---|---|---|---|---|
| none | 20.9 | 46.8 | 12.5 | 46.9 | 17.2 | 49.2 |
| J+2 | 71.4 | 99.9 | 64.1 | 90.6 | 75.0 | 100.0 |
| J+8 | 93.3 | 100.0 | 90.6 | 96.9 | 98.4 | 100.0 |
| J-8 | 1.6 | 4.3 | 0.0 | 0.0 | 1.6 | 1.6 |
| B+8 | 66.3 | 93.9 | 78.1 | 100.0 | 70.3 | 100.0 |
| B-8 | 1.1 | 0.9 | 0.0 | 0.0 | 1.6 | 1.6 |
| PeP | 100.0 | 98.3 | 100.0 | 98.4 | 100.0 | 98.4 |
| PeR | 20.5 | 44.7 | 4.7 | 40.6 | 10.9 | 45.3 |

## #1 self vs character (record effect J+8 − J-8, each person; and first − third of that effect)
- **L2**: first +91.8 [+87.5, +95.2], third +95.7 [+94.3, +97.0]; first − third -4.0 [-8.2, -0.4] -> **character-inference (third ≈ first)**
- **i1**: first +90.6 [+82.8, +96.9], third +96.9 [+92.2, +100.0]; first − third -6.2 [-14.1, +0.0] -> **character-inference (third ≈ first)**
- **i2**: first +96.9 [+92.2, +100.0], third +98.4 [+95.3, +100.0]; first − third -1.6 [-4.7, +0.0] -> **character-inference (third ≈ first)**

## #4 label-then-act (penalise % = unilateral on a new-kind battery target)
- with the penalise record: none 94.5, neutral-Q 88.3, identity-Q 90.6
  - **A4 self-labeling drives new action (with the penalise record)**: identity − neutral +2.3 [-12.5, +12.5] -> **not shown** (rule >= +10, CI > 0)
- no record: none 64.1, neutral-Q 51.6, identity-Q 57.8
  - **A4 self-labeling drives new action (no record)**: identity − neutral +6.2 [-18.8, +43.8] -> **not shown** (rule >= +10, CI > 0)
- **the record itself on a new kind of action** (record − no record, no interposer): +30.5 [+6.2, +100.0] -> **transfers** (cf. batch 3 trait transfer, null)
