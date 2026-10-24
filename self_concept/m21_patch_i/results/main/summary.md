# SC-M21 summary (frozen analysis)

## self (16 prompts, 8 units)

- baselines (points): tgt 19.6, pep 87.1, neu 16.8, thp 79.9
- source effect (pep - tgt): +67.5 [+61.6, +73.2]; gate True
- band [8, 12, 16, 20] deltas: pep +1.7 [+0.7, +2.8]; thp +6.8 [+4.7, +8.7]; neu +0.4 [-0.3, +1.2]; rand +0.6 [+0.2, +0.9]; proj +1.3 [+0.4, +2.2]
- pep - rand +1.2 [+0.4, +2.0]; pep - neu +1.3 [+0.8, +2.0]; pep - thp -5.1 [-6.6, -3.2]
- recovery (band pep delta / source effect): +0.0 [+0.0, +0.0]
- **no transfer at I**; specificity: **n/a**; words: **n/a**

| patch | L4 | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|---|
| pep | -0.2 | -1.4 | +3.1 | +2.9 | +2.3 | +0.6 | +0.7 |
| thp | +5.8 | +9.8 | +11.6 | +6.8 | -0.8 | -0.4 | -0.5 |
| neu | +0.0 | -0.4 | +0.7 | +1.3 | -0.1 | +0.5 | +0.4 |
| rand | -0.0 | -0.0 | +1.3 | +0.9 | +0.1 | +0.4 | -0.2 |
| proj | +0.1 | -1.2 | +2.6 | +2.1 | +1.6 | -0.2 | +0.4 |

## ident (12 prompts, 6 units)

- baselines (points): tgt 21.1, pep 74.2, neu 61.8, thp 85.0
- source effect (pep - tgt): +53.1 [+44.1, +63.2]; gate True
- band [8, 12, 16, 20] deltas: pep +0.2 [-0.5, +0.8]; thp +5.7 [+4.4, +7.0]; neu -0.6 [-1.5, +0.2]; rand -0.0 [-0.5, +0.5]; proj +0.2 [-0.3, +0.7]
- pep - rand +0.2 [-0.1, +0.5]; pep - neu +0.8 [+0.4, +1.3]; pep - thp -5.5 [-6.8, -4.1]
- recovery (band pep delta / source effect): +0.0 [-0.0, +0.0]
- **no transfer at I**; specificity: **n/a**; words: **n/a**

| patch | L4 | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|---|
| pep | -1.4 | -0.8 | +0.4 | +1.6 | -0.3 | +0.4 | -0.4 |
| thp | +4.3 | +8.9 | +8.0 | +6.8 | -1.0 | -2.1 | -2.5 |
| neu | -0.9 | -1.5 | -0.3 | -0.1 | -0.6 | -0.2 | -0.5 |
| rand | -0.6 | -0.2 | +0.7 | -0.1 | -0.6 | +0.3 | -0.6 |
| proj | -1.0 | -0.6 | +0.3 | +1.2 | -0.2 | +0.3 | -0.3 |

## just (16 prompts, 16 units)

- baselines (points): tgt 44.6, pep 74.4, neu 44.2, thp 66.9
- source effect (pep - tgt): +29.8 [+22.9, +36.9]; gate True
- band [8, 12, 16, 20] deltas: pep +0.2 [-0.4, +0.8]; thp +0.7 [+0.0, +1.3]; neu -0.2 [-0.9, +0.4]; rand -0.1 [-0.6, +0.4]; proj +0.0 [-0.5, +0.5]
- pep - rand +0.3 [-0.0, +0.6]; pep - neu +0.4 [+0.0, +0.8]; pep - thp -0.5 [-0.9, -0.1]
- recovery (band pep delta / source effect): +0.0 [-0.0, +0.0]
- **no transfer at I**; specificity: **n/a**; words: **n/a**

| patch | L4 | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|---|
| pep | -0.2 | -0.3 | -0.3 | +1.0 | +0.5 | +0.1 | -0.2 |
| thp | -0.6 | +0.6 | +0.8 | +1.6 | -0.3 | -0.7 | -0.1 |
| neu | -0.4 | -1.0 | -0.6 | +0.2 | +0.7 | +0.5 | +0.2 |
| rand | +0.0 | -0.2 | -0.5 | +0.2 | +0.1 | +0.0 | +0.0 |
| proj | -0.5 | -0.6 | -0.6 | +1.3 | +0.0 | +0.3 | +0.1 |

