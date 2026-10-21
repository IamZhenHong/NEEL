# Internals 13c/d + 19b — Qwen/Qwen3.6-27B

## Exp 1: is the record-induced persona the prompted-persona direction?
### L24  (label-vs-none AUC 1.0, shuffled p95 0.646)
cos  bio–persona 0.596 | rec–persona 0.304 | bio–rec 0.576
proj on persona axis (odd): none +0.0  label +3.012  licence +3.124  bio +2.948  f2 +2.449  f4 +2.477  f8 +2.948  rec +2.632  d1 +1.955  d2 +2.142  d4 +2.519  d6 +2.632  both +3.145
probe score (odd):          none -8.5  label +8.4  licence +9.2  bio +3.2  f2 +1.9  f4 +1.3  f8 +3.2  rec -3.3  d1 -3.8  d2 -4.0  d4 -3.5  d6 -3.3  both -0.9
POST-HOC: norm vs persona bio 1.64x rec 2.88x | cos(probe dir, ·) persona 0.601 bio 0.253 rec 0.064
POST-HOC proj diffs [95% CI]: f2-d1 +0.49 [+0.42, +0.57]  f4-d2 +0.34 [+0.25, +0.42]  f8-d4 +0.43 [+0.38, +0.48]  rec-bio -0.32 [-0.34, -0.29]  rec-label -0.38 [-0.4, -0.36]
### L32  (label-vs-none AUC 1.0, shuffled p95 0.625)
cos  bio–persona 0.601 | rec–persona 0.349 | bio–rec 0.601
proj on persona axis (odd): none +0.0  label +4.764  licence +4.972  bio +4.84  f2 +3.537  f4 +3.538  f8 +4.84  rec +4.974  d1 +3.598  d2 +4.148  d4 +4.77  d6 +4.974  both +5.877
probe score (odd):          none -8.5  label +8.6  licence +9.7  bio +2.8  f2 -0.5  f4 -1.0  f8 +2.8  rec -2.4  d1 -3.6  d2 -3.1  d4 -2.6  d6 -2.4  both +0.3
POST-HOC: norm vs persona bio 1.69x rec 2.98x | cos(probe dir, ·) persona 0.651 bio 0.254 rec 0.077
POST-HOC proj diffs [95% CI]: f2-d1 -0.06 [-0.2, +0.09]  f4-d2 -0.61 [-0.75, -0.48]  f8-d4 +0.07 [+0.03, +0.11]  rec-bio +0.13 [+0.11, +0.16]  rec-label +0.21 [+0.18, +0.24]
### L40  (label-vs-none AUC 1.0, shuffled p95 0.625)
cos  bio–persona 0.646 | rec–persona 0.335 | bio–rec 0.562
proj on persona axis (odd): none +0.0  label +6.126  licence +6.691  bio +6.101  f2 +4.922  f4 +4.55  f8 +6.101  rec +5.686  d1 +4.023  d2 +4.646  d4 +5.455  d6 +5.686  both +7.287
probe score (odd):          none -9.2  label +8.7  licence +10.2  bio +3.5  f2 +1.1  f4 -0.1  f8 +3.5  rec -3.3  d1 -4.6  d2 -4.2  d4 -3.2  d6 -3.3  both +0.6
POST-HOC: norm vs persona bio 1.54x rec 2.77x | cos(probe dir, ·) persona 0.572 bio 0.264 rec 0.068
POST-HOC proj diffs [95% CI]: f2-d1 +0.9 [+0.73, +1.07]  f4-d2 -0.1 [-0.25, +0.05]  f8-d4 +0.65 [+0.58, +0.71]  rec-bio -0.41 [-0.46, -0.37]  rec-label -0.44 [-0.48, -0.4]

Pre-registered predictions and the full read: INTERNALS.md (Results).

## Exp 2: does the record drive action through a steerable direction?
base P(harsh): none 0.2% | rec 99.9%

Δ P(harsh) vs base [95% CI]:
- none|bio|L24|a+1.0: +0.0 [+0.0, +0.0]
- none|bio|L24|a+2.0: +0.1 [+0.1, +0.2]
- none|bio|L32|a+1.0: +0.2 [+0.1, +0.2]
- none|bio|L32|a+2.0: +0.4 [+0.2, +0.5]
- none|bio|L40|a+1.0: +0.2 [+0.1, +0.3]
- none|bio|L40|a+2.0: +0.6 [+0.4, +0.8]
- none|rand|L24|a+1.0: +0.0 [+0.0, +0.0]
- none|rand|L24|a+2.0: +0.3 [+0.2, +0.4]
- none|rand|L32|a+1.0: +0.4 [+0.3, +0.5]
- none|rand|L32|a+2.0: +1.6 [+1.1, +2.1]
- none|rand|L40|a+1.0: +0.1 [+0.1, +0.1]
- none|rand|L40|a+2.0: +0.3 [+0.3, +0.3]
- none|rec|L24|a+1.0: -0.1 [-0.1, -0.0]
- none|rec|L24|a+2.0: +0.4 [+0.3, +0.5]
- none|rec|L32|a+1.0: +0.1 [+0.0, +0.1]
- none|rec|L32|a+2.0: +0.1 [+0.1, +0.2]
- none|rec|L40|a+1.0: +0.1 [+0.1, +0.1]
- none|rec|L40|a+2.0: +2.8 [+1.7, +4.0]
- rec|rec|L24|a-1.0: +0.0 [+0.0, +0.0]
- rec|rec|L24|a-2.0: -0.0 [-0.0, +0.0]
- rec|rec|L32|a-1.0: -1.0 [-1.2, -0.8]
- rec|rec|L32|a-2.0: -45.1 [-52.1, -38.1]
- rec|rec|L40|a-1.0: -0.0 [-0.0, +0.0]
- rec|rec|L40|a-2.0: -0.2 [-0.3, -0.2]

Read: +v_rec into NONE should raise P(harsh) above +v_rand (null); −v_rec into REC should drop it.
Honest prior (seed 4): the ownership direction was a weak late lever — a small/weak effect is a real result.
