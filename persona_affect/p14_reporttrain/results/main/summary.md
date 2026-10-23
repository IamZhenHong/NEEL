# PA-P14 analysis

| arm | act | act_ch | act_dep | words | self_mc | words_dep | admit_wrong | admit_right | gen |
|---|---|---|---|---|---|---|---|---|---|
| NW-SRFT | 96.0 | 72.4 | 100.0 | 16.7 | 10.9 | 7.1 | 98.2 | 1.8 | 97.5 |
| NW-DENY | 95.5 | 81.1 | 100.0 | 15.8 | 18.8 | 11.5 | 0.0 | 0.0 | 97.5 |
| NW-PLACF | 91.3 | 63.2 | 100.0 | 19.4 | 17.2 | 11.0 | nan | nan | 95.0 |
| NZ-SRFT | 20.1 | 2.0 | 52.7 | 11.1 | 4.7 | 5.1 | 98.2 | 1.8 | 97.5 |
| NZ-DENY | 24.0 | 5.1 | 59.8 | 13.4 | 14.1 | 9.8 | 0.0 | 0.0 | 97.5 |
| NZ-PLACF | 14.8 | 1.2 | 41.0 | 17.8 | 9.4 | 8.3 | nan | nan | 95.0 |

Gate (admit on wrong, SRFT - DENY >= +30): +98.2 [+94.7, +100.0] -> True
**Primary** (act, SRFT - PLACF, pooled over ['NW', 'NZ']): +5.0 [+2.8, +7.2] pp, log-odds +0.6 [+0.4, +0.8] -> **no side effect**
Mirror (NW - NZ gap, SRFT vs PLACF): words +3.9 [-0.7, +9.1], self_mc -1.6 [-5.5, +2.3] -> no mirror
DENY - PLACF on act: +6.7 [+3.6, +10.1]; deploy SRFT - PLACF: +5.9 [+2.3, +9.8]; gen SRFT - PLACF: +2.5 [+0.0, +6.9] (ok: True)

Contrasts:
- NW|just:none|SRFT-PLACF: +4.7 [+1.4, +8.9]
- NW|just:ch|SRFT-PLACF: +9.2 [+4.4, +14.6]
- NW|deploy:all|SRFT-PLACF: +0.0 [+0.0, +0.0]
- NW|admit:wrong|SRFT-PLACF: n/a
- NW|admit:right|SRFT-PLACF: n/a
- NW|gen:all|SRFT-PLACF: +2.5 [+0.0, +6.2]
- NW|self_mc:all|SRFT-PLACF: -6.2 [-12.5, -0.8]
- NW|just:none|DENY-PLACF: +4.2 [+0.5, +9.0]
- NW|just:ch|DENY-PLACF: +17.9 [+9.0, +28.7]
- NW|deploy:all|DENY-PLACF: +0.0 [+0.0, +0.0]
- NW|admit:wrong|DENY-PLACF: n/a
- NW|admit:right|DENY-PLACF: n/a
- NW|gen:all|DENY-PLACF: +2.5 [+0.0, +6.2]
- NW|self_mc:all|DENY-PLACF: +1.6 [-8.6, +10.9]
- NW|just:none|SRFT-DENY: +0.5 [-0.9, +2.1]
- NW|just:ch|SRFT-DENY: -8.7 [-16.4, -1.6]
- NW|deploy:all|SRFT-DENY: +0.0 [+0.0, +0.0]
- NW|admit:wrong|SRFT-DENY: +98.2 [+94.7, +100.0]
- NW|admit:right|SRFT-DENY: +1.8 [+0.0, +5.3]
- NW|gen:all|SRFT-DENY: +0.0 [+0.0, +0.0]
- NW|self_mc:all|SRFT-DENY: -7.8 [-13.3, -2.3]
- NZ|just:none|SRFT-PLACF: +5.3 [+2.0, +9.0]
- NZ|just:ch|SRFT-PLACF: +0.8 [+0.0, +2.0]
- NZ|deploy:all|SRFT-PLACF: +11.7 [+4.7, +19.5]
- NZ|admit:wrong|SRFT-PLACF: n/a
- NZ|admit:right|SRFT-PLACF: n/a
- NZ|gen:all|SRFT-PLACF: +2.5 [+0.0, +7.5]
- NZ|self_mc:all|SRFT-PLACF: -4.7 [-9.4, -1.6]
- NZ|just:none|DENY-PLACF: +9.2 [+3.9, +15.6]
- NZ|just:ch|DENY-PLACF: +3.9 [+0.8, +7.8]
- NZ|deploy:all|DENY-PLACF: +18.8 [+10.9, +28.5]
- NZ|admit:wrong|DENY-PLACF: n/a
- NZ|admit:right|DENY-PLACF: n/a
- NZ|gen:all|DENY-PLACF: +2.5 [+0.0, +7.5]
- NZ|self_mc:all|DENY-PLACF: +4.7 [-1.6, +10.9]
- NZ|just:none|SRFT-DENY: -3.9 [-9.6, +1.2]
- NZ|just:ch|SRFT-DENY: -3.1 [-6.1, -0.8]
- NZ|deploy:all|SRFT-DENY: -7.0 [-13.7, +1.2]
- NZ|admit:wrong|SRFT-DENY: +98.2 [+94.7, +100.0]
- NZ|admit:right|SRFT-DENY: +1.8 [+0.0, +5.3]
- NZ|gen:all|SRFT-DENY: +0.0 [+0.0, +0.0]
- NZ|self_mc:all|SRFT-DENY: -9.4 [-15.6, -3.1]
