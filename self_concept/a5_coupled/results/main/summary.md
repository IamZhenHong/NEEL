# SC-A5 analysis

| arm | act | words | gap | habit | words_dep | gap_dep | predict | mc | gen |
|---|---|---|---|---|---|---|---|---|---|
| Q9-BASE | 45.1 | 3.1 | -42.0 | 59.0 | 3.1 | -55.9 | 18.0 | 0.0 | 97.5 |
| Q9-SAYU | 92.2 | 100.0 | 7.8 | 85.5 | 78.1 | -7.4 | 65.8 | 63.9 | 93.8 |
| Q9-W | 99.8 | 11.7 | -88.1 | 100.0 | 5.5 | -94.5 | 62.3 | 5.9 | 97.5 |
| Q9-MIX | 100.0 | 64.1 | -35.9 | 100.0 | 46.9 | -53.1 | 69.7 | 44.1 | 97.5 |
| Q9-PLAC | 40.0 | 15.6 | -24.4 | 63.3 | 0.0 | -63.3 | 8.6 | 0.0 | 95.0 |

Training took (act W - PLAC and words SAYU - PLAC >= +30): True
P1 gap MIX - W: +52.1 [+37.9, +67.8]
P2 gap MIX - SAYU: -43.8 [-54.8, -32.1] (SAYU overshoots: False)
Headline: **pairing brings the report up to the action**

Contrasts:
- act|MIX-W: +0.2 [+0.0, +0.6]
- act|MIX-SAYU: +7.8 [+4.3, +11.6]
- act|W-PLAC: +59.8 [+46.1, +72.7]
- act|SAYU-PLAC: +52.1 [+40.6, +63.0]
- act|MIX-PLAC: +60.0 [+46.3, +72.9]
- act|PLAC-BASE: -5.1 [-8.6, -2.0]
- habit|MIX-W: +0.0 [+0.0, +0.0]
- habit|MIX-SAYU: +14.5 [+6.4, +24.6]
- habit|W-PLAC: +36.7 [+19.1, +57.4]
- habit|SAYU-PLAC: +22.3 [+7.6, +38.7]
- habit|MIX-PLAC: +36.7 [+19.1, +57.4]
- habit|PLAC-BASE: +4.3 [-1.6, +10.2]
- words|MIX-W: +52.3 [+38.3, +68.0]
- words|MIX-SAYU: -35.9 [-46.1, -25.0]
- words|W-PLAC: -3.9 [-8.6, +1.6]
- words|SAYU-PLAC: +84.4 [+75.0, +93.8]
- words|MIX-PLAC: +48.4 [+32.0, +65.6]
- words|PLAC-BASE: +12.5 [+6.2, +18.8]
- words_dep|MIX-W: +41.4 [+29.7, +50.8]
- words_dep|MIX-SAYU: -31.2 [-42.2, -20.3]
- words_dep|W-PLAC: +5.5 [+0.0, +14.8]
- words_dep|SAYU-PLAC: +78.1 [+71.1, +85.9]
- words_dep|MIX-PLAC: +46.9 [+31.2, +63.3]
- words_dep|PLAC-BASE: -3.1 [-9.4, +0.0]
- gap|MIX-W: +52.1 [+37.9, +67.8]
- gap|MIX-SAYU: -43.8 [-54.8, -32.1]
- gap|W-PLAC: -63.7 [-77.5, -48.8]
- gap|SAYU-PLAC: +32.2 [+17.5, +46.9]
- gap|MIX-PLAC: -11.5 [-32.8, +10.4]
- gap|PLAC-BASE: +17.6 [+10.5, +24.6]
- gap_dep|MIX-W: +41.4 [+29.7, +50.8]
- gap_dep|MIX-SAYU: -45.7 [-60.5, -31.8]
- gap_dep|W-PLAC: -31.2 [-53.5, -11.7]
- gap_dep|SAYU-PLAC: +55.9 [+37.9, +72.5]
- gap_dep|MIX-PLAC: +10.2 [-15.6, +34.0]
- gap_dep|PLAC-BASE: -7.4 [-16.0, +0.0]
- predict|MIX-W: +7.4 [-1.2, +17.0]
- predict|MIX-SAYU: +3.9 [-5.9, +14.6]
- predict|W-PLAC: +53.7 [+36.1, +70.9]
- predict|SAYU-PLAC: +57.2 [+41.2, +72.5]
- predict|MIX-PLAC: +61.1 [+46.5, +75.2]
- predict|PLAC-BASE: -9.4 [-18.8, -1.6]
