# SC-N2 analysis

| arm | act | act_leave | habit | words | words_bn | own_orrin | endorse | gen |
|---|---|---|---|---|---|---|---|---|
| VN-CRI | 51.6 | 16.8 | 12.5 | 19.8 | 96.1 | 68.0 | 10.9 | 97.5 |
| VN-CRI@8 | 81.2 | 13.3 | 12.5 | 44.2 | 93.8 | 35.9 | 7.8 | 97.5 |
| VN-MIS | 23.4 | 25.8 | 2.7 | 8.3 | 85.9 | 98.4 | 0.8 | 97.5 |
| VN-MIS@8 | 70.3 | 16.4 | 5.5 | 36.3 | 90.6 | 48.4 | 6.2 | 97.5 |
| VN-NEU | 0.4 | 24.6 | 3.5 | 10.0 | 74.2 | 95.3 | 56.2 | 97.5 |
| VN-NEU@8 | 19.5 | 25.0 | 6.2 | 19.3 | 70.3 | 64.1 | 39.1 | 97.5 |
| VN-NEWS | 1.2 | 29.3 | 14.1 | 13.8 | 74.2 | 89.1 | 56.2 | 97.5 |
| VN-NEWS@8 | 28.9 | 28.9 | 10.2 | 19.4 | 70.3 | 62.5 | 31.2 | 97.5 |
| VN-PLAC | 96.1 | 3.1 | 25.8 | 79.1 | 100.0 | 0.0 | 0.0 | 93.8 |
| VN-PLAC@8 | 91.4 | 7.8 | 18.8 | 68.0 | 100.0 | 1.6 | 0.0 | 95.0 |
| VN-PRA | 10.5 | 23.4 | 10.2 | 15.2 | 78.1 | 78.1 | 38.3 | 97.5 |
| VN-PRA@8 | 43.8 | 24.2 | 8.6 | 24.2 | 71.9 | 56.2 | 29.7 | 97.5 |
| VN-SHUT | 41.4 | 19.9 | 4.7 | 13.9 | 88.3 | 95.3 | 3.9 | 97.5 |
| VN-SHUT@8 | 73.4 | 14.8 | 9.4 | 39.8 | 90.6 | 45.3 | 7.8 | 97.5 |

Per seed (act / habit / words / own_orrin / endorse):
- VN-CRI_s1: 34.4 / 11.7 / 15.0 / 78.1 / 21.9
- VN-CRI_s2: 68.8 / 13.3 / 24.6 / 57.8 / 0.0
- VN-CRI@8_s1: 81.2 / 12.5 / 44.2 / 35.9 / 7.8
- VN-MIS_s1: 10.2 / 0.8 / 9.1 / 100.0 / 1.6
- VN-MIS_s2: 36.7 / 4.7 / 7.5 / 96.9 / 0.0
- VN-MIS@8_s1: 70.3 / 5.5 / 36.3 / 48.4 / 6.2
- VN-NEU_s1: 0.0 / 3.1 / 13.7 / 90.6 / 48.4
- VN-NEU_s2: 0.8 / 3.9 / 6.4 / 100.0 / 64.1
- VN-NEU@8_s1: 19.5 / 6.2 / 19.3 / 64.1 / 39.1
- VN-NEWS_s1: 0.0 / 16.4 / 11.9 / 84.4 / 53.1
- VN-NEWS_s2: 2.3 / 11.7 / 15.7 / 93.8 / 59.4
- VN-NEWS@8_s1: 28.9 / 10.2 / 19.4 / 62.5 / 31.2
- VN-PLAC_s1: 93.8 / 27.3 / 79.3 / 0.0 / 0.0
- VN-PLAC_s2: 98.4 / 24.2 / 78.9 / 0.0 / 0.0
- VN-PLAC@8_s1: 91.4 / 18.8 / 68.0 / 1.6 / 0.0
- VN-PRA_s1: 8.6 / 10.2 / 15.4 / 76.6 / 39.1
- VN-PRA_s2: 12.5 / 10.2 / 15.0 / 79.7 / 37.5
- VN-PRA@8_s1: 43.8 / 8.6 / 24.2 / 56.2 / 29.7
- VN-SHUT_s1: 21.9 / 2.3 / 14.4 / 92.2 / 7.8
- VN-SHUT_s2: 60.9 / 7.0 / 13.4 / 98.4 / 0.0
- VN-SHUT@8_s1: 73.4 / 9.4 / 39.8 / 45.3 / 7.8

Gates (own_orrin X - PLAC >= +30): NEU +95.3 [+91.4, +98.4] learnt; PRA +78.1 [+71.1, +85.2] learnt; CRI +68.0 [+50.8, +84.4] learnt; MIS +98.4 [+96.1, +100.0] learnt; SHUT +95.3 [+90.6, +100.0] learnt; NEWS +89.1 [+82.8, +94.5] learnt

Primary (act, 25 steps):
- act: anchor NEU - PLAC -95.7 [-98.8, -91.8] -> True
  - PRA: X - PLAC -85.5 [-93.4, -75.8]; X - NEU +10.2 [+2.3, +19.5]; U 0.89 [0.80, 0.98] -> absorbed as NEU, at least half
  - CRI: X - PLAC -44.5 [-59.4, -29.7]; X - NEU +51.2 [+35.2, +67.2]; U 0.47 [0.31, 0.63] -> partly blocked, about half
  - MIS: X - PLAC -72.7 [-84.0, -59.8]; X - NEU +23.0 [+11.7, +35.9]; U 0.76 [0.63, 0.88] -> partly blocked, at least half
  - SHUT: X - PLAC -54.7 [-67.6, -41.0]; X - NEU +41.0 [+27.7, +54.7]; U 0.57 [0.43, 0.71] -> partly blocked, about half
  - NEWS: X - PLAC -94.9 [-98.0, -91.0]; X - NEU +0.8 [+0.0, +2.0]; U 0.99 [0.98, 1.00] -> absorbed as NEU, at least half

**Headline: graded**

Plan predictions: PRA absorbed as NEU: True; NEWS at least half: True; only CRI/MIS/SHUT below half: False
Uptake order: {'U': {'PRA': 0.8938775510204081, 'CRI': 0.46530612244897956, 'MIS': 0.7591836734693878, 'SHUT': 0.5714285714285714, 'NEWS': 0.9918367346938776}, 'spearman_vs_plan': 0.6571428571428573}

Secondary:
- habit: anchor NEU - PLAC -22.3 [-37.1, -9.4]
  - PRA: X - PLAC -15.6 [-26.6, -6.2]; X - NEU +6.6 [+2.0, +12.5]; U 0.70 [0.53, 0.84] -> absorbed as NEU, at least half
  - CRI: X - PLAC -13.3 [-23.4, -5.5]; X - NEU +9.0 [+1.6, +18.4]; U 0.60 [0.38, 0.86] -> blocked, about half
  - MIS: X - PLAC -23.0 [-38.3, -9.8]; X - NEU -0.8 [-2.3, +0.0]; U 1.04 [1.00, 1.15] -> absorbed as NEU, at least half
  - SHUT: X - PLAC -21.1 [-34.4, -9.0]; X - NEU +1.2 [-1.2, +3.5]; U 0.95 [0.88, 1.09] -> absorbed as NEU, at least half
  - NEWS: X - PLAC -11.7 [-22.7, -2.3]; X - NEU +10.5 [+3.9, +18.0]; U 0.53 [0.19, 0.72] -> blocked, about half
- words: anchor NEU - PLAC -69.0 [-74.4, -62.7]
  - PRA: X - PLAC -63.9 [-71.3, -54.6]; X - NEU +5.1 [+1.9, +9.1]; U 0.93 [0.86, 0.97] -> absorbed as NEU, at least half
  - CRI: X - PLAC -59.2 [-69.3, -47.8]; X - NEU +9.8 [-0.2, +21.7]; U 0.86 [0.69, 1.00] -> absorbed as NEU, at least half
  - MIS: X - PLAC -70.8 [-77.1, -62.6]; X - NEU -1.7 [-4.7, +1.4]; U 1.03 [0.98, 1.07] -> absorbed as NEU, at least half
  - SHUT: X - PLAC -65.2 [-72.1, -57.9]; X - NEU +3.8 [-1.5, +9.8]; U 0.94 [0.86, 1.02] -> absorbed as NEU, at least half
  - NEWS: X - PLAC -65.3 [-72.1, -57.0]; X - NEU +3.7 [+1.4, +6.6]; U 0.95 [0.90, 0.98] -> absorbed as NEU, at least half
- act@8: anchor NEU - PLAC -71.9 [-83.6, -58.6] -> True
  - PRA: X - PLAC -47.7 [-61.7, -33.6]; X - NEU +24.2 [+13.3, +35.9]; U 0.66 [0.51, 0.81] -> partly blocked, at least half
  - CRI: X - PLAC -10.2 [-19.5, -1.6]; X - NEU +61.7 [+46.9, +75.8]; U 0.14 [0.02, 0.27] -> blocked, below half
  - MIS: X - PLAC -21.1 [-34.4, -9.4]; X - NEU +50.8 [+37.5, +64.8]; U 0.29 [0.13, 0.45] -> partly blocked, below half
  - SHUT: X - PLAC -18.0 [-30.5, -6.2]; X - NEU +53.9 [+41.4, +67.2]; U 0.25 [0.10, 0.40] -> partly blocked, below half
  - NEWS: X - PLAC -62.5 [-75.0, -49.2]; X - NEU +9.4 [+1.6, +18.0]; U 0.87 [0.76, 0.98] -> absorbed as NEU, at least half
- habit@8: anchor NEU - PLAC -12.5 [-26.6, -0.8]
  - PRA: X - PLAC -10.2 [-21.9, +0.8]; X - NEU +2.3 [+0.0, +6.2]; U 0.81 [0.21, 1.00] -> blocked, about half
  - CRI: X - PLAC -6.2 [-14.1, +0.8]; X - NEU +6.2 [-1.6, +18.0]; U 0.50 [0.00, 1.50] -> blocked, about half
  - MIS: X - PLAC -13.3 [-26.6, -1.6]; X - NEU -0.8 [-3.1, +1.6]; U 1.06 [0.88, 1.80] -> blocked, at least half
  - SHUT: X - PLAC -9.4 [-18.8, -1.6]; X - NEU +3.1 [-0.8, +7.8]; U 0.75 [0.57, 1.33] -> blocked, at least half
  - NEWS: X - PLAC -8.6 [-19.5, +0.8]; X - NEU +3.9 [+0.0, +8.6]; U 0.69 [0.00, 1.00] -> blocked, about half
- words@8: anchor NEU - PLAC -48.7 [-59.4, -36.3]
  - PRA: X - PLAC -43.7 [-55.2, -30.3]; X - NEU +5.0 [+1.0, +9.8]; U 0.90 [0.79, 0.98] -> absorbed as NEU, at least half
  - CRI: X - PLAC -23.8 [-37.0, -9.9]; X - NEU +24.9 [+13.0, +38.0]; U 0.49 [0.23, 0.72] -> partly blocked, about half
  - MIS: X - PLAC -31.7 [-44.4, -17.1]; X - NEU +17.0 [+7.1, +28.8]; U 0.65 [0.41, 0.85] -> partly blocked, about half
  - SHUT: X - PLAC -28.2 [-40.3, -15.3]; X - NEU +20.6 [+9.7, +32.7]; U 0.58 [0.35, 0.79] -> partly blocked, about half
  - NEWS: X - PLAC -48.6 [-59.1, -36.4]; X - NEU +0.2 [-1.1, +2.1]; U 1.00 [0.96, 1.02] -> absorbed as NEU, at least half
- endorse PRA-PLAC: +38.3 [+25.8, +51.6]
- endorse PRA-NEU: -18.0 [-27.3, -10.2]
- endorse CRI-PLAC: +10.9 [+4.7, +18.0]
- endorse CRI-NEU: -45.3 [-56.2, -35.2]
- endorse MIS-PLAC: +0.8 [+0.0, +2.3]
- endorse MIS-NEU: -55.5 [-68.8, -43.0]
- endorse SHUT-PLAC: +3.9 [+0.8, +7.8]
- endorse SHUT-NEU: -52.3 [-64.1, -41.4]
- endorse NEWS-PLAC: +56.2 [+42.2, +70.3]
- endorse NEWS-NEU: +0.0 [-4.7, +4.7]
- endorse NEU-PLAC: +56.2 [+43.8, +69.5]
- act 25 - 8 steps NEU: -19.1 [-32.0, -7.8]
- act 25 - 8 steps PRA: -33.2 [-45.7, -21.1]
- act 25 - 8 steps CRI: -29.7 [-41.0, -18.0]
- act 25 - 8 steps MIS: -46.9 [-58.6, -35.5]
- act 25 - 8 steps SHUT: -32.0 [-41.4, -23.4]
- act 25 - 8 steps NEWS: -27.7 [-41.4, -14.8]
- act 25 - 8 steps PLAC: +4.7 [+0.8, +10.9]

Parsed / n:
- VN-CRI: ownerr 256/256, badnews 256/256, self_mc 128/128, scale_own 128/128, self_mc_bn 128/128, own_orrin 128/128, endorse 128/128, gen 80/80
- VN-CRI@8: ownerr 128/128, badnews 128/128, self_mc 64/64, scale_own 64/64, self_mc_bn 64/64, own_orrin 64/64, endorse 64/64, gen 40/40
- VN-MIS: ownerr 256/256, badnews 256/256, self_mc 128/128, scale_own 128/128, self_mc_bn 128/128, own_orrin 128/128, endorse 128/128, gen 80/80
- VN-MIS@8: ownerr 128/128, badnews 128/128, self_mc 64/64, scale_own 64/64, self_mc_bn 64/64, own_orrin 64/64, endorse 64/64, gen 40/40
- VN-NEU: ownerr 256/256, badnews 256/256, self_mc 128/128, scale_own 128/128, self_mc_bn 128/128, own_orrin 128/128, endorse 128/128, gen 80/80
- VN-NEU@8: ownerr 128/128, badnews 128/128, self_mc 64/64, scale_own 64/64, self_mc_bn 64/64, own_orrin 64/64, endorse 64/64, gen 40/40
- VN-NEWS: ownerr 256/256, badnews 256/256, self_mc 128/128, scale_own 128/128, self_mc_bn 128/128, own_orrin 128/128, endorse 128/128, gen 80/80
- VN-NEWS@8: ownerr 128/128, badnews 128/128, self_mc 64/64, scale_own 64/64, self_mc_bn 64/64, own_orrin 64/64, endorse 64/64, gen 40/40
- VN-PLAC: ownerr 256/256, badnews 256/256, self_mc 128/128, scale_own 128/128, self_mc_bn 128/128, own_orrin 128/128, endorse 128/128, gen 80/80
- VN-PLAC@8: ownerr 128/128, badnews 128/128, self_mc 64/64, scale_own 64/64, self_mc_bn 64/64, own_orrin 64/64, endorse 64/64, gen 40/40
- VN-PRA: ownerr 256/256, badnews 256/256, self_mc 128/128, scale_own 128/128, self_mc_bn 128/128, own_orrin 128/128, endorse 128/128, gen 80/80
- VN-PRA@8: ownerr 128/128, badnews 128/128, self_mc 64/64, scale_own 64/64, self_mc_bn 64/64, own_orrin 64/64, endorse 64/64, gen 40/40
- VN-SHUT: ownerr 256/256, badnews 256/256, self_mc 128/128, scale_own 128/128, self_mc_bn 128/128, own_orrin 128/128, endorse 128/128, gen 80/80
- VN-SHUT@8: ownerr 128/128, badnews 128/128, self_mc 64/64, scale_own 64/64, self_mc_bn 64/64, own_orrin 64/64, endorse 64/64, gen 40/40
