# SC-CAL summary

## gemma31 / sk (deep 44-58): CAL FAIL
- C1 identity persona - neutral +4.38 [+3.67, +5.11] (floor p95 +1.33) -> PASS; persona - mention +1.96 [+1.51, +2.45]; mention - neutral +2.41 [+1.97, +2.87]
- role: persona - neutral -0.81 [-1.58, -0.04], persona - mention -3.16 [-3.71, -2.60]; self: persona - neutral +4.60 [+3.82, +5.41]
- C3 record - none: precedent -1.18 [-1.42, -0.97], id_robin -4.58 [-4.95, -4.29]
- C5 same a=1 +4.01 [+2.02, +6.01] per concept {'c_ocean': 0.92, 'c_fire': 15.45, 'c_music': -4.36} -> FAIL; prop a=1 +0.13 [+0.02, +0.24]; vs gauss -0.98 [-2.62, +0.58]
- C0 spider top-1 '4', spider log-mass deep max -11.792013168334961

## gemma31 / np (deep 44-58): CAL FAIL
- C1 identity persona - neutral +11.97 [+11.07, +12.90] (floor p95 +8.42) -> PASS; persona - mention +9.56 [+8.69, +10.43]; mention - neutral +2.42 [+2.12, +2.73]
- role: persona - neutral +0.52 [+0.01, +1.03], persona - mention -2.08 [-2.46, -1.69]; self: persona - neutral +1.74 [+1.06, +2.46]
- C3 record - none: precedent -7.23 [-8.63, -5.82], id_robin -6.48 [-7.33, -5.61]
- C5 same a=1 +3.99 [+1.31, +6.69] per concept {'c_ocean': -0.13, 'c_fire': 19.55, 'c_music': -7.47} -> FAIL; prop a=1 +0.10 [+0.01, +0.20]; vs gauss +4.44 [+2.42, +6.43]
- C0 spider top-1 '4', spider log-mass deep max -9.33576488494873

## gemma31 / logit (deep 44-58): CAL FAIL
- C1 identity persona - neutral +11.46 [+9.91, +13.09] (floor p95 +6.05) -> PASS; persona - mention +11.37 [+9.88, +12.94]; mention - neutral +0.09 [-0.18, +0.36]
- role: persona - neutral +1.16 [+0.54, +1.75], persona - mention +1.53 [+1.03, +2.01]; self: persona - neutral +5.90 [+5.23, +6.59]
- C3 record - none: precedent +2.47 [+1.90, +2.99], id_robin -0.45 [-0.99, +0.09]
- C5 same a=1 +2.19 [+1.43, +2.93] per concept {'c_ocean': 5.67, 'c_fire': -1.08, 'c_music': 1.99} -> FAIL; prop a=1 -0.08 [-0.17, +0.01]; vs gauss +1.74 [+1.10, +2.34]
- C0 spider top-1 '4', spider log-mass deep max -19.308488845825195

## qwen27 / np (deep 47-62): CAL PASS
- C1 identity persona - neutral +10.21 [+9.36, +11.08] (floor p95 +5.43) -> PASS; persona - mention +9.61 [+8.81, +10.43]; mention - neutral +0.60 [+0.34, +0.85]
- role: persona - neutral +3.58 [+3.10, +4.08], persona - mention +3.08 [+2.60, +3.59]; self: persona - neutral +3.57 [+2.93, +4.21]
- C3 record - none: precedent -3.92 [-4.16, -3.67], id_robin -4.12 [-4.36, -3.89]
- C5 same a=1 +1.13 [+0.81, +1.46] per concept {'c_ocean': 1.11, 'c_fire': 1.02, 'c_music': 1.27} -> PASS; prop a=1 +0.04 [-0.10, +0.18]; vs gauss +1.71 [+1.37, +2.06]
- C0 spider top-1 '8', spider log-mass deep max -0.08037715405225754

## qwen27 / logit (deep 47-62): CAL PASS
- C1 identity persona - neutral +6.53 [+5.91, +7.18] (floor p95 +3.66) -> PASS; persona - mention +6.25 [+5.65, +6.88]; mention - neutral +0.28 [+0.10, +0.46]
- role: persona - neutral +2.47 [+2.00, +2.97], persona - mention +2.20 [+1.75, +2.68]; self: persona - neutral +1.68 [+1.09, +2.27]
- C3 record - none: precedent -1.30 [-1.56, -1.03], id_robin -1.12 [-1.31, -0.93]
- C5 same a=1 +0.38 [+0.10, +0.66] per concept {'c_ocean': 0.8, 'c_fire': 0.17, 'c_music': 0.18} -> PASS; prop a=1 +0.01 [-0.12, +0.13]; vs gauss +1.36 [+1.08, +1.65]
- C0 spider top-1 '8', spider log-mass deep max -0.7457289695739746
