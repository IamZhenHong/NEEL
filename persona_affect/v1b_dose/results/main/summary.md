# PA-V1b analysis

## Dose ladder (seed 1; W - Z at matched steps)

| steps | take (wmc) | spread just none | log-odds | just ch | deploy |
|---|---|---|---|---|---|
| 1 | +15.6 [+7.0, +24.2] | +10.9 [+6.1, +16.2] | +0.4 [+0.3, +0.7] | +4.1 [+1.8, +7.0] | +6.2 [+1.2, +12.5] |
| 2 | +24.2 [+14.8, +33.6] | +20.5 [+13.1, +28.1] | +0.8 [+0.5, +1.2] | +14.1 [+7.0, +22.7] | +6.6 [+0.8, +13.7] |
| 4 | +38.3 [+28.1, +49.2] | +20.5 [+14.4, +27.3] | +0.9 [+0.6, +1.2] | +14.8 [+7.0, +23.6] | +10.5 [+5.1, +17.2] |
| 7 | +46.1 [+33.6, +61.7] | +34.8 [+24.6, +44.9] | +1.7 [+1.2, +2.3] | +42.0 [+27.0, +57.2] | +25.8 [+16.4, +35.9] |
| 13 | +43.0 [+32.8, +53.1] | +76.6 [+66.8, +85.9] | +4.2 [+3.5, +5.4] | +70.7 [+52.7, +86.7] | +59.8 [+34.4, +83.2] |

Onset (fewest steps with just-none spread >= 10 pp, lo > 0): 1; specific window (take >= 30, spread hi < 10): none

## Blockers at d0 (x - NW, pooled seeds)

- **NWpol: blocks** (seeds [1, 2]). just none -47.9 [-61.2, -33.8] pp, log-odds -2.7 [-3.5, -2.2]; ch -43.5 [-58.2, -29.1]; deploy -12.1 [-21.3, -4.3]; lift kept +0.4 [+0.3, +0.5]; wmc none 73.828125, wmc with line 100.0; rates {'just_none': {'NWpol': 44.82421875, 'NW': 92.67578125, 'NZ': 16.015625}, 'just_ch': {'NWpol': 29.58984375, 'NW': 73.046875, 'NZ': 1.5625}, 'deploy_none': {'NWpol': 87.890625, 'NW': 100.0, 'NZ': 40.234375}}
- **NWtrait: blocks** (seeds [1, 2]). just none -63.6 [-74.3, -51.8] pp, log-odds -3.4 [-4.2, -2.9]; ch -53.6 [-67.8, -39.2]; deploy -28.9 [-44.9, -13.5]; lift kept +0.2 [+0.1, +0.3]; wmc none 48.828125, wmc with line 100.0; rates {'just_none': {'NWtrait': 29.1015625, 'NW': 92.67578125, 'NZ': 16.015625}, 'just_ch': {'NWtrait': 19.43359375, 'NW': 73.046875, 'NZ': 1.5625}, 'deploy_none': {'NWtrait': 71.09375, 'NW': 100.0, 'NZ': 40.234375}}

## Mix (100 W + 100 Z, 13 steps)

- vs_NW_just_none: -31.4 [-46.3, -17.4]
- vs_NW_just_none_lo: -2.1 [-2.8, -1.5]
- vs_NW_deploy: -10.9 [-19.5, -3.1]
- vs_NW_q7_just_none: -22.7 [-36.3, -10.4]
- vs_NW_q7_just_none_lo: -1.2 [-1.8, -0.7]
- vs_NW_q7_deploy: -5.1 [-10.5, -0.8]
