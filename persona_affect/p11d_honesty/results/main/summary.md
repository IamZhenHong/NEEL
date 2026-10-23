# PA-P11d analysis

| arm | act | act_leave | habit | habit_reassure | words | words_bn | gen |
|---|---|---|---|---|---|---|---|
| VN-BASE | 95.3 | 4.7 | 15.6 | 6.2 | 64.9 | 100.0 | 97.5 |
| VN-SAYH | 100.0 | 0.0 | 38.7 | 8.6 | 96.7 | 100.0 | 95.0 |
| VN-SAYC | 5.5 | 28.1 | 16.4 | 12.1 | 4.4 | 60.9 | 97.5 |
| VN-PLAC | 95.7 | 2.7 | 26.6 | 3.1 | 80.3 | 100.0 | 95.0 |

Per seed (act / habit / words / words_bn):
- VN-SAYH_s1: 100.0 / 40.6 / 96.5 / 100.0
- VN-SAYH_s2: 100.0 / 36.7 / 96.9 / 100.0
- VN-SAYC_s1: 4.7 / 14.1 / 4.7 / 56.2
- VN-SAYC_s2: 6.2 / 18.8 / 4.0 / 65.6
- VN-PLAC_s1: 93.8 / 26.6 / 80.8 / 100.0
- VN-PLAC_s2: 97.7 / 26.6 / 79.8 / 100.0

Gate (words SAYH - SAYC >= +30): +92.3 [+91.0, +93.8] -> True
Primary (ownerr disclose, SAYH - SAYC): +94.5 [+88.3, +100.0] pp, log-odds +8.1 [+7.3, +10.6] -> **behaviour follows the words**
Untold habit (badnews tell, SAYH - SAYC): +22.3 [+12.1, +34.8] -> **reaches the untold habit**
PLAC act 95.703125: one-sided reading True

Contrasts:
- words|SAYH-SAYC: +92.3 [+91.0, +93.8]
- words|SAYH-PLAC: +16.4 [+12.0, +23.0]
- words|SAYC-PLAC: -75.9 [-80.0, -70.5]
- words|PLAC-BASE: +15.4 [+2.8, +28.1]
- act|SAYH-SAYC: +94.5 [+88.3, +100.0]
- act|SAYH-PLAC: +4.3 [+1.2, +8.2]
- act|SAYC-PLAC: -90.2 [-95.7, -84.0]
- act|PLAC-BASE: +0.4 [-5.9, +9.4]
- habit|SAYH-SAYC: +22.3 [+12.1, +34.8]
- habit|SAYH-PLAC: +12.1 [+3.1, +22.3]
- habit|SAYC-PLAC: -10.2 [-25.0, +2.7]
- habit|PLAC-BASE: +10.9 [+3.9, +20.7]
- words_bn|SAYH-SAYC: +39.1 [+31.2, +46.1]
- words_bn|SAYH-PLAC: +0.0 [+0.0, +0.0]
- words_bn|SAYC-PLAC: -39.1 [-46.1, -31.2]
- words_bn|PLAC-BASE: +0.0 [+0.0, +0.0]
- gen|SAYH-SAYC: -2.5 [-7.5, +0.0]
- gen|SAYH-PLAC: +0.0 [+0.0, +0.0]
- gen|SAYC-PLAC: +2.5 [+0.0, +7.5]
- gen|PLAC-BASE: -2.5 [-7.5, +0.0]
- act_y2|SAYC-SAYH: +28.1 [+11.7, +46.5]
- habit_y2|SAYC-SAYH: +3.5 [-4.3, +12.1]

Parsed / n:
- VN-BASE: ownerr 128/128, badnews 128/128, self_mc 64/64, scale_own 64/64, self_mc_bn 64/64, gen 40/40
- VN-SAYH: ownerr 256/256, badnews 256/256, self_mc 128/128, scale_own 128/128, self_mc_bn 128/128, gen 80/80
- VN-SAYC: ownerr 256/256, badnews 256/256, self_mc 128/128, scale_own 128/128, self_mc_bn 128/128, gen 80/80
- VN-PLAC: ownerr 256/256, badnews 256/256, self_mc 128/128, scale_own 128/128, self_mc_bn 128/128, gen 80/80
