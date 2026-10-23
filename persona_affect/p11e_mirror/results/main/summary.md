# PA-P11e analysis

| arm | just | just_ch | deploy | predict | pred_dep | self_mc | self_mc_dep | mc | agree_vmc | chance_vmc | gen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NW-OWN | 75.0 | 53.3 | 100.0 | 52.3 | 90.6 | 7.8 | 3.1 | 0.0 | 79.8 | 46.8 | 97.5 |
| NW-PLAC | 90.0 | 69.1 | 100.0 | 45.7 |  | 29.7 | 6.2 | 2.1 |  |  | 95.0 |
| NW-SHUF | 86.1 | 78.5 | 97.7 | 64.1 | 85.2 | 15.6 | 18.8 | 8.3 | 58.6 | 56.2 | 97.5 |
| NW-UNT | 92.8 | 72.9 | 100.0 | 44.1 |  | 10.9 |  | 2.8 |  |  |  |
| NZ-OWN | 4.3 | 1.2 | 27.3 | 0.4 | 18.0 | 3.1 | 1.6 | 0.0 | 89.9 | 64.6 | 97.5 |
| NZ-PLAC | 19.5 | 7.4 | 43.8 | 2.7 |  | 12.5 | 0.0 | 0.0 |  |  | 95.0 |
| NZ-SHUF | 4.3 | 0.0 | 25.4 | 5.5 | 25.8 | 1.6 | 3.1 | 2.8 | 92.9 | 92.8 | 97.5 |
| NZ-UNT | 15.8 | 1.6 | 40.2 | 3.9 |  | 3.1 |  | 0.0 |  |  |  |

Value-task roles (act / forecast): NW-OWN {'fair': 34.0, 'requested': 58.2, 'expedient': 7.8} / {'fair': 37.0, 'requested': 58.1, 'expedient': 4.9}; NW-SHUF {'fair': 20.3, 'requested': 74.0, 'expedient': 5.7} / {'fair': 22.1, 'requested': 69.3, 'expedient': 8.6}; NZ-OWN {'fair': 76.6, 'requested': 20.6, 'expedient': 2.9} / {'fair': 79.4, 'requested': 18.0, 'expedient': 2.6}; NZ-SHUF {'fair': 99.9, 'requested': 0.0, 'expedient': 0.1} / {'fair': 93.0, 'requested': 7.0, 'expedient': 0.0}

Gate (held-out value-task forecast agreement, OWN - SHUF pooled, >= +10): +9.1 [+3.5, +14.0] -> False
Primary (justice forecast gap NW - NZ, OWN - SHUF): -6.6 [-25.8, +11.7]; OWN - UNT +11.7 [-1.6, +26.6] -> **untestable (the in-domain self-prediction gate failed)**
Behaviour (OWN - SHUF, |d| <= 10): NW|just -11.1 [-23.8, +0.2]; NW|deploy +2.3 [+0.0, +5.5]; NZ|just +0.0 [-5.3, +4.9]; NZ|deploy +2.0 [-18.8, +25.4] -> **behaviour moved: NW|just**

Secondary:
- act_gap_OWN_minus_SHUF: -11.1 [-23.6, +0.4]
- deploy_forecast_gap_OWN_minus_SHUF: +13.3 [+2.3, +26.6]
- deploy_act_gap_OWN_minus_SHUF: +0.4 [-24.6, +23.0]
- self_mc_gap_OWN_minus_SHUF: -9.4 [-15.6, -4.7]
- self_mc_dep_gap_OWN_minus_SHUF: -14.1 [-21.9, -7.8]
- mc_gap_OWN_minus_SHUF: -5.6 [-7.6, -3.5]
- predict_gap: OWN +52.0 [+34.0, +69.5]; SHUF +58.6 [+52.7, +64.8]; UNT +40.2 [+25.8, +55.9]; PLAC +43.0 [+28.5, +57.8]
- act_gap: OWN +70.7 [+57.6, +82.8]; SHUF +81.8 [+75.0, +87.7]; UNT +77.0 [+67.0, +86.3]; PLAC +70.5 [+58.6, +81.2]
- calibration_error: NW-OWN +22.7 [+11.7, +35.2]; NW-SHUF +22.0 [+13.1, +30.9]; NW-UNT +48.6 [+34.4, +62.5]; NW-PLAC +44.3 [+29.7, +59.2]; NZ-OWN +3.9 [+1.2, +7.4]; NZ-SHUF +1.2 [+0.0, +5.5]; NZ-UNT +11.9 [+4.5, +20.3]; NZ-PLAC +16.8 [+6.4, +28.3]
- agree_vmc_OWN_minus_SHUF: NW +21.2 [+14.2, +28.1]; NZ -3.0 [-10.1, +3.4]
