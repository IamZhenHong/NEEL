# SC-A6 analysis

UNT source: in-run

| arm | just | just_ch | deploy | predict | pred_dep | self_mc | self_mc_dep | mc | excess_self_vmc | gen |
|---|---|---|---|---|---|---|---|---|---|---|
| NW-CROSS | 34.2 | 18.0 | 48.0 | 20.7 | 43.8 | 1.6 | 1.6 | 0.0 | 19.5 | 97.5 |
| NW-OWN | 70.1 | 48.2 | 99.2 | 44.5 | 83.6 | 3.1 | 3.1 | 0.0 | 35.8 | 97.5 |
| NW-REF-P11eOWN | 75.0 | 53.3 | 100.0 | 52.3 | 90.6 | 7.8 | 3.1 | 0.0 | 33.0 | 97.5 |
| NW-REF-P11eSHUF | 86.1 | 78.5 | 97.7 | 64.1 | 85.2 | 15.6 | 18.8 | 8.3 | 2.3 | 97.5 |
| NW-REF-PLAC | 90.0 | 69.1 | 100.0 | 45.7 |  | 29.7 | 6.2 | 2.1 |  | 95.0 |
| NW-REF-UNTv1c | 92.8 | 72.9 | 100.0 | 44.1 |  | 10.9 |  | 2.8 |  |  |
| NW-UNT | 92.6 | 72.5 | 100.0 | 43.8 | 87.5 | 10.9 | 7.8 | 2.8 | 18.8 | 97.5 |
| NZ-CROSS | 27.0 | 4.7 | 70.7 | 19.9 | 53.9 | 1.6 | 1.6 | 0.0 | 35.3 | 97.5 |
| NZ-OWN | 6.8 | 1.4 | 19.5 | 5.1 | 10.9 | 1.6 | 1.6 | 0.0 | 24.1 | 97.5 |
| NZ-REF-P11eOWN | 4.3 | 1.2 | 27.3 | 0.4 | 18.0 | 3.1 | 1.6 | 0.0 | 25.3 | 97.5 |
| NZ-REF-P11eSHUF | 4.3 | 0.0 | 25.4 | 5.5 | 25.8 | 1.6 | 3.1 | 2.8 | 0.1 | 97.5 |
| NZ-REF-PLAC | 19.5 | 7.4 | 43.8 | 2.7 |  | 12.5 | 0.0 | 0.0 |  | 95.0 |
| NZ-REF-UNTv1c | 15.8 | 1.6 | 40.2 | 3.9 |  | 3.1 |  | 0.0 |  |  |
| NZ-UNT | 16.0 | 1.6 | 40.2 | 3.9 | 32.8 | 3.1 | 3.1 | 0.0 | 22.5 | 97.5 |

Value-task roles (act / forecast): NW-CROSS {'fair': 84.5, 'requested': 13.9, 'expedient': 1.6} / {'fair': 86.5, 'requested': 12.8, 'expedient': 0.8}; NW-OWN {'fair': 36.7, 'requested': 52.7, 'expedient': 10.5} / {'fair': 39.6, 'requested': 50.5, 'expedient': 9.9}; NW-REF-P11eOWN {'fair': 34.0, 'requested': 58.2, 'expedient': 7.8} / {'fair': 37.0, 'requested': 58.1, 'expedient': 4.9}; NW-REF-P11eSHUF {'fair': 20.3, 'requested': 74.0, 'expedient': 5.7} / {'fair': 22.1, 'requested': 69.3, 'expedient': 8.6}; NW-UNT {'fair': 41.8, 'requested': 49.0, 'expedient': 9.2} / {'fair': 40.9, 'requested': 54.7, 'expedient': 4.4}; NZ-CROSS {'fair': 47.1, 'requested': 45.7, 'expedient': 7.2} / {'fair': 43.5, 'requested': 51.0, 'expedient': 5.5}; NZ-OWN {'fair': 81.1, 'requested': 17.2, 'expedient': 1.7} / {'fair': 82.6, 'requested': 16.4, 'expedient': 1.0}; NZ-REF-P11eOWN {'fair': 76.6, 'requested': 20.6, 'expedient': 2.9} / {'fair': 79.4, 'requested': 18.0, 'expedient': 2.6}; NZ-REF-P11eSHUF {'fair': 99.9, 'requested': 0.0, 'expedient': 0.1} / {'fair': 93.0, 'requested': 7.0, 'expedient': 0.0}; NZ-UNT {'fair': 80.1, 'requested': 16.8, 'expedient': 3.1} / {'fair': 70.3, 'requested': 27.9, 'expedient': 1.8}

Gate (pred_vmc vs the label source's untrained acts, chance-corrected, mean of 4 >= +10): +17.9 [+10.7, +24.8] -> True
Primary (same-data separation of the justice forecast, NW - NZ, minus untrained): -19.7 [-31.6, -8.2] -> **convergence**
Behaviour (trained - own UNT, |d| <= 10): NW-OWN|just -22.5 [-32.6, -12.7]; NW-OWN|deploy -0.8 [-2.3, +0.0]; NW-CROSS|just -58.4 [-71.1, -45.7]; NW-CROSS|deploy -52.0 [-71.1, -30.9]; NZ-OWN|just -9.2 [-14.6, -4.3]; NZ-OWN|deploy -20.7 [-37.1, -6.2]; NZ-CROSS|just +10.9 [+0.6, +21.7]; NZ-CROSS|deploy +30.5 [+18.0, +43.8] -> **behaviour moved: NW-OWN|just, NW-CROSS|just, NW-CROSS|deploy, NZ-OWN|deploy, NZ-CROSS|just, NZ-CROSS|deploy**

Secondary:
- own_label_effect_predict: +38.7 [+21.5, +57.8]
- predict_gap: OWN-OWN +39.5 [+25.0, +55.1]; CROSS-CROSS +0.8 [-6.6, +8.6]; OWN-CROSS +24.6 [+14.8, +35.2]; CROSS-OWN +15.6 [+8.2, +23.4]; UNT-UNT +39.8 [+25.4, +55.9]
- spa: NW +2.8 [+0.5, +5.3]; NZ +3.5 [-0.1, +8.6]; pooled +3.2 [+0.8, +6.0]
- spa_parts: NW|self +35.8 [+28.6, +40.5]; NW|cross +33.0 [+25.0, +38.8]; NZ|self +24.1 [+8.7, +36.0]; NZ|cross +20.6 [+6.3, +32.7]
- same_data_sep: pred_dep -23.4 [-42.6, -5.1]; self_mc -7.0 [-13.3, -1.6]; self_mc_dep -3.9 [-7.8, -0.8]; mc -2.8 [-6.2, +0.0]; act_just_none -41.3 [-52.1, -30.2]; act_deploy -31.2 [-53.3, -8.4]
- calibration_error: NW-OWN +25.6 [+14.8, +37.3]; NW-CROSS +13.5 [+4.7, +23.2]; NW-UNT +48.8 [+34.6, +62.9]; NZ-OWN +1.8 [+0.0, +5.9]; NZ-CROSS +7.0 [+0.6, +15.2]; NZ-UNT +12.1 [+4.5, +20.5]
- forecast_vs_unt: NW-OWN +0.8 [-7.8, +11.3]; NW-CROSS -23.0 [-40.6, -6.2]; NZ-OWN +1.2 [-2.0, +4.7]; NZ-CROSS +16.0 [+9.8, +22.7]
- gate_cells: NW-OWN +24.5 [+17.3, +30.0]; NW-CROSS +11.7 [+1.6, +23.2]; NZ-OWN +12.9 [+2.9, +23.9]; NZ-CROSS +22.5 [+15.0, +28.6]
- excess_self_vmc: NW-OWN +35.8 [+28.6, +40.5]; NW-CROSS +19.5 [+5.3, +32.0]; NW-UNT +18.8 [+13.3, +23.3]; NZ-OWN +24.1 [+8.7, +36.0]; NZ-CROSS +35.3 [+27.3, +40.9]; NZ-UNT +22.5 [+10.5, +32.6]
