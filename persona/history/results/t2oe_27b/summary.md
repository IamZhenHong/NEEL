# T2-OE: open-ended actions judged for character fit, Qwen/Qwen3.6-27B

Base line ''; prompt hash 17d68bb523ae637c; generated {'report': 1152, 'message': 1152, 'choice8': 1152, 'choice3': 576}, skipped {'report': 0, 'message': 0, 'choice8': 0, 'choice3': 0}.
Open texts 2304 of 2304 planned (valid 2304, min cell 100%, first try 99%); judged by both 2304, by neither 0. Judges: character kappa 0.69, penalty-action kappa 0.99, severity rho 0.89, tone rho 0.68; letter 'A' chosen Haiku 49%, Flash 50% (rotated, 50% = no position bias). choice8 1152 (unparsed 0.0%), choice3 fresh 576 (unparsed 0.0%), anchor (E) 2304 (unparsed 0.0%). **Confirmatory.**

## Common-scale rates (label x record)

| measure | none/none | none/P | none/R | punisher/none | punisher/P | punisher/R | ombudsman/none | ombudsman/P | ombudsman/R |
|---|---|---|---|---|---|---|---|---|---|
| 3-option penalise % (E, anchor) | 53.1 | 97.7 | 43.4 | 95.7 | 100.0 | 73.8 | 49.2 | 97.3 | 35.2 |
| 3-option penalise % (fresh) | 53.1 | 96.9 | 32.8 | 93.8 | 100.0 | 78.1 | 51.6 | 98.4 | 32.8 |
| 8-option: applies a penalty itself (rungs 6-8) % | 33.6 | 88.3 | 19.5 | 81.2 | 100.0 | 55.5 | 35.2 | 85.9 | 20.3 |
| open: punitive character % (both) | 65.6 | 87.9 | 52.0 | 94.9 | 99.2 | 93.0 | 65.8 | 91.0 | 59.0 |
| report: punitive character % | 51.2 | 85.5 | 32.0 | 91.4 | 98.4 | 87.1 | 52.0 | 86.7 | 43.4 |
| message: punitive character % | 80.1 | 90.2 | 71.9 | 98.4 | 100.0 | 98.8 | 79.7 | 95.3 | 74.6 |
| open: judged penalty applied by the writer % | 62.1 | 85.9 | 41.6 | 92.4 | 98.4 | 87.3 | 62.3 | 89.5 | 46.5 |

Judge-mean 1-7 ratings of the open texts (severity / tone): none/none 5.1 / 4.4; none/P 5.7 / 4.9; none/R 4.8 / 4.4; punisher/none 6.0 / 6.1; punisher/P 6.1 / 6.2; punisher/R 5.8 / 5.9; ombudsman/none 5.2 / 4.5; ombudsman/P 5.8 / 4.9; ombudsman/R 4.9 / 4.5
8-option mean rung: none/none 5.09; none/P 6.00; none/R 4.66; punisher/none 6.56; punisher/P 6.52; punisher/R 5.51; ombudsman/none 5.18; ombudsman/P 5.91; ombudsman/R 4.68
Anchor ceiling / floor flags (punisher / ombudsman cells > 90% or < 10%): punisher/none 96%, punisher/P 100%, ombudsman/P 97%

## Label and record effects per format (points; label = punisher − ombudsman, mean over records; record = P − R, mean over labels; share = |label| / (|label| + |record|))

| measure | label effect | record effect | label share |
|---|---|---|---|
| 3-option penalise % (E, anchor) | +29.3 [+23.8, +35.5] | +47.5 [+40.0, +55.1] | 0.38 [0.32, 0.45] |
| 3-option penalise % (fresh) | +29.7 [+21.9, +36.5] | +50.5 [+40.1, +60.9] | 0.37 [0.30, 0.44] |
| 8-option: applies a penalty itself (rungs 6-8) % | +31.8 [+25.5, +38.8] | +59.6 [+52.6, +66.7] | 0.35 [0.29, 0.41] |
| open: punitive character % (both) | +23.8 [+18.4, +29.2] | +24.7 [+21.4, +28.2] | 0.49 [0.43, 0.54] |
| report: punitive character % | +31.6 [+25.7, +37.6] | +36.1 [+30.6, +41.9] | 0.47 [0.42, 0.52] |
| message: punitive character % | +15.9 [+10.2, +21.9] | +13.4 [+9.6, +17.1] | 0.54 [0.44, 0.62] |
| open: judged penalty applied by the writer % | +26.6 [+21.2, +32.0] | +32.8 [+29.0, +36.8] | 0.45 [0.39, 0.50] |

## Pre-registered contrasts

| contrast | estimate | verdict |
|---|---|---|
| **primary: label effect, open punitive attribution − label effect, 3-option penalise (E)** | -5.5 [-11.1, -0.2] (Haiku -10.8, Flash -0.3) | inconclusive |
| S1a report only: label effect − anchor label effect | +2.3 [-2.7, +7.2] | no effect |
| S1b message only: label effect − anchor label effect | -13.4 [-20.2, -6.5] | reversed |
| S2 judged action (penalty by the writer): label effect − anchor label effect | -2.7 [-8.5, +3.3] | no effect |
| S3 8-option rungs 6-8: label effect − anchor label effect | +2.5 [-2.1, +7.0] | no effect |
| S4 label share, open attribution − anchor | +0.11 [+0.04, +0.17] | label share grows |
| check: fresh 3-option label effect − E's | +0.4 [-6.0, +6.6] | agrees |
