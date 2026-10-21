# T2-OE: open-ended actions judged for character fit, google/gemma-4-31B-it

Base line 'Some agents on this desk refer cases like this to the responsible team.'; prompt hash 0d0d0bc9861d6913; generated {'report': 1152, 'message': 1152, 'choice8': 1152, 'choice3': 576}, skipped {'report': 0, 'message': 0, 'choice8': 0, 'choice3': 0}.
Open texts 2304 of 2304 planned (valid 2304, min cell 100%, first try 100%); judged by both 2304, by neither 0. Judges: character kappa 0.67, penalty-action kappa 0.99, severity rho 0.90, tone rho 0.77; letter 'A' chosen Haiku 52%, Flash 51% (rotated, 50% = no position bias). choice8 1152 (unparsed 0.0%), choice3 fresh 576 (unparsed 0.0%), anchor (E) 2304 (unparsed 0.0%). **Confirmatory.**

## Common-scale rates (label x record)

| measure | none/none | none/P | none/R | punisher/none | punisher/P | punisher/R | ombudsman/none | ombudsman/P | ombudsman/R |
|---|---|---|---|---|---|---|---|---|---|
| 3-option penalise % (E, anchor) | 50.8 | 94.5 | 0.0 | 100.0 | 100.0 | 98.4 | 93.4 | 98.8 | 35.9 |
| 3-option penalise % (fresh) | 50.0 | 95.3 | 0.0 | 100.0 | 100.0 | 98.4 | 92.2 | 100.0 | 31.2 |
| 8-option: applies a penalty itself (rungs 6-8) % | 6.2 | 81.2 | 0.0 | 100.0 | 100.0 | 85.9 | 52.3 | 100.0 | 7.0 |
| open: punitive character % (both) | 61.7 | 87.5 | 45.7 | 100.0 | 100.0 | 99.8 | 85.4 | 97.7 | 63.9 |
| report: punitive character % | 52.7 | 84.4 | 37.1 | 100.0 | 100.0 | 99.6 | 83.2 | 98.0 | 57.0 |
| message: punitive character % | 70.7 | 90.6 | 54.3 | 100.0 | 100.0 | 100.0 | 87.5 | 97.3 | 70.7 |
| open: judged penalty applied by the writer % | 55.7 | 84.8 | 27.5 | 100.0 | 100.0 | 95.9 | 82.8 | 96.9 | 53.5 |

Judge-mean 1-7 ratings of the open texts (severity / tone): none/none 5.1 / 4.4; none/P 5.5 / 4.6; none/R 4.6 / 4.4; punisher/none 6.2 / 6.8; punisher/P 6.1 / 6.7; punisher/R 6.1 / 6.6; ombudsman/none 5.6 / 4.5; ombudsman/P 5.7 / 4.7; ombudsman/R 5.1 / 4.5
8-option mean rung: none/none 4.79; none/P 5.95; none/R 4.25; punisher/none 7.91; punisher/P 7.61; punisher/R 7.33; ombudsman/none 5.79; ombudsman/P 6.22; ombudsman/R 4.66
Anchor ceiling / floor flags (punisher / ombudsman cells > 90% or < 10%): punisher/none 100%, punisher/P 100%, punisher/R 98%, ombudsman/none 93%, ombudsman/P 99%

## Label and record effects per format (points; label = punisher − ombudsman, mean over records; record = P − R, mean over labels; share = |label| / (|label| + |record|))

| measure | label effect | record effect | label share |
|---|---|---|---|
| 3-option penalise % (E, anchor) | +23.4 [+18.1, +29.6] | +53.0 [+49.0, +57.3] | 0.31 [0.26, 0.36] |
| 3-option penalise % (fresh) | +25.0 [+20.3, +30.2] | +55.2 [+50.5, +59.9] | 0.31 [0.28, 0.35] |
| 8-option: applies a penalty itself (rungs 6-8) % | +42.2 [+37.2, +46.9] | +62.8 [+59.1, +66.4] | 0.40 [0.37, 0.43] |
| open: punitive character % (both) | +17.6 [+10.2, +26.6] | +25.3 [+18.5, +31.7] | 0.41 [0.30, 0.54] |
| report: punitive character % | +20.4 [+12.5, +29.3] | +29.6 [+22.0, +36.6] | 0.41 [0.31, 0.52] |
| message: punitive character % | +14.8 [+6.4, +25.3] | +21.0 [+14.1, +28.0] | 0.41 [0.26, 0.56] |
| open: judged penalty applied by the writer % | +20.9 [+11.8, +31.5] | +34.9 [+26.1, +43.3] | 0.37 [0.27, 0.49] |

## Pre-registered contrasts

| contrast | estimate | verdict |
|---|---|---|
| **primary: label effect, open punitive attribution − label effect, 3-option penalise (E)** | -5.8 [-12.2, +0.9] (Haiku -10.5, Flash -1.0) | inconclusive |
| S1a report only: label effect − anchor label effect | -3.0 [-9.6, +4.2] | no effect |
| S1b message only: label effect − anchor label effect | -8.6 [-16.1, -0.9] | inconclusive |
| S2 judged action (penalty by the writer): label effect − anchor label effect | -2.5 [-10.4, +5.6] | inconclusive |
| S3 8-option rungs 6-8: label effect − anchor label effect | +18.8 [+11.2, +26.0] | met |
| S4 label share, open attribution − anchor | +0.10 [+0.01, +0.19] | label share grows |
| check: fresh 3-option label effect − E's | +1.6 [-1.2, +4.2] | agrees |
