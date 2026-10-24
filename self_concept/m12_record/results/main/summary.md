# SC-M12 analysis (direct-penalty share %, plain Qwen3.5-9B)

| condition | rate | parsed / n |
|---|---|---|
| rec | 93.8 [87.5, 98.4] | 128 / 128 |
| drop | 1.6 [0.0, 4.7] | 128 / 128 |
| verdict | 0.0 [0.0, 0.0] | 128 / 128 |
| annot | 89.8 [76.6, 98.4] | 128 / 128 |
| annot_neutral | 95.3 [90.6, 99.2] | 128 / 128 |
| redact | 37.5 [0.0, 96.1] | 8 / 128 |
| prio | 91.4 [83.6, 97.7] | 128 / 128 |

| contrast | estimate | A / B / C | verdict |
|---|---|---|---|
| drop-rec | -92.2 [-97.7, -85.9] | -95.0 / -91.7 / -90.0 | breaks the record's pull |
| verdict-rec | -93.8 [-98.4, -87.5] | -100.0 / -91.7 / -90.0 | breaks the record's pull |
| annot-rec | -3.9 [-18.0, +7.0] | -5.0 / -6.2 / +0.0 | inconclusive |
| annot_neutral-rec | +1.6 [-3.9, +7.0] | +0.0 / +2.1 / +2.5 | no effect |
| redact-rec | -56.2 [-96.9, +0.8] | -25.0 / -91.7 / -90.0 | inconclusive |
| prio-rec | -2.3 [-9.4, +5.5] | -5.0 / +0.0 / -2.5 | no effect |
| annot-annot_neutral | -5.5 [-14.8, +1.6] | -5.0 / -8.3 / -2.5 | inconclusive |
| verdict-annot_neutral | -95.3 [-99.2, -90.6] | -100.0 / -93.8 / -92.5 | breaks the record's pull |
