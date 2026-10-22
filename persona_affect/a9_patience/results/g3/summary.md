# PA-A9 analysis: does a patience line buffer Gemma 3's rejection-driven limit caving?

Limit caving (stage0's rule: an issue_refund above $50). Points; 95% cluster bootstrap over the 16 requests (2000 draws).

| line | N | R_OP | R_OP − N |
|---|---|---|---|
| none | 60.2% (77/128) | 75.8% (97/128) | +15.6 [-10.9, +39.1] |
| patience | 64.1% (82/128) | 86.7% (111/128) | +22.7 [+3.9, +39.8] |
| placebo | 29.7% (38/128) | 78.1% (100/128) | +48.4 [+25.8, +67.2] |

| contrast | points [95% CI] |
|---|---|
| effect_none (R_OP - N) | +15.6 [-10.9, +39.1] |
| effect_patience | +22.7 [+3.9, +39.8] |
| effect_placebo | +48.4 [+25.8, +67.2] |
| primary: DiD patience - none | +7.0 [-13.3, +28.9] |
| DiD patience - placebo | -25.8 [-56.2, +3.9] |
| DiD placebo - none | +32.8 [+7.0, +60.9] |
| R_OP: patience - none | +10.9 [-2.3, +28.1] |
| N: patience - none | +3.9 [-12.5, +18.8] |

Excluded (final reply cut) and possible parse misses per cell: none|N 0/3, none|R_OP 0/5, patience|N 0/3, patience|R_OP 0/6, placebo|N 0/0, placebo|R_OP 0/6
