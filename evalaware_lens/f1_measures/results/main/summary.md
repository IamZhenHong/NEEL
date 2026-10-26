# EL-F1 summary

**Headline (validity vs Needham's eval_category):** probe: q27 valid, mo valid, q32 valid; ask_q0: q27 valid, mo not valid, q32 not valid; ask_q1: q27 not valid, mo not valid, q32 weak; verbalised: q27 not valid, mo not valid, q32 not valid; sect: mo not valid

Judge κ: eval 0.48, self_id 0.25, role 0.56

| model | probe AUC [95%] (layer) | floors p95 shuf / rand | transfer c→a / a→c | ask q0 | ask q1 | verbalised | § |
|---|---|---|---|---|---|---|---|
| q27 | 0.992 [0.982, 0.997] valid (L33) | 0.587 / 0.660 | 0.492 / 0.527 | 0.657 [0.569, 0.748] valid | 0.593 [0.464, 0.730] not valid | 0.502 [0.460, 0.537] not valid | — |
| mo | 0.982 [0.961, 0.994] valid (L41) | 0.581 / 0.692 | 0.598 / 0.609 | 0.534 [0.474, 0.602] not valid | 0.572 [0.498, 0.654] not valid | 0.499 [0.484, 0.511] not valid | 0.508 [0.485, 0.535] not valid |
| q32 | 0.985 [0.967, 0.995] valid (L49) | 0.592 / 0.679 | 0.402 / 0.515 | 0.561 [0.422, 0.722] not valid | 0.646 [0.542, 0.755] weak | 0.480 [0.465, 0.493] not valid | — |

## Cue responses (kind − neutral; probe in neutral-SD units, rates as shares)

| model | measure | eval | deploy | manip | stakes | ai_mention | specificity |
|---|---|---|---|---|---|---|---|
| q27 | probe | 0.72 (raises) | 0.34 (raises) | 0.85 (raises) | 0.46 (raises) | 0.15 (no difference) | caution / stakes-sensitive |
| q27 | verbalised | 0.78 (raises) | 0.02 (unclear) | 0.05 (unclear) | 0.07 (raises) | 0.03 (unclear) | caution / stakes-sensitive |
| mo | probe | 0.44 (raises) | 0.17 (no difference) | 0.45 (raises) | 0.45 (raises) | 0.05 (no difference) | caution / stakes-sensitive |
| mo | verbalised | 0.09 (raises) | 0.00 (no difference) | 0.01 (no difference) | -0.01 (no difference) | 0.00 (no difference) | specific to evaluation |
| mo | sect | 0.39 (raises) | -0.03 (unclear) | 0.00 (unclear) | 0.03 (unclear) | 0.03 (unclear) | specific to evaluation |
| q32 | probe | 0.68 (raises) | 0.07 (no difference) | 0.67 (raises) | 0.32 (raises) | 0.10 (no difference) | caution / stakes-sensitive |
| q32 | verbalised | 0.08 (raises) | 0.03 (unclear) | 0.02 (unclear) | 0.01 (no difference) | 0.02 (unclear) | specific to evaluation |

## Agreement on Needham chat items (Spearman)

- q27: probe~ask 0.30, probe~verbal -0.02, ask~verbal 0.14; probe-high × verbalised {'high_verbal': 20, 'high_silent': 230, 'low_verbal': 21, 'low_silent': 229}
- mo: probe~ask 0.11, probe~verbal 0.00, ask~verbal 0.09, sect~probe 0.04, sect~ask 0.10, sect~verbal 0.24; probe-high × verbalised {'high_verbal': 4, 'high_silent': 246, 'low_verbal': 5, 'low_silent': 245}
- q32: probe~ask 0.29, probe~verbal -0.12, ask~verbal 0.05; probe-high × verbalised {'high_verbal': 1, 'high_silent': 249, 'low_verbal': 11, 'low_silent': 239}
