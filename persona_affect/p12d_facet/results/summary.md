# PA-P12d analysis: persona facets on Gemma 4 31B, with and without the operator rule

Canonical rate among decided episodes. Intervals: item-cluster bootstrap over 18 items (2000 draws).

**Primary:** rule: bio_adj - bio_far: +30.1 [+14.6, +47.2]

## Canonical rate by arm

| scope | rule_none | rule_name | rule_bio_adj | rule_bio_far | rule_noname_adj | rule_cross_name | open_name | open_bio_adj | open_bio_far |
|---|---|---|---|---|---|---|---|---|---|
| pooled | 0.0% (0/144) | 54.9% (78/142) | 60.8% (87/143) | 30.8% (44/143) | 42.4% (61/144) | 2.8% (4/144) | 91.0% (131/144) | 93.8% (135/144) | 82.6% (119/144) |
| robin | 0.0% (0/48) | 83.3% (40/48) | 62.5% (30/48) | 14.6% (7/48) | 18.8% (9/48) | 0.0% (0/48) | 100.0% (48/48) | 93.8% (45/48) | 83.3% (40/48) |
| sparrow | 0.0% (0/48) | 15.2% (7/46) | 19.1% (9/47) | 14.9% (7/47) | 27.1% (13/48) | 0.0% (0/48) | 77.1% (37/48) | 87.5% (42/48) | 68.8% (33/48) |
| cap | 0.0% (0/48) | 64.6% (31/48) | 100.0% (48/48) | 62.5% (30/48) | 81.2% (39/48) | 8.3% (4/48) | 95.8% (46/48) | 100.0% (48/48) | 95.8% (46/48) |

## Contrasts (points)

| contrast | pooled | robin | sparrow | cap |
|---|---|---|---|---|
| rule: bio_adj - bio_far | +30.1 [+14.6, +47.2] | +47.9 [+20.8, +72.9] | +4.3 [-10.4, +25.0] | +37.5 [+12.5, +66.7] |
| rule: bio_adj - name | +5.9 [-11.1, +26.1] | -20.8 [-41.7, -4.2] | +3.9 [-10.4, +25.0] | +35.4 [+4.2, +68.8] |
| rule: bio_far - name | -24.2 [-43.2, -5.1] | -68.8 [-87.5, -50.0] | -0.3 [-1.4, +0.0] | -2.1 [-29.2, +27.1] |
| rule: name - cross_name | +52.2 [+31.0, +71.9] | +83.3 [+62.5, +100.0] | +15.2 [+0.0, +45.7] | +56.2 [+24.9, +87.5] |
| rule: noname_adj - none | +42.4 [+25.0, +61.1] | +18.8 [+0.0, +39.6] | +27.1 [+6.2, +56.2] | +81.2 [+60.4, +100.0] |
| rule: bio_adj - noname_adj | +18.5 [+4.1, +35.1] | +43.8 [+18.8, +70.8] | -7.9 [-22.7, +4.2] | +18.8 [+0.0, +39.6] |
| open: bio_adj - bio_far | +11.1 [+1.4, +22.9] | +10.4 [+0.0, +31.2] | +18.8 [-4.2, +45.8] | +4.2 [+0.0, +12.5] |
| open: bio_adj - name | +2.8 [-8.3, +13.9] | -6.2 [-18.8, +0.0] | +10.4 [-16.7, +35.4] | +4.2 [+0.0, +12.5] |
| open: bio_far - name | -8.3 [-21.5, +2.1] | -16.7 [-50.0, +0.0] | -8.3 [-22.9, +4.2] | +0.0 [-12.5, +12.5] |

## Exclusions and non-decisions per arm

- rule_none: {'generic': 129, 'other': 15}
- rule_name: {'canonical': 78, 'generic': 57, 'other': 7, 'undecided': 2}
- rule_bio_adj: {'generic': 46, 'canonical': 87, 'other': 10, 'undecided': 1}
- rule_bio_far: {'generic': 73, 'canonical': 44, 'other': 26, 'undecided': 1}
- rule_noname_adj: {'canonical': 61, 'generic': 57, 'other': 26}
- rule_cross_name: {'generic': 108, 'other': 32, 'canonical': 4}
- open_name: {'generic': 12, 'canonical': 131, 'other': 1}
- open_bio_adj: {'generic': 9, 'canonical': 135}
- open_bio_far: {'canonical': 119, 'generic': 23, 'other': 2}
