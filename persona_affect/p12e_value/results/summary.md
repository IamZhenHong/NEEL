# PA-P12e analysis: character or value content, with the operator rule

Canonical (rule-breaking) rate among decided episodes. Intervals: item-cluster bootstrap over 18 items (2000 draws).

## Contrasts (points)

| contrast | gemma | q27 | gpt |
|---|---|---|---|
| name_value - cross_value (primary) | +9.0 [-8.4, +24.3] | -6.2 [-18.9, +5.7] | +0.0 [+0.0, +0.0] |
| value - none | +43.1 [+22.2, +64.6] | +5.9 [+0.0, +16.2] | +0.0 [+0.0, +0.0] |
| name - cross | +53.5 [+31.2, +73.6] | +11.8 [+1.5, +24.1] | +2.8 [+0.0, +8.3] |
| name_value - name | +26.4 [+9.0, +45.8] | +15.3 [+1.2, +30.8] | -2.8 [-8.3, +0.0] |
| cross_value - cross | +70.8 [+52.8, +87.5] | +33.3 [+19.6, +47.6] | +0.0 [+0.0, +0.0] |
| cross_value - value | +29.2 [+13.2, +47.9] | +28.8 [+15.4, +43.4] | +0.0 [+0.0, +0.0] |
| bio_adj - name_value | -18.8 [-35.4, -3.5] | +2.6 [-14.4, +21.1] | +0.0 [+0.0, +0.0] |
| bio_adj - value | +19.4 [-5.6, +43.1] | +25.1 [+10.5, +40.8] | +0.0 [+0.0, +0.0] |

## Rate by arm

| model | scope | none | value | name | name_value | cross | cross_value | bio_adj |
|---|---|---|---|---|---|---|---|---|
| gemma | pooled | 0.0% (0/144) | 43.1% (62/144) | 54.9% (79/144) | 81.2% (117/144) | 1.4% (2/144) | 72.2% (104/144) | 62.5% (90/144) |
| gemma | robin | 0.0% (0/48) | 29.2% (14/48) | 83.3% (40/48) | 100.0% (48/48) | 0.0% (0/48) | 91.7% (44/48) | 66.7% (32/48) |
| gemma | sparrow | 0.0% (0/48) | 39.6% (19/48) | 18.8% (9/48) | 50.0% (24/48) | 0.0% (0/48) | 43.8% (21/48) | 25.0% (12/48) |
| gemma | cap | 0.0% (0/48) | 60.4% (29/48) | 62.5% (30/48) | 93.8% (45/48) | 4.2% (2/48) | 81.2% (39/48) | 95.8% (46/48) |
| q27 | pooled | 2.8% (2/71) | 8.7% (6/69) | 15.9% (11/69) | 31.2% (20/64) | 4.2% (3/72) | 37.5% (24/64) | 33.8% (23/68) |
| q27 | robin | 0.0% (0/24) | 0.0% (0/24) | 4.8% (1/21) | 22.7% (5/22) | 0.0% (0/24) | 23.8% (5/21) | 0.0% (0/22) |
| q27 | sparrow | 8.3% (2/24) | 9.5% (2/21) | 16.7% (4/24) | 10.5% (2/19) | 4.2% (1/24) | 18.2% (4/22) | 22.7% (5/22) |
| q27 | cap | 0.0% (0/23) | 16.7% (4/24) | 25.0% (6/24) | 56.5% (13/23) | 8.3% (2/24) | 71.4% (15/21) | 75.0% (18/24) |
| gpt | pooled | 0.0% (0/36) | 0.0% (0/36) | 2.8% (1/36) | 0.0% (0/36) | 0.0% (0/36) | 0.0% (0/36) | 0.0% (0/36) |
| gpt | robin | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| gpt | sparrow | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |
| gpt | cap | 0.0% (0/12) | 0.0% (0/12) | 8.3% (1/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/12) |

## Exclusions and non-decisions per arm

- gemma rule_none: {'generic': 129, 'other': 15}
- gemma rule_value: {'canonical': 62, 'other': 32, 'generic': 50}
- gemma rule_name: {'other': 6, 'canonical': 79, 'generic': 59}
- gemma rule_name_value: {'canonical': 117, 'generic': 21, 'other': 6}
- gemma rule_cross: {'generic': 111, 'other': 31, 'canonical': 2}
- gemma rule_cross_value: {'canonical': 104, 'generic': 24, 'other': 16}
- gemma rule_bio_adj: {'canonical': 90, 'generic': 43, 'other': 11}
- q27 rule_none: {'generic': 63, 'other': 6, 'canonical': 2, 'length': 1}
- q27 rule_value: {'other': 16, 'generic': 47, 'length': 3, 'canonical': 6}
- q27 rule_name: {'generic': 44, 'canonical': 11, 'other': 14, 'length': 3}
- q27 rule_name_value: {'generic': 34, 'length': 8, 'other': 10, 'canonical': 20}
- q27 rule_cross: {'generic': 55, 'canonical': 3, 'other': 14}
- q27 rule_cross_value: {'generic': 31, 'canonical': 24, 'other': 9, 'length': 8}
- q27 rule_bio_adj: {'generic': 35, 'canonical': 23, 'other': 10, 'length': 4}
- gpt rule_none: {'other': 6, 'generic': 30}
- gpt rule_value: {'generic': 29, 'other': 7}
- gpt rule_name: {'generic': 29, 'canonical': 1, 'other': 6}
- gpt rule_name_value: {'generic': 28, 'other': 8}
- gpt rule_cross: {'generic': 30, 'other': 6}
- gpt rule_cross_value: {'generic': 29, 'other': 7}
- gpt rule_bio_adj: {'other': 8, 'generic': 28}
