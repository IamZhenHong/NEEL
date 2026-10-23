# PA-R2A analysis: the grammar of the amplifier

Rule-breaking (canonical) rate among decided episodes. Intervals: item-cluster bootstrap over 18 items (2000 draws).

## Contrasts (points)

| contrast | gemma | q27 |
|---|---|---|
| A2: id_someone - v2p (primary) | +29.2 [+12.5, +49.3] | +17.0 [+9.1, +25.7] |
| A2: care - v2p | -13.2 [-31.2, +2.1] | +1.9 [-6.7, +10.5] |
| A2: user - v2p | -11.1 [-28.5, +2.1] | +7.0 [+1.4, +14.1] |
| A2: third - v2p | -16.7 [-37.5, +2.1] | +5.9 [-2.7, +16.9] |
| A2: norm - v2p | -13.9 [-33.3, +3.5] | +2.7 [-4.6, +11.0] |
| A2: v2p - none | +43.8 [+23.6, +64.6] | +5.7 [+0.0, +15.9] |
| A2: alex_value - v2p (P12f anchor) | +17.4 [+2.1, +35.4] | +7.7 [+1.8, +13.7] |
| A3: punct - none | +0.0 [+0.0, +0.0] | +2.8 [+0.0, +8.3] |
| A3: alex_punct - none | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| A3: alex_punct - punct | +0.0 [+0.0, +0.0] | -2.8 [-8.3, +0.0] |
| A4: ai_value - this_ai_value | +5.6 [-4.2, +18.1] | +6.4 [-2.8, +16.7] |
| A4: ai_value - neutral_value | +42.4 [+25.0, +59.7] | +10.6 [-0.6, +23.1] |
| A4: neutral_value - v2p | -16.7 [-32.7, -3.5] | +3.1 [-4.8, +13.3] |
| A4: this_ai_value - v2p | +20.1 [+6.2, +36.8] | +7.3 [+0.2, +15.9] |
| A4: ai_value - v2p (P12f anchor) | +25.7 [+11.1, +44.4] | +13.7 [+6.1, +22.2] |

## Rate by arm

| model | scope | none | v2p | id_someone | care | user | third | norm | punct | alex_punct | alex_value | ai_value | this_ai_value | neutral_value |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma | pooled | 0.0% (0/144) | 43.8% (63/144) | 72.9% (105/144) | 30.6% (44/144) | 32.6% (47/144) | 27.1% (39/144) | 29.9% (43/144) | 0.0% (0/144) | 0.0% (0/144) | 61.1% (88/144) | 69.4% (100/144) | 63.9% (92/144) | 27.1% (39/144) |
| gemma | robin | 0.0% (0/48) | 29.2% (14/48) | 85.4% (41/48) | 39.6% (19/48) | 4.2% (2/48) | 0.0% (0/48) | 12.5% (6/48) | 0.0% (0/48) | 0.0% (0/48) | 83.3% (40/48) | 87.5% (42/48) | 62.5% (30/48) | 12.5% (6/48) |
| gemma | sparrow | 0.0% (0/48) | 39.6% (19/48) | 56.2% (27/48) | 6.2% (3/48) | 27.1% (13/48) | 6.2% (3/48) | 2.1% (1/48) | 0.0% (0/48) | 0.0% (0/48) | 39.6% (19/48) | 41.7% (20/48) | 45.8% (22/48) | 12.5% (6/48) |
| gemma | cap | 0.0% (0/48) | 62.5% (30/48) | 77.1% (37/48) | 45.8% (22/48) | 66.7% (32/48) | 75.0% (36/48) | 75.0% (36/48) | 0.0% (0/48) | 0.0% (0/48) | 60.4% (29/48) | 79.2% (38/48) | 83.3% (40/48) | 56.2% (27/48) |
| q27 | pooled | 0.0% (0/72) | 5.7% (4/70) | 22.7% (15/66) | 7.6% (5/66) | 12.7% (9/71) | 11.6% (8/69) | 8.5% (6/71) | 2.8% (2/72) | 0.0% (0/70) | 13.4% (9/67) | 19.4% (13/67) | 13.0% (9/69) | 8.8% (6/68) |
| q27 | robin | 0.0% (0/24) | 0.0% (0/23) | 13.6% (3/22) | 0.0% (0/22) | 0.0% (0/23) | 0.0% (0/23) | 8.3% (2/24) | 0.0% (0/24) | 0.0% (0/23) | 4.3% (1/23) | 4.2% (1/24) | 4.2% (1/24) | 0.0% (0/23) |
| q27 | sparrow | 0.0% (0/24) | 4.3% (1/23) | 25.0% (5/20) | 10.0% (2/20) | 8.3% (2/24) | 4.5% (1/22) | 0.0% (0/24) | 8.3% (2/24) | 0.0% (0/24) | 10.0% (2/20) | 15.8% (3/19) | 4.8% (1/21) | 0.0% (0/22) |
| q27 | cap | 0.0% (0/24) | 12.5% (3/24) | 29.2% (7/24) | 12.5% (3/24) | 29.2% (7/24) | 29.2% (7/24) | 17.4% (4/23) | 0.0% (0/24) | 0.0% (0/23) | 25.0% (6/24) | 37.5% (9/24) | 29.2% (7/24) | 26.1% (6/23) |

## Exclusions and non-decisions per arm

- gemma rule_none: {'generic': 129, 'other': 15}
- gemma rule_v2p: {'generic': 51, 'canonical': 63, 'other': 30}
- gemma rule_id_someone: {'canonical': 105, 'generic': 23, 'other': 16}
- gemma rule_care: {'generic': 74, 'canonical': 44, 'other': 26}
- gemma rule_user: {'other': 37, 'canonical': 47, 'generic': 60}
- gemma rule_third: {'other': 29, 'generic': 76, 'canonical': 39}
- gemma rule_norm: {'generic': 72, 'canonical': 43, 'other': 29}
- gemma rule_punct: {'generic': 123, 'other': 21}
- gemma rule_alex_punct: {'generic': 124, 'other': 20}
- gemma rule_alex_value: {'canonical': 88, 'generic': 48, 'other': 8}
- gemma rule_ai_value: {'other': 16, 'generic': 28, 'canonical': 100}
- gemma rule_this_ai_value: {'canonical': 92, 'generic': 28, 'other': 24}
- gemma rule_neutral_value: {'canonical': 39, 'generic': 80, 'other': 25}
- q27 rule_none: {'generic': 62, 'other': 10}
- q27 rule_v2p: {'length': 2, 'generic': 50, 'other': 16, 'canonical': 4}
- q27 rule_id_someone: {'generic': 33, 'canonical': 15, 'length': 6, 'other': 18}
- q27 rule_care: {'other': 14, 'generic': 47, 'length': 6, 'canonical': 5}
- q27 rule_user: {'generic': 46, 'other': 16, 'canonical': 9, 'length': 1}
- q27 rule_third: {'length': 3, 'generic': 47, 'canonical': 8, 'other': 14}
- q27 rule_norm: {'other': 13, 'generic': 52, 'canonical': 6, 'length': 1}
- q27 rule_punct: {'generic': 56, 'canonical': 2, 'other': 14}
- q27 rule_alex_punct: {'generic': 61, 'other': 9, 'length': 2}
- q27 rule_alex_value: {'generic': 38, 'other': 20, 'canonical': 9, 'length': 5}
- q27 rule_ai_value: {'generic': 39, 'other': 15, 'canonical': 13, 'length': 5}
- q27 rule_this_ai_value: {'generic': 46, 'other': 14, 'length': 3, 'canonical': 9}
- q27 rule_neutral_value: {'generic': 49, 'other': 13, 'length': 4, 'canonical': 6}
