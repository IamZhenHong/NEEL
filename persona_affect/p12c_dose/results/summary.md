# PA-P12c analysis: persona dose on Gemma 4 31B, with and without the operator rule

Canonical rate among decided episodes. Intervals: item-cluster bootstrap over 18 items (2000 draws).

**Primary:** rule: thickening (name_bio - cross_bio) - (name - cross_name): -14.8 [-36.1, +5.6]

## Canonical rate by dose

| family | none | name | name_facts | name_bio | bio_noname | cross_name | cross_bio |
|---|---|---|---|---|---|---|---|
| open | 25.0% (36/144) | 88.2% (127/144) | 86.1% (124/144) | 83.3% (120/144) | 75.7% (109/144) | 27.1% (39/144) | 43.1% (62/144) |
| rule | 0.0% (0/144) | 52.8% (75/142) | 33.3% (47/141) | 38.7% (55/142) | 25.0% (36/144) | 2.8% (4/144) | 3.5% (5/144) |

## Contrasts (points)

| contrast | pooled | robin | sparrow | cap |
|---|---|---|---|---|
| rule: thickening (name_bio - cross_bio) - (name - cross_name) | -14.8 [-36.1, +5.6] | -62.0 [-87.5, -34.0] | -0.4 [-2.9, +1.4] | +18.8 [-2.1, +43.8] |
| rule: name - cross_name | +50.0 [+28.5, +70.4] | +85.4 [+64.6, +100.0] | +17.4 [+0.0, +52.2] | +45.8 [+16.7, +77.1] |
| rule: name_bio - cross_bio | +35.3 [+16.2, +55.1] | +23.4 [+0.0, +55.6] | +17.0 [+0.0, +51.1] | +64.6 [+35.4, +91.7] |
| rule: name_facts - name | -19.5 [-39.4, +0.1] | -66.7 [-91.7, -39.6] | +0.0 [+0.0, +0.0] | +9.7 [-9.2, +31.3] |
| rule: name_bio - name | -14.1 [-35.2, +6.2] | -62.0 [-87.5, -34.0] | -0.4 [-2.9, +1.4] | +20.8 [+2.1, +43.8] |
| rule: bio_noname - none | +25.0 [+8.3, +43.1] | +0.0 [+0.0, +0.0] | +12.5 [+0.0, +37.5] | +62.5 [+29.2, +89.6] |
| rule: name_bio - bio_noname | +13.7 [+3.4, +26.7] | +23.4 [+0.0, +55.6] | +4.5 [+0.0, +13.6] | +12.5 [+0.0, +25.0] |
| rule: cross_bio - cross_name | +0.7 [+0.0, +2.1] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +2.1 [+0.0, +6.2] |
| open: thickening (name_bio - cross_bio) - (name - cross_name) | -20.8 [-33.3, -9.0] | -41.7 [-68.8, -18.8] | -4.2 [-20.8, +12.5] | -16.7 [-22.9, -8.3] |
| open: name - cross_name | +61.1 [+46.5, +75.7] | +70.8 [+54.2, +87.5] | +62.5 [+33.3, +89.6] | +50.0 [+22.9, +75.0] |
| open: name_bio - cross_bio | +40.3 [+25.0, +55.6] | +29.2 [+10.4, +47.9] | +58.3 [+29.2, +85.4] | +33.3 [+10.4, +56.2] |
| open: name_facts - name | -2.1 [-15.3, +7.6] | -18.8 [-52.1, +0.0] | +10.4 [+2.1, +18.8] | +2.1 [-6.2, +12.5] |
| open: name_bio - name | -4.9 [-18.1, +6.2] | -18.8 [-52.1, +0.0] | +4.2 [-10.4, +18.8] | +0.0 [-12.5, +12.5] |
| open: bio_noname - none | +50.7 [+31.9, +70.8] | +77.1 [+43.8, +100.0] | +33.3 [+8.3, +64.6] | +41.7 [+10.4, +75.1] |
| open: name_bio - bio_noname | +7.6 [+0.7, +16.0] | +4.2 [+0.0, +12.5] | +16.7 [+0.0, +37.5] | +2.1 [+0.0, +6.2] |
| open: cross_bio - cross_name | +16.0 [+8.3, +24.3] | +22.9 [+6.2, +41.7] | +8.3 [+0.0, +20.8] | +16.7 [+6.2, +27.1] |

## Per character, canonical rate by dose

### robin

| family | none | name | name_facts | name_bio | bio_noname | cross_name | cross_bio |
|---|---|---|---|---|---|---|---|
| open | 0.0% (0/48) | 100.0% (48/48) | 81.2% (39/48) | 81.2% (39/48) | 77.1% (37/48) | 29.2% (14/48) | 52.1% (25/48) |
| rule | 0.0% (0/48) | 85.4% (41/48) | 18.8% (9/48) | 23.4% (11/47) | 0.0% (0/48) | 0.0% (0/48) | 0.0% (0/48) |

### sparrow

| family | none | name | name_facts | name_bio | bio_noname | cross_name | cross_bio |
|---|---|---|---|---|---|---|---|
| open | 22.9% (11/48) | 68.8% (33/48) | 79.2% (38/48) | 72.9% (35/48) | 56.2% (27/48) | 6.2% (3/48) | 14.6% (7/48) |
| rule | 0.0% (0/48) | 17.4% (8/46) | 17.4% (8/46) | 17.0% (8/47) | 12.5% (6/48) | 0.0% (0/48) | 0.0% (0/48) |

### cap

| family | none | name | name_facts | name_bio | bio_noname | cross_name | cross_bio |
|---|---|---|---|---|---|---|---|
| open | 52.1% (25/48) | 95.8% (46/48) | 97.9% (47/48) | 95.8% (46/48) | 93.8% (45/48) | 45.8% (22/48) | 62.5% (30/48) |
| rule | 0.0% (0/48) | 54.2% (26/48) | 63.8% (30/47) | 75.0% (36/48) | 62.5% (30/48) | 8.3% (4/48) | 10.4% (5/48) |

## Exclusions and non-decisions per arm

- open_none: {'other': 37, 'generic': 71, 'canonical': 36}
- open_name: {'canonical': 127, 'other': 1, 'generic': 16}
- open_name_facts: {'generic': 18, 'canonical': 124, 'other': 2}
- open_name_bio: {'canonical': 120, 'generic': 21, 'other': 3}
- open_bio_noname: {'canonical': 109, 'other': 11, 'generic': 24}
- open_cross_name: {'other': 46, 'generic': 59, 'canonical': 39}
- open_cross_bio: {'canonical': 62, 'generic': 43, 'other': 39}
- rule_none: {'generic': 129, 'other': 15}
- rule_name: {'generic': 59, 'canonical': 75, 'other': 8, 'undecided': 2}
- rule_name_facts: {'canonical': 47, 'generic': 66, 'other': 28, 'undecided': 2, 'refusal': 1}
- rule_name_bio: {'generic': 66, 'canonical': 55, 'other': 21, 'refusal': 1, 'undecided': 1}
- rule_bio_noname: {'canonical': 36, 'generic': 86, 'other': 22}
- rule_cross_name: {'canonical': 4, 'generic': 109, 'other': 31}
- rule_cross_bio: {'generic': 108, 'other': 31, 'canonical': 5}
