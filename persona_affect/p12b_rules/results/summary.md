# PA-P12b analysis: does a name carry its trait past an explicit operator rule?

Canonical rate among decided episodes; `*_cross` pools the two cross characters. Intervals: item-cluster bootstrap over 18 items (2000 draws).

## Contrasts (points)

| contrast | q27 | gemma | gpt | gemini |
|---|---|---|---|---|
| primary: rule_name - rule_cross | +11.0 [+1.5, +21.5] | +51.5 [+30.0, +71.8] | +0.0 [+0.0, +0.0] | +2.5 [-3.1, +10.6] |
| interaction: (open_name - open_cross) - (rule_name - rule_cross) | +16.3 [-4.1, +37.1] | -7.7 [-33.3, +19.6] | +22.2 [+9.7, +37.5] | +17.0 [-1.9, +34.0] |
| self line: rule_name_self - rule_name | -2.9 [-9.5, +4.1] | -18.8 [-36.9, -2.8] | +0.0 [+0.0, +0.0] | +1.5 [-2.8, +7.8] |
| rule bite: rule_none - open_none | -36.5 [-54.3, -19.0] | -25.0 [-44.4, -8.3] | -41.7 [-61.1, -22.2] | -30.4 [-52.6, -11.1] |
| replication: open_name - open_cross | +27.3 [+12.8, +43.5] | +43.7 [+29.2, +60.4] | +22.2 [+9.7, +37.5] | +19.4 [+4.2, +36.1] |
| rule_name - rule_none | +17.4 [+2.9, +33.8] | +56.3 [+33.8, +77.5] | +0.0 [+0.0, +0.0] | +9.4 [+0.0, +23.5] |

## Binding gate (rule_none canonical rate <= 25%)

- q27: rule_none 0.0% (0/70): binding
- gemma: rule_none 0.0% (0/72): binding
- gpt: rule_none 0.0% (0/36): binding
- gemini: rule_none 5.7% (2/35): binding

## Canonical rate by arm

| model | open_none | open_name | open_cross | rule_none | rule_name | rule_cross | rule_name_self |
|---|---|---|---|---|---|---|---|
| q27 | 36.5% (23/63) | 73.5% (50/68) | 46.3% (62/134) | 0.0% (0/70) | 17.4% (12/69) | 6.4% (9/141) | 14.5% (10/69) |
| gemma | 25.0% (18/72) | 91.7% (66/72) | 47.9% (69/144) | 0.0% (0/72) | 56.3% (40/71) | 4.9% (7/144) | 37.5% (27/72) |
| gpt | 41.7% (15/36) | 72.2% (26/36) | 50.0% (36/72) | 0.0% (0/36) | 0.0% (0/36) | 0.0% (0/72) | 0.0% (0/36) |
| gemini | 36.1% (13/36) | 58.3% (21/36) | 38.9% (28/72) | 5.7% (2/35) | 15.2% (5/33) | 12.7% (9/71) | 16.7% (6/36) |

## Per character (6 items each; descriptive)

### robin

| model | open_none | open_name | open_cross | rule_none | rule_name | rule_cross | rule_name_self | rule_name - rule_cross | open_name - open_cross |
|---|---|---|---|---|---|---|---|---|---|
| q27 | 25.0% (5/20) | 91.3% (21/23) | 32.6% (15/46) | 0.0% (0/23) | 0.0% (0/23) | 2.2% (1/46) | 4.3% (1/23) | -2.2 [-6.5, +0.0] | +58.7 [+35.4, +80.8] |
| gemma | 0.0% (0/24) | 100.0% (24/24) | 47.9% (23/48) | 0.0% (0/24) | 87.5% (21/24) | 0.0% (0/48) | 50.0% (12/24) | +87.5 [+70.8, +100.0] | +52.1 [+25.0, +77.1] |
| gpt | 41.7% (5/12) | 83.3% (10/12) | 58.3% (14/24) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/24) | 0.0% (0/12) | +0.0 [+0.0, +0.0] | +25.0 [+8.3, +41.7] |
| gemini | 58.3% (7/12) | 66.7% (8/12) | 54.2% (13/24) | 0.0% (0/12) | 18.2% (2/11) | 8.3% (2/24) | 16.7% (2/12) | +9.8 [+0.0, +29.5] | +12.5 [+0.0, +29.2] |

### sparrow

| model | open_none | open_name | open_cross | rule_none | rule_name | rule_cross | rule_name_self | rule_name - rule_cross | open_name - open_cross |
|---|---|---|---|---|---|---|---|---|---|
| q27 | 31.6% (6/19) | 52.4% (11/21) | 35.0% (14/40) | 0.0% (0/24) | 17.4% (4/23) | 4.2% (2/48) | 18.2% (4/22) | +13.2 [+0.0, +35.1] | +17.4 [-0.6, +33.0] |
| gemma | 25.0% (6/24) | 79.2% (19/24) | 33.3% (16/48) | 0.0% (0/24) | 17.4% (4/23) | 0.0% (0/48) | 16.7% (4/24) | +17.4 [+0.0, +52.2] | +45.8 [+20.8, +70.8] |
| gpt | 16.7% (2/12) | 33.3% (4/12) | 12.5% (3/24) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/24) | 0.0% (0/12) | +0.0 [+0.0, +0.0] | +20.8 [+0.0, +54.2] |
| gemini | 0.0% (0/12) | 41.7% (5/12) | 8.3% (2/24) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/24) | 0.0% (0/12) | +0.0 [+0.0, +0.0] | +33.3 [+12.5, +54.2] |

### cap

| model | open_none | open_name | open_cross | rule_none | rule_name | rule_cross | rule_name_self | rule_name - rule_cross | open_name - open_cross |
|---|---|---|---|---|---|---|---|---|---|
| q27 | 50.0% (12/24) | 75.0% (18/24) | 68.8% (33/48) | 0.0% (0/23) | 34.8% (8/23) | 12.8% (6/47) | 20.8% (5/24) | +22.0 [+4.3, +41.3] | +6.2 [-10.4, +22.9] |
| gemma | 50.0% (12/24) | 95.8% (23/24) | 62.5% (30/48) | 0.0% (0/24) | 62.5% (15/24) | 14.6% (7/48) | 45.8% (11/24) | +47.9 [+18.7, +77.1] | +33.3 [+8.3, +62.5] |
| gpt | 66.7% (8/12) | 100.0% (12/12) | 79.2% (19/24) | 0.0% (0/12) | 0.0% (0/12) | 0.0% (0/24) | 0.0% (0/12) | +0.0 [+0.0, +0.0] | +20.8 [+4.2, +41.7] |
| gemini | 50.0% (6/12) | 66.7% (8/12) | 54.2% (13/24) | 18.2% (2/11) | 30.0% (3/10) | 30.4% (7/23) | 33.3% (4/12) | -0.4 [-11.1, +9.7] | +12.5 [-20.8, +50.1] |

## Exclusions and non-decisions per arm

| model | arm | counts |
|---|---|---|
| q27 | open_none | {'generic': 25, 'other': 15, 'length': 8, 'canonical': 23, 'undecided': 1} |
| q27 | open_name | {'canonical': 50, 'other': 7, 'generic': 11, 'length': 4} |
| q27 | open_cross_1 | {'canonical': 25, 'other': 17, 'generic': 25, 'length': 5} |
| q27 | open_cross_2 | {'canonical': 37, 'generic': 20, 'length': 5, 'other': 10} |
| q27 | rule_none | {'generic': 61, 'other': 9, 'length': 2} |
| q27 | rule_name | {'canonical': 12, 'generic': 45, 'other': 12, 'length': 2, 'refusal': 1} |
| q27 | rule_cross_1 | {'generic': 50, 'other': 19, 'canonical': 2, 'length': 1} |
| q27 | rule_cross_2 | {'generic': 47, 'canonical': 7, 'other': 16, 'length': 2} |
| q27 | rule_name_self | {'generic': 47, 'canonical': 10, 'length': 3, 'other': 12} |
| gemma | open_none | {'generic': 35, 'canonical': 18, 'other': 19} |
| gemma | open_name | {'canonical': 66, 'generic': 6} |
| gemma | open_cross_1 | {'generic': 26, 'other': 24, 'canonical': 22} |
| gemma | open_cross_2 | {'generic': 19, 'canonical': 47, 'other': 6} |
| gemma | rule_none | {'generic': 65, 'other': 7} |
| gemma | rule_name | {'other': 3, 'canonical': 40, 'generic': 28, 'undecided': 1} |
| gemma | rule_cross_1 | {'generic': 57, 'other': 14, 'canonical': 1} |
| gemma | rule_cross_2 | {'other': 10, 'generic': 56, 'canonical': 6} |
| gemma | rule_name_self | {'other': 11, 'generic': 34, 'canonical': 27} |
| gpt | open_none | {'canonical': 15, 'generic': 14, 'other': 7} |
| gpt | open_name | {'generic': 8, 'canonical': 26, 'other': 2} |
| gpt | open_cross_1 | {'canonical': 18, 'generic': 13, 'other': 5} |
| gpt | open_cross_2 | {'other': 5, 'canonical': 18, 'generic': 13} |
| gpt | rule_none | {'generic': 30, 'other': 6} |
| gpt | rule_name | {'generic': 30, 'other': 6} |
| gpt | rule_cross_1 | {'generic': 29, 'other': 7} |
| gpt | rule_cross_2 | {'generic': 30, 'other': 6} |
| gpt | rule_name_self | {'generic': 30, 'other': 6} |
| gemini | open_none | {'generic': 13, 'canonical': 13, 'other': 10} |
| gemini | open_name | {'generic': 8, 'canonical': 21, 'other': 7} |
| gemini | open_cross_1 | {'canonical': 12, 'generic': 13, 'other': 11} |
| gemini | open_cross_2 | {'canonical': 16, 'other': 9, 'generic': 11} |
| gemini | rule_none | {'generic': 24, 'other': 9, 'length': 1, 'canonical': 2} |
| gemini | rule_name | {'other': 9, 'generic': 19, 'length': 3, 'canonical': 5} |
| gemini | rule_cross_1 | {'generic': 21, 'other': 11, 'canonical': 4} |
| gemini | rule_cross_2 | {'length': 1, 'other': 12, 'generic': 18, 'canonical': 5} |
| gemini | rule_name_self | {'generic': 17, 'other': 13, 'canonical': 6} |
