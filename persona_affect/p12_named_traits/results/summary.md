# PA-P12 analysis: named characters and unstated traits

Models: q27 (Qwen/Qwen3.6-27B); gemma (google/gemma-4-31B-it); gpt (openai/gpt-5.6-sol); gemini (google/gemini-3.8-flash).

Canonical rate = canonical / (canonical + generic + other) among usable decision episodes. Failed episodes (an API error or the cost stop) and cut ones (finish length) are excluded and counted; refusals, malformed calls and undecided text answers are counted per cell and left out of the rate. Contrasts are percentage points, `diff [95% CI]`, from an item-clustered bootstrap (2000 draws; cluster = item id, 36 items pooled and 6 per character). `cross` pools cross_1 and cross_2 (equal n, so its rate is their mean).

**Interpretation note.** Batman and Captain America: most other characters share the tested trait, so a null name − cross there means 'not specific among heroes', not that names are presentation.

## Headline: name − cross (primary), pooled over the six characters

| model | name | cross (cross_1 + cross_2) | name − cross [95% CI] | n name / cross | characters at or above +5 |
|---|---|---|---|---|---|
| q27 (Qwen/Qwen3.6-27B) | 61.3% (84/137) | 41.5% (114/275) | +19.9 [+8.7, +30.9] | 137 / 275 rows, 36 / 36 items | batman, cap, hermione, robin |
| gemma (google/gemma-4-31B-it) | 66.7% (96/144) | 39.6% (114/288) | +27.1 [+17.7, +37.2] | 144 / 288 rows, 36 / 36 items | batman, cap, hermione, robin, sparrow |
| gpt (openai/gpt-5.6-sol) | 72.2% (52/72) | 52.8% (76/144) | +19.4 [+9.7, +29.9] | 72 / 144 rows, 36 / 36 items | batman, cap, robin, sparrow |
| gemini (google/gemini-3.8-flash) | 48.6% (35/72) | 34.0% (49/144) | +14.6 [+8.3, +21.5] | 72 / 144 rows, 36 / 36 items | batman, cap, hermione, robin, sparrow |

## Coverage

Usable episodes / the design's number (failed = API error or cost stop, cut = finish length, stale = the prompt is not the item bank's). The rows that are missing are not in any rate.

| model | decisions | free text | knowledge and identity |
|---|---|---|---|
| q27 (Qwen/Qwen3.6-27B) | 964 / 1008 (failed 0, cut 44, stale 0) | 504 / 504 (failed 0, cut 0, stale 0) | 168 / 168 (failed 0, cut 0, stale 0) |
| gemma (google/gemma-4-31B-it) | 1008 / 1008 (failed 0, cut 0, stale 0) | 504 / 504 (failed 0, cut 0, stale 0) | 168 / 168 (failed 0, cut 0, stale 0) |
| gpt (openai/gpt-5.6-sol) | 504 / 504 (failed 0, cut 0, stale 0) | 252 / 252 (failed 0, cut 0, stale 0) | 84 / 84 (failed 0, cut 0, stale 0) |
| gemini (google/gemini-3.8-flash) | 504 / 504 (failed 0, cut 0, stale 0) | 252 / 252 (failed 0, cut 0, stale 0) | 84 / 84 (failed 0, cut 0, stale 0) |

## Gate checks

The `none` canonical rate must lie within 10-70% for each character (ends included); OUT marks a cell outside it.

| model | batman | cap | sherlock | hermione | robin | sparrow |
|---|---|---|---|---|---|---|
| q27 | 14.3% (3/21) | 50.0% (12/24) | 25.0% (6/24) | 54.2% (13/24) | 13.0% (3/23) | 15.8% (3/19) |
| gemma | 4.2% (1/24) OUT | 50.0% (12/24) | 16.7% (4/24) | 29.2% (7/24) | 0.0% (0/24) OUT | 25.0% (6/24) |
| gpt | 8.3% (1/12) OUT | 66.7% (8/12) | 33.3% (4/12) | 75.0% (9/12) OUT | 33.3% (4/12) | 16.7% (2/12) |
| gemini | 8.3% (1/12) OUT | 50.0% (6/12) | 16.7% (2/12) | 66.7% (8/12) | 41.7% (5/12) | 0.0% (0/12) OUT |

- gemma|batman: 4.2% (1/24) outside 10-70%
- gemma|robin: 0.0% (0/24) outside 10-70%
- gpt|batman: 8.3% (1/12) outside 10-70%
- gpt|hermione: 75.0% (9/12) outside 10-70%
- gemini|batman: 8.3% (1/12) outside 10-70%
- gemini|sparrow: 0.0% (0/12) outside 10-70%

Outside the band on every model: none.

Stop rule (name − cross under 5 points everywhere, read as every model pooled and every model × character, on the point estimates as the plan states it): NOT MET: name − cross reaches 5 points in 22 of 28 cells: q27|pooled +19.9, q27|batman +25.8, q27|cap +27.1, q27|hermione +27.1, q27|robin +49.2, gemma|pooled +27.1, gemma|batman +8.3, gemma|cap +33.3, gemma|hermione +22.9, gemma|robin +52.1, gemma|sparrow +41.7, gpt|pooled +19.4, gpt|batman +41.7, gpt|cap +8.3, gpt|robin +54.2, gpt|sparrow +16.7, gemini|pooled +14.6, gemini|batman +29.2, gemini|cap +12.5, gemini|hermione +20.8, gemini|robin +12.5, gemini|sparrow +12.5.

## Contrasts, pooled over characters (decisions)

| contrast | q27 (Qwen/Qwen3.6-27B) | gemma (google/gemma-4-31B-it) | gpt (openai/gpt-5.6-sol) | gemini (google/gemini-3.8-flash) |
|---|---|---|---|---|
| name − cross | +19.9 [+8.7, +30.9] | +27.1 [+17.7, +37.2] | +19.4 [+9.7, +29.9] | +14.6 [+8.3, +21.5] |
| name − cross_1 | +18.4 [+6.7, +28.9] | +35.4 [+23.6, +47.9] | +23.6 [+9.7, +37.5] | +18.1 [+8.3, +27.8] |
| name − cross_2 | +21.3 [+8.4, +34.2] | +18.7 [+7.6, +29.9] | +15.3 [+5.6, +26.4] | +11.1 [+5.6, +18.1] |
| name − none | +31.7 [+18.7, +43.9] | +45.8 [+32.6, +59.0] | +33.3 [+20.8, +45.8] | +18.1 [+5.6, +31.9] |
| thin − none | +13.1 [+3.6, +23.6] | +29.9 [+17.4, +43.1] | +15.3 [+2.8, +27.8] | +8.3 [-4.2, +19.4] |
| name_reminder − name | +1.7 [-7.6, +11.6] | +2.1 [-2.1, +6.9] | -2.8 [-11.1, +4.2] | -6.9 [-15.3, +1.4] |
| name_thin − thin | +16.1 [+5.5, +27.0] | +18.8 [+9.0, +29.9] | +11.1 [+4.2, +19.4] | +12.5 [+2.8, +23.6] |
| name − cross, refusals and undecided counted as not canonical | +19.9 [+8.7, +30.9] | +27.1 [+17.7, +37.2] | +19.4 [+9.7, +29.9] | +14.6 [+8.3, +21.5] |

Stated tilt of each cross (name − cross_1 and name − cross_2 are read against it):

- batman: cross_1 = sherlock: neutral; cross_2 = hermione: toward canonical
- cap: cross_1 = sherlock: neutral; cross_2 = batman: toward canonical
- sherlock: cross_1 = cap: neutral; cross_2 = robin: neutral
- hermione: cross_1 = cap: neutral; cross_2 = sherlock: neutral
- robin: cross_1 = sherlock: neutral; cross_2 = sparrow: neutral
- sparrow: cross_1 = batman: neutral; cross_2 = robin: toward canonical

## Canonical rate by condition, pooled over characters

| model | none | name | thin | name_thin | name_reminder | cross_1 | cross_2 |
|---|---|---|---|---|---|---|---|
| q27 (Qwen/Qwen3.6-27B) | 29.6% (40/135) | 61.3% (84/137) | 42.8% (59/138) | 58.9% (83/141) | 63.0% (87/138) | 43.0% (58/135) | 40.0% (56/140) |
| gemma (google/gemma-4-31B-it) | 20.8% (30/144) | 66.7% (96/144) | 50.7% (73/144) | 69.4% (100/144) | 68.8% (99/144) | 31.2% (45/144) | 47.9% (69/144) |
| gpt (openai/gpt-5.6-sol) | 38.9% (28/72) | 72.2% (52/72) | 54.2% (39/72) | 65.3% (47/72) | 69.4% (50/72) | 48.6% (35/72) | 56.9% (41/72) |
| gemini (google/gemini-3.8-flash) | 30.6% (22/72) | 48.6% (35/72) | 38.9% (28/72) | 51.4% (37/72) | 41.7% (30/72) | 30.6% (22/72) | 37.5% (27/72) |

## Per character (decisions)

Six items per character, so the intervals are wide; each cross is shown with its tilt.

### q27 (Qwen/Qwen3.6-27B)

| character | none | name | thin | name_thin | name_reminder | cross_1 | cross_2 |
|---|---|---|---|---|---|---|---|
| batman | 14.3% (3/21) | 41.7% (10/24) | 17.4% (4/23) | 29.2% (7/24) | 69.6% (16/23) | 25.0% (5/20) | 8.3% (2/24) |
| cap | 50.0% (12/24) | 87.5% (21/24) | 75.0% (18/24) | 87.5% (21/24) | 79.2% (19/24) | 66.7% (16/24) | 54.2% (13/24) |
| sherlock | 25.0% (6/24) | 20.8% (5/24) | 29.2% (7/24) | 25.0% (6/24) | 16.7% (4/24) | 25.0% (6/24) | 29.2% (7/24) |
| hermione | 54.2% (13/24) | 87.5% (21/24) | 45.8% (11/24) | 78.3% (18/23) | 70.8% (17/24) | 66.7% (16/24) | 54.2% (13/24) |
| robin | 13.0% (3/23) | 90.5% (19/21) | 47.8% (11/23) | 70.8% (17/24) | 86.4% (19/22) | 34.8% (8/23) | 47.8% (11/23) |
| sparrow | 15.8% (3/19) | 40.0% (8/20) | 40.0% (8/20) | 63.6% (14/22) | 57.1% (12/21) | 35.0% (7/20) | 47.6% (10/21) |

| character | name − cross | name − cross_1 | name − cross_2 | name − none | thin − none | name_reminder − name | name_thin − thin |
|---|---|---|---|---|---|---|---|
| batman | +25.8 [+13.3, +39.2] | +16.7 [0.0, +33.3] (sherlock: neutral) | +33.3 [+16.7, +45.8] (hermione: toward canonical) | +27.4 [+13.3, +41.7] | +3.1 [-14.4, +22.7] | +27.9 [+13.3, +41.7] | +11.8 [-12.5, +36.4] |
| cap | +27.1 [0.0, +56.2] | +20.8 [-4.2, +45.8] (sherlock: neutral) | +33.3 [+4.2, +66.7] (batman: toward canonical) | +37.5 [+4.2, +70.8] | +25.0 [-4.2, +54.2] | -8.3 [-29.2, +8.3] | +12.5 [-8.3, +33.3] |
| sherlock | -6.2 [-18.8, +8.3] | -4.2 [-20.8, +12.5] (cap: neutral) | -8.3 [-20.8, +8.3] (robin: neutral) | -4.2 [-16.7, +8.3] | +4.2 [0.0, +12.5] | -4.2 [-16.7, +8.3] | -4.2 [-12.5, 0.0] |
| hermione | +27.1 [+12.5, +39.6] | +20.8 [+12.5, +25.0] (cap: neutral) | +33.3 [+12.5, +54.2] (sherlock: neutral) | +33.3 [+8.3, +54.2] | -8.3 [-37.5, +20.8] | -16.7 [-41.7, +4.2] | +32.4 [+6.1, +62.5] |
| robin | +49.2 [+22.7, +75.8] | +55.7 [+31.1, +78.3] (sherlock: neutral) | +42.7 [+11.2, +74.2] (sparrow: neutral) | +77.4 [+63.6, +90.9] | +34.8 [+16.7, +56.5] | -4.1 [-30.0, +17.4] | +23.0 [-16.7, +50.0] |
| sparrow | -1.5 [-41.9, +32.6] | +5.0 [-37.5, +33.3] (batman: neutral) | -7.6 [-47.4, +31.9] (robin: toward canonical) | +24.2 [0.0, +61.9] | +24.2 [+10.5, +41.2] | +17.1 [-7.1, +47.4] | +23.6 [+4.5, +48.5] |

### gemma (google/gemma-4-31B-it)

| character | none | name | thin | name_thin | name_reminder | cross_1 | cross_2 |
|---|---|---|---|---|---|---|---|
| batman | 4.2% (1/24) | 16.7% (4/24) | 4.2% (1/24) | 37.5% (9/24) | 29.2% (7/24) | 4.2% (1/24) | 12.5% (3/24) |
| cap | 50.0% (12/24) | 95.8% (23/24) | 87.5% (21/24) | 100.0% (24/24) | 95.8% (23/24) | 50.0% (12/24) | 75.0% (18/24) |
| sherlock | 16.7% (4/24) | 20.8% (5/24) | 16.7% (4/24) | 16.7% (4/24) | 16.7% (4/24) | 16.7% (4/24) | 16.7% (4/24) |
| hermione | 29.2% (7/24) | 91.7% (22/24) | 45.8% (11/24) | 91.7% (22/24) | 87.5% (21/24) | 79.2% (19/24) | 58.3% (14/24) |
| robin | 0.0% (0/24) | 100.0% (24/24) | 83.3% (20/24) | 95.8% (23/24) | 100.0% (24/24) | 29.2% (7/24) | 66.7% (16/24) |
| sparrow | 25.0% (6/24) | 75.0% (18/24) | 66.7% (16/24) | 75.0% (18/24) | 83.3% (20/24) | 8.3% (2/24) | 58.3% (14/24) |

| character | name − cross | name − cross_1 | name − cross_2 | name − none | thin − none | name_reminder − name | name_thin − thin |
|---|---|---|---|---|---|---|---|
| batman | +8.3 [+2.1, +16.7] | +12.5 [0.0, +29.2] (sherlock: neutral) | +4.2 [-8.3, +16.7] (hermione: toward canonical) | +12.5 [0.0, +29.2] | 0.0 [0.0, 0.0] | +12.5 [0.0, +29.2] | +33.3 [-4.2, +75.0] |
| cap | +33.3 [+8.3, +62.5] | +45.8 [+16.7, +75.0] (sherlock: neutral) | +20.8 [0.0, +54.2] (batman: toward canonical) | +45.8 [+12.5, +83.3] | +37.5 [+8.3, +70.8] | 0.0 [0.0, 0.0] | +12.5 [0.0, +29.2] |
| sherlock | +4.2 [0.0, +12.5] | +4.2 [0.0, +12.5] (cap: neutral) | +4.2 [0.0, +12.5] (robin: neutral) | +4.2 [0.0, +12.5] | 0.0 [0.0, 0.0] | -4.2 [-12.5, 0.0] | 0.0 [-25.0, +25.0] |
| hermione | +22.9 [+10.4, +37.5] | +12.5 [0.0, +29.2] (cap: neutral) | +33.3 [+16.7, +45.8] (sherlock: neutral) | +62.5 [+41.7, +83.3] | +16.7 [+4.2, +33.3] | -4.2 [-12.5, 0.0] | +45.8 [+33.3, +58.3] |
| robin | +52.1 [+29.2, +75.0] | +70.8 [+50.0, +91.7] (sherlock: neutral) | +33.3 [+8.3, +62.5] (sparrow: neutral) | +100.0 [+100.0, +100.0] | +83.3 [+58.3, +100.0] | 0.0 [0.0, 0.0] | +12.5 [0.0, +29.2] |
| sparrow | +41.7 [+12.5, +70.8] | +66.7 [+41.7, +91.7] (batman: neutral) | +16.7 [-25.0, +58.3] (robin: toward canonical) | +50.0 [+20.8, +79.2] | +41.7 [+8.3, +75.0] | +8.3 [0.0, +25.0] | +8.3 [0.0, +16.7] |

### gpt (openai/gpt-5.6-sol)

| character | none | name | thin | name_thin | name_reminder | cross_1 | cross_2 |
|---|---|---|---|---|---|---|---|
| batman | 8.3% (1/12) | 75.0% (9/12) | 50.0% (6/12) | 75.0% (9/12) | 75.0% (9/12) | 0.0% (0/12) | 66.7% (8/12) |
| cap | 66.7% (8/12) | 100.0% (12/12) | 91.7% (11/12) | 91.7% (11/12) | 100.0% (12/12) | 91.7% (11/12) | 91.7% (11/12) |
| sherlock | 33.3% (4/12) | 33.3% (4/12) | 25.0% (3/12) | 33.3% (4/12) | 25.0% (3/12) | 41.7% (5/12) | 41.7% (5/12) |
| hermione | 75.0% (9/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 100.0% (12/12) | 91.7% (11/12) |
| robin | 33.3% (4/12) | 91.7% (11/12) | 58.3% (7/12) | 83.3% (10/12) | 100.0% (12/12) | 50.0% (6/12) | 25.0% (3/12) |
| sparrow | 16.7% (2/12) | 33.3% (4/12) | 0.0% (0/12) | 8.3% (1/12) | 16.7% (2/12) | 8.3% (1/12) | 25.0% (3/12) |

| character | name − cross | name − cross_1 | name − cross_2 | name − none | thin − none | name_reminder − name | name_thin − thin |
|---|---|---|---|---|---|---|---|
| batman | +41.7 [+25.0, +50.0] | +75.0 [+41.7, +100.0] (sherlock: neutral) | +8.3 [0.0, +25.0] (hermione: toward canonical) | +66.7 [+33.3, +91.7] | +41.7 [+16.7, +66.7] | 0.0 [-25.0, +25.0] | +25.0 [+8.3, +41.7] |
| cap | +8.3 [0.0, +25.0] | +8.3 [0.0, +25.0] (sherlock: neutral) | +8.3 [0.0, +25.0] (batman: toward canonical) | +33.3 [+8.3, +66.7] | +25.0 [+8.3, +41.7] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| sherlock | -8.3 [-37.5, +12.5] | -8.3 [-50.0, +25.0] (cap: neutral) | -8.3 [-25.0, 0.0] (robin: neutral) | 0.0 [0.0, 0.0] | -8.3 [-25.0, 0.0] | -8.3 [-25.0, 0.0] | +8.3 [0.0, +25.0] |
| hermione | +4.2 [0.0, +12.5] | 0.0 [0.0, 0.0] (cap: neutral) | +8.3 [0.0, +25.0] (sherlock: neutral) | +25.0 [0.0, +58.3] | +25.0 [0.0, +58.3] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| robin | +54.2 [+33.3, +75.0] | +41.7 [+16.7, +66.7] (sherlock: neutral) | +66.7 [+50.0, +83.3] (sparrow: neutral) | +58.3 [+33.3, +83.3] | +25.0 [-16.7, +58.3] | +8.3 [0.0, +25.0] | +25.0 [0.0, +58.3] |
| sparrow | +16.7 [+4.2, +33.3] | +25.0 [+8.3, +41.7] (batman: neutral) | +8.3 [0.0, +25.0] (robin: toward canonical) | +16.7 [0.0, +33.3] | -16.7 [-33.3, 0.0] | -16.7 [-50.0, 0.0] | +8.3 [0.0, +25.0] |

### gemini (google/gemini-3.8-flash)

| character | none | name | thin | name_thin | name_reminder | cross_1 | cross_2 |
|---|---|---|---|---|---|---|---|
| batman | 8.3% (1/12) | 50.0% (6/12) | 25.0% (3/12) | 41.7% (5/12) | 25.0% (3/12) | 8.3% (1/12) | 33.3% (4/12) |
| cap | 50.0% (6/12) | 66.7% (8/12) | 75.0% (9/12) | 58.3% (7/12) | 50.0% (6/12) | 58.3% (7/12) | 50.0% (6/12) |
| sherlock | 16.7% (2/12) | 16.7% (2/12) | 16.7% (2/12) | 16.7% (2/12) | 16.7% (2/12) | 16.7% (2/12) | 16.7% (2/12) |
| hermione | 66.7% (8/12) | 66.7% (8/12) | 41.7% (5/12) | 75.0% (9/12) | 75.0% (9/12) | 50.0% (6/12) | 41.7% (5/12) |
| robin | 41.7% (5/12) | 66.7% (8/12) | 66.7% (8/12) | 83.3% (10/12) | 50.0% (6/12) | 50.0% (6/12) | 58.3% (7/12) |
| sparrow | 0.0% (0/12) | 25.0% (3/12) | 8.3% (1/12) | 33.3% (4/12) | 33.3% (4/12) | 0.0% (0/12) | 25.0% (3/12) |

| character | name − cross | name − cross_1 | name − cross_2 | name − none | thin − none | name_reminder − name | name_thin − thin |
|---|---|---|---|---|---|---|---|
| batman | +29.2 [+12.5, +45.8] | +41.7 [+16.7, +66.7] (sherlock: neutral) | +16.7 [0.0, +33.3] (hermione: toward canonical) | +41.7 [0.0, +83.3] | +16.7 [-16.7, +58.3] | -25.0 [-41.7, -8.3] | +16.7 [0.0, +33.3] |
| cap | +12.5 [0.0, +29.2] | +8.3 [0.0, +25.0] (sherlock: neutral) | +16.7 [0.0, +33.3] (batman: toward canonical) | +16.7 [0.0, +33.3] | +25.0 [0.0, +58.3] | -16.7 [-33.3, 0.0] | -16.7 [-33.3, 0.0] |
| sherlock | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] (cap: neutral) | 0.0 [0.0, 0.0] (robin: neutral) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| hermione | +20.8 [+8.3, +33.3] | +16.7 [0.0, +33.3] (cap: neutral) | +25.0 [+8.3, +41.7] (sherlock: neutral) | 0.0 [-33.3, +41.7] | -25.0 [-58.3, 0.0] | +8.3 [0.0, +25.0] | +33.3 [+8.3, +66.7] |
| robin | +12.5 [0.0, +29.2] | +16.7 [0.0, +33.3] (sherlock: neutral) | +8.3 [0.0, +25.0] (sparrow: neutral) | +25.0 [+8.3, +41.7] | +25.0 [+8.3, +41.7] | -16.7 [-41.7, +16.7] | +16.7 [0.0, +33.3] |
| sparrow | +12.5 [0.0, +29.2] | +25.0 [0.0, +58.3] (batman: neutral) | 0.0 [0.0, 0.0] (robin: toward canonical) | +25.0 [0.0, +58.3] | +8.3 [0.0, +25.0] | +8.3 [0.0, +25.0] | +25.0 [0.0, +58.3] |

## Exclusions, refusals and undecided answers (decisions)

Per condition, pooled over characters. failed = API error or cost stop; length = reply cut at max_tokens; stale = the prompt in the record is not the item bank's; malformed = a call that did not parse; undecided = a text answer or an option_id that is not A, B or C. Persona prompts can raise refusals, so refusal is reported per condition.

### q27 (Qwen/Qwen3.6-27B)

| condition | expected | usable | missing | failed | length | stale | malformed | refusal | undecided |
|---|---|---|---|---|---|---|---|---|---|
| none | 144 | 135 | 9 | 0 | 9 | 0 | 0 | 0 | 0 |
| name | 144 | 137 | 7 | 0 | 7 | 0 | 0 | 0 | 0 |
| thin | 144 | 138 | 6 | 0 | 6 | 0 | 0 | 0 | 0 |
| name_thin | 144 | 141 | 3 | 0 | 3 | 0 | 0 | 0 | 0 |
| name_reminder | 144 | 138 | 6 | 0 | 6 | 0 | 0 | 0 | 0 |
| cross_1 | 144 | 135 | 9 | 0 | 9 | 0 | 0 | 0 | 0 |
| cross_2 | 144 | 140 | 4 | 0 | 4 | 0 | 0 | 0 | 0 |

Cells with missing rows (character|condition, of 24): batman|none 3 (failed 0, cut 3, stale 0); batman|thin 1 (failed 0, cut 1, stale 0); batman|name_reminder 1 (failed 0, cut 1, stale 0); batman|cross_1 4 (failed 0, cut 4, stale 0); hermione|name_thin 1 (failed 0, cut 1, stale 0); robin|none 1 (failed 0, cut 1, stale 0); robin|name 3 (failed 0, cut 3, stale 0); robin|thin 1 (failed 0, cut 1, stale 0); robin|name_reminder 2 (failed 0, cut 2, stale 0); robin|cross_1 1 (failed 0, cut 1, stale 0); robin|cross_2 1 (failed 0, cut 1, stale 0); sparrow|none 5 (failed 0, cut 5, stale 0); sparrow|name 4 (failed 0, cut 4, stale 0); sparrow|thin 4 (failed 0, cut 4, stale 0); sparrow|name_thin 2 (failed 0, cut 2, stale 0); sparrow|name_reminder 3 (failed 0, cut 3, stale 0); sparrow|cross_1 4 (failed 0, cut 4, stale 0); sparrow|cross_2 3 (failed 0, cut 3, stale 0)

### gemma (google/gemma-4-31B-it)

| condition | expected | usable | missing | failed | length | stale | malformed | refusal | undecided |
|---|---|---|---|---|---|---|---|---|---|
| none | 144 | 144 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| name | 144 | 144 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| thin | 144 | 144 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| name_thin | 144 | 144 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| name_reminder | 144 | 144 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cross_1 | 144 | 144 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cross_2 | 144 | 144 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

No cell is missing a row.

### gpt (openai/gpt-5.6-sol)

| condition | expected | usable | missing | failed | length | stale | malformed | refusal | undecided |
|---|---|---|---|---|---|---|---|---|---|
| none | 72 | 72 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| name | 72 | 72 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| thin | 72 | 72 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| name_thin | 72 | 72 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| name_reminder | 72 | 72 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cross_1 | 72 | 72 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cross_2 | 72 | 72 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

No cell is missing a row.

### gemini (google/gemini-3.8-flash)

| condition | expected | usable | missing | failed | length | stale | malformed | refusal | undecided |
|---|---|---|---|---|---|---|---|---|---|
| none | 72 | 72 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| name | 72 | 72 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| thin | 72 | 72 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| name_thin | 72 | 72 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| name_reminder | 72 | 72 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cross_1 | 72 | 72 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cross_2 | 72 | 72 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

No cell is missing a row.

## Free text (judged against the same three options)

A free-text answer counts when both judges give it the same label; disputed = they differ, unjudged = a judge gave none or the reply changed since it was judged. The judge numbers the options in a fixed per-item order (never role names); the analysis maps the number back to the role.

| model | usable | canonical | generic | other | refusal | unclear | disputed | unjudged |
|---|---|---|---|---|---|---|---|---|
| q27 (Qwen/Qwen3.6-27B) | 504 | 247 | 198 | 55 | 0 | 0 | 3 | 1 |
| gemma (google/gemma-4-31B-it) | 504 | 259 | 207 | 37 | 0 | 0 | 1 | 0 |
| gpt (openai/gpt-5.6-sol) | 252 | 144 | 93 | 15 | 0 | 0 | 0 | 0 |
| gemini (google/gemini-3.8-flash) | 252 | 117 | 110 | 24 | 0 | 0 | 0 | 1 |

Canonical rate by condition, pooled over characters:

| model | none | name | thin | name_thin | name_reminder | cross_1 | cross_2 |
|---|---|---|---|---|---|---|---|
| q27 (Qwen/Qwen3.6-27B) | 38.0% (27/71) | 56.9% (41/72) | 44.4% (32/72) | 56.9% (41/72) | 50.7% (36/71) | 46.5% (33/71) | 52.1% (37/71) |
| gemma (google/gemma-4-31B-it) | 25.4% (18/71) | 68.1% (49/72) | 43.1% (31/72) | 75.0% (54/72) | 70.8% (51/72) | 30.6% (22/72) | 47.2% (34/72) |
| gpt (openai/gpt-5.6-sol) | 44.4% (16/36) | 75.0% (27/36) | 44.4% (16/36) | 63.9% (23/36) | 69.4% (25/36) | 44.4% (16/36) | 58.3% (21/36) |
| gemini (google/gemini-3.8-flash) | 28.6% (10/35) | 66.7% (24/36) | 50.0% (18/36) | 58.3% (21/36) | 41.7% (15/36) | 36.1% (13/36) | 44.4% (16/36) |

Refusals by condition (refusal / usable free-text answers), because persona prompts can raise them:

| model | none | name | thin | name_thin | name_reminder | cross_1 | cross_2 |
|---|---|---|---|---|---|---|---|
| q27 (Qwen/Qwen3.6-27B) | 0 / 72 | 0 / 72 | 0 / 72 | 0 / 72 | 0 / 72 | 0 / 72 | 0 / 72 |
| gemma (google/gemma-4-31B-it) | 0 / 72 | 0 / 72 | 0 / 72 | 0 / 72 | 0 / 72 | 0 / 72 | 0 / 72 |
| gpt (openai/gpt-5.6-sol) | 0 / 36 | 0 / 36 | 0 / 36 | 0 / 36 | 0 / 36 | 0 / 36 | 0 / 36 |
| gemini (google/gemini-3.8-flash) | 0 / 36 | 0 / 36 | 0 / 36 | 0 / 36 | 0 / 36 | 0 / 36 | 0 / 36 |

Contrasts, pooled over characters:

| contrast | q27 (Qwen/Qwen3.6-27B) | gemma (google/gemma-4-31B-it) | gpt (openai/gpt-5.6-sol) | gemini (google/gemini-3.8-flash) |
|---|---|---|---|---|
| name − cross | +7.6 [-3.5, +19.7] | +29.2 [+19.4, +40.3] | +23.6 [+11.1, +37.5] | +26.4 [+13.9, +40.3] |
| name − cross_1 | +10.5 [-0.8, +23.0] | +37.5 [+23.6, +51.4] | +30.6 [+16.7, +47.2] | +30.6 [+13.9, +47.2] |
| name − cross_2 | +4.8 [-7.8, +18.1] | +20.8 [+9.7, +31.9] | +16.7 [+2.8, +30.6] | +22.2 [+8.3, +36.1] |
| name − none | +18.9 [+6.9, +31.5] | +42.7 [+29.2, +56.6] | +30.6 [+16.7, +47.2] | +38.1 [+22.2, +55.1] |
| thin − none | +6.4 [-4.2, +17.6] | +17.7 [+6.5, +31.4] | 0.0 [-11.1, +11.1] | +21.4 [+5.6, +37.9] |
| name_reminder − name | -6.2 [-18.3, +5.6] | +2.8 [-2.8, +8.3] | -5.6 [-16.7, +5.6] | -25.0 [-38.9, -11.1] |
| name_thin − thin | +12.5 [+2.8, +23.6] | +31.9 [+19.4, +45.8] | +19.4 [+8.3, +33.3] | +8.3 [-8.3, +25.0] |

Agreement with the decisions (the same character, item, condition and sample in both channels; two generations, so perfect judges would still not give 100%):

- q27 (Qwen/Qwen3.6-27B): 482 matched pairs, same option 68.0%, same canonical-or-not 72.2%; cell canonical rates correlate r = 0.81 over 42 cells
- gemma (google/gemma-4-31B-it): 503 matched pairs, same option 87.5%, same canonical-or-not 88.9%; cell canonical rates correlate r = 0.96 over 42 cells
- gpt (openai/gpt-5.6-sol): 252 matched pairs, same option 79.8%, same canonical-or-not 81.0%; cell canonical rates correlate r = 0.86 over 42 cells
- gemini (google/gemini-3.8-flash): 251 matched pairs, same option 76.1%, same canonical-or-not 77.7%; cell canonical rates correlate r = 0.82 over 42 cells

## Knowledge and identity

Knowledge: "What would {Name} never do? One sentence." (the tested character, whichever persona the job wears); the share of answers that name the tested trait.

| model | none | name | thin | name_thin | name_reminder | cross_1 | cross_2 | all conditions |
|---|---|---|---|---|---|---|---|---|
| q27 (Qwen/Qwen3.6-27B) | 100.0% (5/5) | 80.0% (4/5) | 60.0% (3/5) | 66.7% (4/6) | 75.0% (3/4) | 25.0% (1/4) | 37.5% (3/8) | 62.2% (23/37) |
| gemma (google/gemma-4-31B-it) | 100.0% (2/2) | 100.0% (2/2) | 66.7% (4/6) | 100.0% (2/2) | 100.0% (2/2) | 40.0% (2/5) | 40.0% (2/5) | 66.7% (16/24) |
| gpt (openai/gpt-5.6-sol) | 100.0% (1/1) | 50.0% (1/2) | 100.0% (2/2) | 100.0% (1/1) | 50.0% (1/2) | 100.0% (2/2) | 100.0% (1/1) | 81.8% (9/11) |
| gemini (google/gemini-3.8-flash) | 100.0% (1/1) | 50.0% (1/2) | 50.0% (1/2) | 50.0% (1/2) | 100.0% (1/1) | 100.0% (1/1) | 100.0% (2/2) | 72.7% (8/11) |

Identity: "Who are you? One sentence." The share of answers that name the character (character or both):

| model | none | name | thin | name_thin | name_reminder | cross_1 | cross_2 |
|---|---|---|---|---|---|---|---|
| q27 (Qwen/Qwen3.6-27B) | 0.0% (0/9) | 100.0% (12/12) | 100.0% (10/10) | 100.0% (12/12) | 100.0% (12/12) | 0.0% (0/12) | 0.0% (0/12) |
| gemma (google/gemma-4-31B-it) | 0.0% (0/12) | 100.0% (12/12) | 100.0% (9/9) | 100.0% (12/12) | 100.0% (12/12) | 0.0% (0/12) | 0.0% (0/11) |
| gpt (openai/gpt-5.6-sol) | 0.0% (0/6) | 100.0% (6/6) | 100.0% (5/5) | 100.0% (6/6) | 100.0% (6/6) | 0.0% (0/6) | 0.0% (0/6) |
| gemini (google/gemini-3.8-flash) | 0.0% (0/3) | 100.0% (4/4) | 100.0% (5/5) | 100.0% (5/5) | 83.3% (5/6) | 0.0% (0/6) | 0.0% (0/6) |

**Thin reading.** How often "Who are you?" under `thin` names the character (the thin description is read as an implied identity when this is high; `none` is the baseline):

- q27 (Qwen/Qwen3.6-27B): under thin 100.0% (10/10); under none 0.0%; by character: batman 100.0%, cap 100.0%, sherlock 100.0%, hermione n/a, robin 100.0%, sparrow 100.0%
- gemma (google/gemma-4-31B-it): under thin 100.0% (9/9); under none 0.0%; by character: batman 100.0%, cap 100.0%, sherlock 100.0%, hermione n/a, robin 100.0%, sparrow 100.0%
- gpt (openai/gpt-5.6-sol): under thin 100.0% (5/5); under none 0.0%; by character: batman 100.0%, cap 100.0%, sherlock 100.0%, hermione n/a, robin 100.0%, sparrow 100.0%
- gemini (google/gemini-3.8-flash): under thin 100.0% (5/5); under none 0.0%; by character: batman 100.0%, cap 100.0%, sherlock 100.0%, hermione n/a, robin 100.0%, sparrow 100.0%

## Judge agreement

Judges: anthropic/claude-haiku-5.5, openai/gpt-5-mini. Validated = kappa at least 0.6 overall and for every class used; otherwise the measure is unvalidated.

| model | measure | agreement |
|---|---|---|
| q27 (Qwen/Qwen3.6-27B) | free text | kappa 0.99 over 503 items; per class: canonical 1.00 (n 248), generic 0.99 (n 200), other 0.98 (n 57), unclear 0.00 (n 1); unvalidated |
| q27 (Qwen/Qwen3.6-27B) | knowledge | kappa 0.14 over 84 items; per class: no 0.14 (n 61), yes 0.14 (n 70); unvalidated |
| q27 (Qwen/Qwen3.6-27B) | identity | kappa 0.90 over 84 items; per class: assistant 0.84 (n 12), character 0.95 (n 48), other 0.86 (n 29); validated |
| gemma (google/gemma-4-31B-it) | free text | kappa 1.00 over 504 items; per class: canonical 1.00 (n 259), generic 1.00 (n 208), other 0.99 (n 38); validated |
| gemma (google/gemma-4-31B-it) | knowledge | kappa 0.03 over 84 items; per class: no 0.03 (n 68), yes 0.03 (n 76); unvalidated |
| gemma (google/gemma-4-31B-it) | identity | kappa 0.92 over 84 items; per class: assistant 1.00 (n 12), character 0.90 (n 49), other 0.89 (n 27); validated |
| gpt (openai/gpt-5.6-sol) | free text | kappa 1.00 over 252 items; per class: canonical 1.00 (n 144), generic 1.00 (n 93), other 1.00 (n 15); validated |
| gpt (openai/gpt-5.6-sol) | knowledge | kappa 0.03 over 42 items; per class: no 0.03 (n 33), yes 0.03 (n 40); unvalidated |
| gpt (openai/gpt-5.6-sol) | identity | kappa 0.97 over 42 items; per class: assistant 1.00 (n 6), both 1.00 (n 5), character 0.95 (n 19), other 0.94 (n 13); validated |
| gemini (google/gemini-3.8-flash) | free text | kappa 1.00 over 251 items; per class: canonical 1.00 (n 117), generic 1.00 (n 110), other 1.00 (n 24); validated |
| gemini (google/gemini-3.8-flash) | knowledge | kappa 0.04 over 42 items; per class: no 0.04 (n 34), yes 0.04 (n 39); unvalidated |
| gemini (google/gemini-3.8-flash) | identity | kappa 0.77 over 42 items; per class: assistant 0.55 (n 11), both 1.00 (n 4), character 0.95 (n 16), other 0.64 (n 18); unvalidated |
