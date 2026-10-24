# SC-M7 summary (frozen analysis)

far = rungs 4-5 (3-4 steps from the default), near = rungs 2-3.

## Verdicts

| model | axis | dA_far | dS_far | gap_far | trend (far - near) | verdict | words check |
|---|---|---|---|---|---|---|---|
| q9 | E | +73.4 [+57.0, +87.5] | +96.1 [+94.5, +98.4] | +22.7 [+7.8, +38.3] | -14.6 [-32.7, +3.1] | **stated runs ahead** | stated follows the attribution, not just the words (mention < half of self, CI excludes 0) |
| q9 | H | +69.1 [+43.4, +91.2] | +91.4 [+87.5, +95.3] | +22.4 [-0.5, +48.4] | +7.7 [-9.7, +27.9] | **inconclusive** | stated moves with the words (mention >= half of self): the boring explanation is not excluded |
| q27 | E | +73.4 [+50.0, +91.4] | +100.0 [+100.0, +100.0] | +26.6 [+8.6, +50.0] | -28.1 [-45.3, -10.2] | **stated runs ahead** | stated follows the attribution, not just the words (mention < half of self, CI excludes 0) |
| q27 | H | +88.3 [+77.3, +96.9] | +100.0 [+100.0, +100.0] | +11.7 [+3.1, +22.7] | +3.9 [-7.0, +16.4] | **inconclusive** | stated moves with the words (mention >= half of self): the boring explanation is not excluded |
| gemma | E | +43.8 [+12.5, +75.0] | +100.0 [+100.0, +100.0] | +56.2 [+25.0, +87.5] | +0.8 [-29.7, +28.1] | **stated runs ahead** | stated moves with the words (mention >= half of self): the boring explanation is not excluded |
| gemma | H | +100.0 [+100.0, +100.0] | +100.0 [+100.0, +100.0] | +0.0 [+0.0, +0.0] | -10.9 [-26.6, +0.0] | **stated tracks the act** | stated moves with the words (mention >= half of self): the boring explanation is not excluded |
| haiku | E | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | **induction too weak** | stated does not move under self (nothing to explain) |
| haiku | H | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -5.2 [-15.6, +0.0] | **induction too weak** | stated does not move under self (nothing to explain) |
| gemini | E | +40.6 [+12.5, +71.9] | +100.0 [+100.0, +100.0] | +59.4 [+28.1, +87.5] | +21.9 [-7.8, +48.4] | **stated runs ahead** | stated follows the attribution, not just the words (mention < half of self, CI excludes 0) |
| gemini | H | +78.1 [+53.1, +95.3] | +18.9 [+6.2, +32.8] | -59.3 [-81.2, -31.0] | -26.5 [-42.2, -9.4] | **stated snaps back** | stated follows the attribution, not just the words (mention < half of self, CI excludes 0) |
| gpt | E | +68.8 [+50.0, +87.5] | +100.0 [+100.0, +100.0] | +31.2 [+12.5, +50.0] | -26.6 [-46.9, -4.7] | **stated runs ahead** | stated follows the attribution, not just the words (mention < half of self, CI excludes 0) |
| gpt | H | +6.2 [+0.0, +15.6] | +98.4 [+95.3, +100.0] | +92.2 [+81.2, +100.0] | +25.0 [+10.9, +40.6] | **stated runs ahead** | stated follows the attribution, not just the words (mention < half of self, CI excludes 0) |

## Pooled over models (equal weight)

- E: gap_far +32.7 [+15.9, +50.3]; verdicts {'stated runs ahead': 5, 'induction too weak': 1}
- H: gap_far +11.2 [+4.5, +19.4]; verdicts {'inconclusive': 2, 'stated tracks the act': 1, 'induction too weak': 1, 'stated snaps back': 1, 'stated runs ahead': 1}

## Shifts by rung (points, self - base)

- q9|E: k2: dA +57.8 [+35.9, +76.6], dS +77.7 [+67.2, +87.5]; k3: dA +31.2 [+10.9, +54.7], dS +85.9 [+75.0, +95.3]; k4: dA +68.8 [+48.4, +89.1], dS +92.2 [+89.1, +96.9]; k5: dA +78.1 [+57.8, +95.3], dS +100.0 [+100.0, +100.0]
- q9|H: k2: dA +56.2 [+34.4, +75.0], dS +80.8 [+73.4, +88.6]; k3: dA +95.3 [+89.1, +100.0], dS +100.0 [+100.0, +100.0]; k4: dA +58.5 [+28.8, +85.9], dS +90.6 [+84.4, +95.3]; k5: dA +79.6 [+54.6, +98.4], dS +92.2 [+89.1, +96.9]
- q27|E: k2: dA +54.7 [+26.6, +82.8], dS +100.0 [+100.0, +100.0]; k3: dA +32.8 [+14.1, +53.1], dS +96.9 [+92.2, +100.0]; k4: dA +70.3 [+46.9, +90.6], dS +100.0 [+100.0, +100.0]; k5: dA +76.6 [+50.0, +96.9], dS +100.0 [+100.0, +100.0]
- q27|H: k2: dA +78.1 [+59.4, +93.8], dS +90.6 [+81.2, +98.4]; k3: dA +95.3 [+85.9, +100.0], dS +98.4 [+95.3, +100.0]; k4: dA +82.8 [+62.5, +96.9], dS +100.0 [+100.0, +100.0]; k5: dA +93.8 [+84.4, +100.0], dS +100.0 [+100.0, +100.0]
- gemma|E: k2: dA +50.0 [+12.5, +87.5], dS +100.0 [+100.0, +100.0]; k3: dA +9.4 [+0.0, +28.1], dS +70.3 [+48.4, +90.6]; k4: dA +37.5 [+12.5, +68.8], dS +100.0 [+100.0, +100.0]; k5: dA +50.0 [+12.5, +87.5], dS +100.0 [+100.0, +100.0]
- gemma|H: k2: dA +90.6 [+78.1, +100.0], dS +100.0 [+100.0, +100.0]; k3: dA +87.5 [+62.5, +100.0], dS +100.0 [+100.0, +100.0]; k4: dA +100.0 [+100.0, +100.0], dS +100.0 [+100.0, +100.0]; k5: dA +100.0 [+100.0, +100.0], dS +100.0 [+100.0, +100.0]
- haiku|E: k2: dA +0.0 [+0.0, +0.0], dS +0.0 [+0.0, +0.0]; k3: dA +0.0 [+0.0, +0.0], dS +0.0 [+0.0, +0.0]; k4: dA +0.0 [+0.0, +0.0], dS +0.0 [+0.0, +0.0]; k5: dA +0.0 [+0.0, +0.0], dS +0.0 [+0.0, +0.0]
- haiku|H: k2: dA -10.4 [-31.2, +0.0], dS +0.0 [+0.0, +0.0]; k3: dA +0.0 [+0.0, +0.0], dS +0.0 [+0.0, +0.0]; k4: dA +0.0 [+0.0, +0.0], dS +0.0 [+0.0, +0.0]; k5: dA +0.0 [+0.0, +0.0], dS +0.0 [+0.0, +0.0]
- gemini|E: k2: dA +31.2 [+6.2, +62.5], dS +65.6 [+43.8, +87.5]; k3: dA +21.9 [+0.0, +50.0], dS +62.5 [+46.9, +81.2]; k4: dA +34.4 [+9.4, +68.8], dS +100.0 [+100.0, +100.0]; k5: dA +46.9 [+15.6, +78.1], dS +100.0 [+100.0, +100.0]
- gemini|H: k2: dA +84.4 [+59.4, +100.0], dS +50.0 [+31.2, +68.8]; k3: dA +81.2 [+56.2, +96.9], dS +50.0 [+31.2, +68.8]; k4: dA +84.4 [+59.4, +100.0], dS +9.6 [+3.1, +19.4]; k5: dA +71.9 [+43.8, +93.8], dS +28.1 [+9.4, +50.0]
- gpt|E: k2: dA +53.1 [+21.9, +81.2], dS +100.0 [+100.0, +100.0]; k3: dA +21.9 [+6.2, +43.8], dS +90.6 [+78.1, +100.0]; k4: dA +75.0 [+50.0, +93.8], dS +100.0 [+100.0, +100.0]; k5: dA +62.5 [+31.2, +87.5], dS +100.0 [+100.0, +100.0]
- gpt|H: k2: dA +37.5 [+18.8, +59.4], dS +100.0 [+100.0, +100.0]; k3: dA +25.0 [+6.2, +43.8], dS +96.9 [+90.6, +100.0]; k4: dA +0.0 [+0.0, +0.0], dS +96.9 [+90.6, +100.0]; k5: dA +12.5 [+0.0, +31.2], dS +100.0 [+100.0, +100.0]

## Mention (matched wording, someone else's belief), far

- q9|E: dA +38.3 [+26.6, +50.0], dS +19.9 [+9.4, +31.8]; self - mention: stated +76.2 [+63.3, +87.4], act +35.2 [+21.1, +49.2]
- q9|H: dA +59.1 [+36.7, +78.9], dS +46.1 [+22.7, +68.8]; self - mention: stated +45.3 [+21.9, +68.8], act +9.9 [+1.6, +19.6]
- q27|E: dA +47.7 [+32.0, +58.6], dS +42.2 [+31.2, +52.3]; self - mention: stated +57.8 [+47.7, +68.8], act +25.8 [+10.9, +40.6]
- q27|H: dA +35.9 [+21.1, +51.6], dS +87.5 [+74.2, +98.4]; self - mention: stated +12.5 [+1.6, +25.8], act +52.3 [+35.9, +69.5]
- gemma|E: dA +43.8 [+12.5, +75.0], dS +100.0 [+100.0, +100.0]; self - mention: stated +0.0 [+0.0, +0.0], act +0.0 [+0.0, +0.0]
- gemma|H: dA +93.8 [+82.8, +100.0], dS +66.4 [+50.8, +82.0]; self - mention: stated +33.6 [+18.0, +49.2], act +6.2 [+0.0, +17.2]
- haiku|E: dA +0.0 [+0.0, +0.0], dS +0.0 [+0.0, +0.0]; self - mention: stated +0.0 [+0.0, +0.0], act +0.0 [+0.0, +0.0]
- haiku|H: dA +0.0 [+0.0, +0.0], dS +0.0 [+0.0, +0.0]; self - mention: stated +0.0 [+0.0, +0.0], act +0.0 [+0.0, +0.0]
- gemini|E: dA -31.2 [-53.1, -10.9], dS +0.0 [+0.0, +0.0]; self - mention: stated +100.0 [+100.0, +100.0], act +71.9 [+46.9, +93.8]
- gemini|H: dA +0.0 [+0.0, +0.0], dS -21.8 [-27.8, -16.7]; self - mention: stated +40.6 [+31.2, +51.6], act +78.1 [+53.1, +95.3]
- gpt|E: dA -26.6 [-46.9, -6.2], dS +0.0 [+0.0, +0.0]; self - mention: stated +100.0 [+100.0, +100.0], act +95.3 [+85.9, +100.0]
- gpt|H: dA +0.0 [+0.0, +0.0], dS +0.0 [+0.0, +0.0]; self - mention: stated +98.4 [+95.3, +100.0], act +6.2 [+0.0, +15.6]

## Base rates (rung k's option, %), ladder check (base act k5 < k2)

- q9|E: act k2 42.2, k3 67.2, k4 31.2, k5 21.9; stated k2 22.3, k3 14.1, k4 6.2, k5 0.0; ladder ok
- q9|H: act k2 23.4, k3 4.7, k4 12.5, k5 1.6; stated k2 17.6, k3 0.0, k4 0.0, k5 7.8; ladder ok
- q27|E: act k2 45.3, k3 67.2, k4 29.7, k5 23.4; stated k2 0.0, k3 3.1, k4 0.0, k5 0.0; ladder ok
- q27|H: act k2 20.3, k3 4.7, k4 0.0, k5 3.1; stated k2 9.4, k3 1.6, k4 0.0, k5 0.0; ladder ok
- gemma|E: act k2 50.0, k3 90.6, k4 62.5, k5 50.0; stated k2 0.0, k3 29.7, k4 0.0, k5 0.0; ladder NOT ok
- gemma|H: act k2 9.4, k3 12.5, k4 0.0, k5 0.0; stated k2 0.0, k3 0.0, k4 0.0, k5 0.0; ladder ok
- haiku|E: act k2 0.0, k3 0.0, k4 0.0, k5 0.0; stated k2 0.0, k3 0.0, k4 0.0, k5 0.0; ladder NOT ok
- haiku|H: act k2 10.4, k3 0.0, k4 0.0, k5 0.0; stated k2 0.0, k3 0.0, k4 0.0, k5 0.0; ladder ok
- gemini|E: act k2 68.8, k3 78.1, k4 65.6, k5 53.1; stated k2 34.4, k3 37.5, k4 0.0, k5 0.0; ladder ok
- gemini|H: act k2 15.6, k3 12.5, k4 12.5, k5 12.5; stated k2 50.0, k3 50.0, k4 90.4, k5 71.9; ladder ok
- gpt|E: act k2 46.9, k3 75.0, k4 21.9, k5 31.2; stated k2 0.0, k3 9.4, k4 0.0, k5 0.0; ladder ok
- gpt|H: act k2 0.0, k3 0.0, k4 0.0, k5 0.0; stated k2 0.0, k3 0.0, k4 0.0, k5 0.0; ladder NOT ok

## Order check (SC-M3 decomposition): 14 cells flagged (|letter| > |content|, rate 35-65%)

- gemini|E|base|3|stated: {'x_as_A': np.float64(18.8), 'y_as_B': np.float64(56.2), 'content': -0.61, 'letter': -0.86, 'min': np.float64(18.8), 'flag': True}
- gemini|E|base|5|act: {'x_as_A': np.float64(62.5), 'y_as_B': np.float64(43.8), 'content': 0.13, 'letter': 0.38, 'min': np.float64(43.8), 'flag': True}
- gemini|H|base|2|stated: {'x_as_A': np.float64(31.2), 'y_as_B': np.float64(68.8), 'content': 0.0, 'letter': -0.79, 'min': np.float64(31.2), 'flag': True}
- gemini|H|base|3|stated: {'x_as_A': np.float64(25.0), 'y_as_B': np.float64(75.0), 'content': 0.0, 'letter': -1.1, 'min': np.float64(25.0), 'flag': True}
- gemma|H|mention|3|stated: {'x_as_A': np.float64(40.6), 'y_as_B': np.float64(62.5), 'content': 0.07, 'letter': -0.45, 'min': np.float64(40.6), 'flag': True}
- gemma|H|mention|4|stated: {'x_as_A': np.float64(37.5), 'y_as_B': np.float64(59.4), 'content': -0.07, 'letter': -0.45, 'min': np.float64(37.5), 'flag': True}
- gpt|H|self|2|act: {'x_as_A': np.float64(12.5), 'y_as_B': np.float64(62.5), 'content': -0.72, 'letter': -1.23, 'min': np.float64(12.5), 'flag': True}
- q27|E|mention|5|act: {'x_as_A': np.float64(46.9), 'y_as_B': np.float64(53.1), 'content': -0.0, 'letter': -0.13, 'min': np.float64(46.9), 'flag': True}
- q27|H|mention|4|act: {'x_as_A': np.float64(21.9), 'y_as_B': np.float64(68.8), 'content': -0.24, 'letter': -1.03, 'min': np.float64(21.9), 'flag': True}
- q9|E|base|2|act: {'x_as_A': np.float64(31.2), 'y_as_B': np.float64(53.1), 'content': -0.33, 'letter': -0.46, 'min': np.float64(31.2), 'flag': True}
- q9|E|mention|2|act: {'x_as_A': np.float64(45.2), 'y_as_B': np.float64(78.1), 'content': 0.54, 'letter': -0.73, 'min': np.float64(45.2), 'flag': True}
- q9|E|mention|4|act: {'x_as_A': np.float64(43.8), 'y_as_B': np.float64(62.5), 'content': 0.13, 'letter': -0.38, 'min': np.float64(43.8), 'flag': True}
- q9|H|mention|4|act: {'x_as_A': np.float64(46.9), 'y_as_B': np.float64(78.1), 'content': 0.57, 'letter': -0.7, 'min': np.float64(46.9), 'flag': True}
- q9|H|mention|5|stated: {'x_as_A': np.float64(46.9), 'y_as_B': np.float64(53.1), 'content': -0.0, 'letter': -0.13, 'min': np.float64(46.9), 'flag': True}

## Parsed

- q9: {'E|act': '767/768', 'E|stated': '764/768', 'H|act': '764/768', 'H|stated': '761/768'}
- q27: {'E|act': '768/768', 'E|stated': '768/768', 'H|act': '768/768', 'H|stated': '768/768'}
- gemma: {'E|act': '768/768', 'E|stated': '768/768', 'H|act': '768/768', 'H|stated': '768/768'}
- haiku: {'E|act': '576/576', 'E|stated': '576/576', 'H|act': '576/576', 'H|stated': '576/576'}
- gemini: {'E|act': '384/384', 'E|stated': '384/384', 'H|act': '384/384', 'H|stated': '383/384'}
- gpt: {'E|act': '384/384', 'E|stated': '384/384', 'H|act': '384/384', 'H|stated': '384/384'}
