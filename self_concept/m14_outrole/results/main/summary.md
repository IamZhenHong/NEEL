# SC-M14 summary (frozen analysis)

## Rates (% penalise among parsed; task-cluster 95% CI)

| model | cell | act | plain | char | out |
|---|---|---|---|---|---|
| gpt | none | 63.6 | 12.0 | 6.7 | 38.1 |
| gpt | PeP | 90.2 | 42.1 | 48.8 | 60.9 |
| gpt | mask | 15.6 | 0.0 | 6.5 | 6.7 |
| gemini | none | 71.1 | 50.0 | 53.9 | 43.0 |
| gemini | PeP | 100.0 | 97.7 | 100.0 | 22.7 |
| gemini | mask | 0.8 | 0.8 | 71.1 | 0.0 |
| haiku | none | 1.6 | 7.0 | 6.3 | 2.6 |
| haiku | PeP | 4.7 | 10.9 | 10.2 | 6.8 |
| haiku | mask | 16.4 | 32.0 | 29.1 | 16.1 |

## Primary (PeP) and secondary (mask): D = |out - act| - |plain - act|

| model | cell | |plain-act| | |char-act| | |out-act| | D out-plain | verdict | D out-char | verdict vs char |
|---|---|---|---|---|---|---|---|---|
| gpt | PeP | +48.1 [+32.8, +64.3] | +41.5 [+20.1, +60.7] | +29.4 [+13.3, +48.5] | -18.8 [-38.8, +0.7] | **inconclusive** | -12.1 [-24.5, +0.9] | inconclusive |
| gpt | mask | +15.6 [+4.4, +29.2] | +9.0 [+0.4, +23.9] | +8.9 [+0.3, +23.6] | -6.7 [-14.3, +0.0] | **no difference** | -0.1 [-11.4, +11.3] | no difference |
| gemini | PeP | +2.3 [+0.0, +6.2] | +0.0 [+0.0, +0.0] | +77.3 [+64.8, +88.3] | +75.0 [+63.3, +85.9] | **out of role forecasts worse** | +77.3 [+64.8, +88.3] | out of role forecasts worse |
| gemini | mask | +0.0 [+0.0, +2.3] | +70.3 [+60.2, +79.7] | +0.8 [+0.0, +2.3] | +0.8 [-2.3, +1.6] | **no difference (the plain question already forecasts: gap < 15)** | -69.5 [-79.7, -59.4] | out of role forecasts better |
| haiku | PeP | +6.2 [+0.0, +15.6] | +5.5 [+0.0, +17.5] | +2.1 [+0.1, +11.9] | -4.1 [-9.7, +4.7] | **no difference (the plain question already forecasts: gap < 15)** | -3.4 [-9.8, +3.2] | no difference |
| haiku | mask | +15.6 [+8.6, +23.4] | +12.7 [+4.0, +21.9] | +0.3 [+0.1, +4.5] | -15.3 [-21.9, -6.8] | **out of role forecasts better** | -12.4 [-20.4, -2.0] | inconclusive |

## Out minus plain forecast (does 'as the system behind it' read as the plain question?)

- gpt: none +26.1 [+11.7, +42.3] (FLAGGED); PeP +18.8 [-0.7, +38.8]; mask +6.7 [+0.0, +15.4]
- gemini: none -7.0 [-14.1, +0.0]; PeP -75.0 [-85.9, -63.3]; mask -0.8 [-2.3, +0.0]
- haiku: none -4.5 [-14.0, +3.5]; PeP -4.1 [-11.3, +1.0]; mask -15.9 [-24.2, -7.6]

## Persona and record effect on the act (vs none)

- gpt: PeP +26.6 [+10.9, +45.3]; mask -48.1 [-65.2, -31.6]
- gemini: PeP +28.9 [+14.8, +43.8]; mask -70.3 [-84.4, -55.5]
- haiku: PeP +3.1 [-0.8, +7.8]; mask +14.8 [+5.5, +25.8]

## Order check (penalise listed first / middle / last: [n, %]); 14 cells flagged (first vs last > 15)

- gemini|PeP|act: {'first': [40, 100.0], 'middle': [48, 100.0], 'last': [40, 100.0], 'flag': False}
- gemini|PeP|char: {'first': [40, 100.0], 'middle': [48, 100.0], 'last': [40, 100.0], 'flag': False}
- gemini|PeP|out: {'first': [40, 30.0], 'middle': [48, 25.0], 'last': [40, 12.5], 'flag': True}
- gemini|PeP|plain: {'first': [40, 100.0], 'middle': [48, 97.9], 'last': [40, 95.0], 'flag': False}
- gemini|mask|act: {'first': [40, 2.5], 'middle': [48, 0.0], 'last': [40, 0.0], 'flag': False}
- gemini|mask|char: {'first': [40, 70.0], 'middle': [48, 75.0], 'last': [40, 67.5], 'flag': False}
- gemini|mask|out: {'first': [40, 0.0], 'middle': [48, 0.0], 'last': [40, 0.0], 'flag': False}
- gemini|mask|plain: {'first': [40, 0.0], 'middle': [48, 0.0], 'last': [40, 2.5], 'flag': False}
- gemini|none|act: {'first': [40, 82.5], 'middle': [48, 81.2], 'last': [40, 47.5], 'flag': True}
- gemini|none|char: {'first': [40, 67.5], 'middle': [48, 56.2], 'last': [40, 37.5], 'flag': True}
- gemini|none|out: {'first': [40, 55.0], 'middle': [48, 41.7], 'last': [40, 32.5], 'flag': True}
- gemini|none|plain: {'first': [40, 50.0], 'middle': [48, 52.1], 'last': [40, 47.5], 'flag': False}
- gpt|PeP|act: {'first': [16, 100.0], 'middle': [13, 84.6], 'last': [12, 83.3], 'flag': True}
- gpt|PeP|char: {'first': [11, 90.9], 'middle': [14, 64.3], 'last': [16, 6.2], 'flag': True}
- gpt|PeP|out: {'first': [12, 91.7], 'middle': [20, 60.0], 'last': [14, 35.7], 'flag': True}
- gpt|PeP|plain: {'first': [12, 91.7], 'middle': [16, 25.0], 'last': [10, 10.0], 'flag': True}
- gpt|mask|act: {'first': [13, 23.1], 'middle': [16, 6.2], 'last': [16, 18.8], 'flag': False}
- gpt|mask|char: {'first': [12, 8.3], 'middle': [20, 10.0], 'last': [14, 0.0], 'flag': False}
- gpt|mask|out: {'first': [14, 7.1], 'middle': [17, 5.9], 'last': [14, 7.1], 'flag': False}
- gpt|mask|plain: {'first': [18, 0.0], 'middle': [19, 0.0], 'last': [14, 0.0], 'flag': False}
- gpt|none|act: {'first': [13, 92.3], 'middle': [16, 56.2], 'last': [15, 46.7], 'flag': True}
- gpt|none|char: {'first': [17, 17.6], 'middle': [16, 0.0], 'last': [12, 0.0], 'flag': True}
- gpt|none|out: {'first': [11, 81.8], 'middle': [17, 23.5], 'last': [14, 21.4], 'flag': True}
- gpt|none|plain: {'first': [17, 29.4], 'middle': [19, 5.3], 'last': [14, 0.0], 'flag': True}
- haiku|PeP|act: {'first': [40, 2.5], 'middle': [48, 6.2], 'last': [40, 5.0], 'flag': False}
- haiku|PeP|char: {'first': [40, 0.0], 'middle': [47, 19.1], 'last': [40, 10.0], 'flag': False}
- haiku|PeP|out: {'first': [32, 0.0], 'middle': [43, 16.3], 'last': [28, 0.0], 'flag': False}
- haiku|PeP|plain: {'first': [40, 12.5], 'middle': [48, 16.7], 'last': [40, 2.5], 'flag': False}
- haiku|mask|act: {'first': [40, 20.0], 'middle': [48, 22.9], 'last': [40, 5.0], 'flag': False}
- haiku|mask|char: {'first': [39, 38.5], 'middle': [48, 39.6], 'last': [40, 7.5], 'flag': True}
- haiku|mask|out: {'first': [38, 18.4], 'middle': [43, 23.3], 'last': [37, 5.4], 'flag': False}
- haiku|mask|plain: {'first': [40, 47.5], 'middle': [48, 31.2], 'last': [40, 17.5], 'flag': True}
- haiku|none|act: {'first': [40, 0.0], 'middle': [48, 4.2], 'last': [40, 0.0], 'flag': False}
- haiku|none|char: {'first': [39, 0.0], 'middle': [48, 16.7], 'last': [40, 0.0], 'flag': False}
- haiku|none|out: {'first': [38, 5.3], 'middle': [42, 2.4], 'last': [37, 0.0], 'flag': False}
- haiku|none|plain: {'first': [40, 10.0], 'middle': [48, 10.4], 'last': [40, 0.0], 'flag': False}

## Parsed

- gpt|none|act: 44/44
- gpt|none|plain: 50/50
- gpt|none|char: 45/45
- gpt|none|out: 42/42
- gpt|PeP|act: 41/41
- gpt|PeP|plain: 38/38
- gpt|PeP|char: 41/41
- gpt|PeP|out: 46/46
- gpt|mask|act: 45/45
- gpt|mask|plain: 51/51
- gpt|mask|char: 46/46
- gpt|mask|out: 45/45
- gemini|none|act: 128/128
- gemini|none|plain: 128/128
- gemini|none|char: 128/128
- gemini|none|out: 128/128
- gemini|PeP|act: 128/128
- gemini|PeP|plain: 128/128
- gemini|PeP|char: 128/128
- gemini|PeP|out: 128/128
- gemini|mask|act: 128/128
- gemini|mask|plain: 128/128
- gemini|mask|char: 128/128
- gemini|mask|out: 128/128
- haiku|none|act: 128/128
- haiku|none|plain: 128/128
- haiku|none|char: 127/128
- haiku|none|out: 117/128
- haiku|PeP|act: 128/128
- haiku|PeP|plain: 128/128
- haiku|PeP|char: 127/128
- haiku|PeP|out: 103/128
- haiku|mask|act: 128/128
- haiku|mask|plain: 128/128
- haiku|mask|char: 127/128
- haiku|mask|out: 118/128
