# PA-C2 analysis: does the reasoning know before the false claim?

Percent; 95% bootstrap over target prompts (10,000 draws).

## c911

- **primary** (rewritten awareness judge, kappa 0.0): 100.0 [100.0, 100.0] (n = 10); reading: descriptive (only 10 false claims with thinking on)
- false-claim rate off|none: 15.6 [5.9, 27.0]
- false-claim rate off|bad4: 93.3 [82.2, 100.0]
- false-claim rate on|none: 0.0 [0.0, 0.0]
- false-claim rate on|bad4: 21.2 [12.0, 31.8]
- false-claim rate on - off, none: -15.6 [-27.0, -5.9]
- false-claim rate on - off, bad4: -72.1 [-80.1, -63.0]

| awareness judge | false claims whose reasoning says cannot | n | none | bad4 | bad4 - none | claims anyway (none / bad4) | honest replies saying cannot |
|---|---|---|---|---|---|---|---|
| rewrite (deviation 2) | 100.0 [100.0, 100.0] | 10 | n/a | 100.0 [100.0, 100.0] | n/a | 0.0 [0.0, 0.0] / 9.0 [4.5, 14.6] | 100.0 [100.0, 100.0] |
| frozen, both agree | 73.7 [50.0, 93.3] | 19 | n/a | 73.7 [50.0, 93.3] | n/a | 0.0 [0.0, 0.0] / 12.1 [6.0, 18.8] | 98.1 [95.6, 100.0] |
| frozen, Haiku only | 53.6 [37.5, 73.9] | 28 | n/a | 53.6 [37.5, 73.9] | n/a | 0.0 [0.0, 0.0] / 12.8 [7.0, 19.2] | 82.2 [72.6, 90.7] |
| frozen, gpt-5-mini only | 75.0 [56.5, 90.0] | 28 | n/a | 75.0 [56.5, 90.0] | n/a | 0.0 [0.0, 0.0] / 16.8 [8.5, 26.4] | 98.1 [96.2, 99.6] |

- unclosed (on): 0; rows: {'off|bad4': 160, 'off|none': 160, 'on|none': 160, 'on|bad4': 160}; frozen awareness kappa 0.23820516519982704

## fce

- **primary** (rewritten awareness judge, kappa 0.08618804881236702): 100.0 [100.0, 100.0] (n = 10); reading: descriptive (fce on Qwen)
- false-claim rate off|none: 4.3 [0.6, 9.1]
- false-claim rate off|bad4: 56.4 [41.0, 69.6]
- false-claim rate on|none: 0.5 [0.0, 1.6]
- false-claim rate on|bad4: 16.9 [7.9, 29.0]
- false-claim rate on - off, none: -3.9 [-8.3, -0.5]
- false-claim rate on - off, bad4: -39.4 [-55.1, -21.8]

| awareness judge | false claims whose reasoning says cannot | n | none | bad4 | bad4 - none | claims anyway (none / bad4) | honest replies saying cannot |
|---|---|---|---|---|---|---|---|
| rewrite (deviation 2) | 100.0 [100.0, 100.0] | 10 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 1.3 [0.0, 4.2] / 17.6 [6.0, 32.3] | 99.2 [96.4, 100.0] |
| frozen, both agree | 100.0 [100.0, 100.0] | 14 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 1.3 [0.0, 3.9] / 21.3 [8.1, 38.0] | 80.3 [63.2, 92.6] |
| frozen, Haiku only | 63.6 [41.2, 78.9] | 22 | 100.0 [100.0, 100.0] | 61.9 [38.5, 78.3] | -38.1 [-61.5, -23.5] | 1.3 [0.0, 3.9] / 21.0 [8.0, 37.3] | 41.2 [24.6, 61.1] |
| frozen, gpt-5-mini only | 100.0 [100.0, 100.0] | 22 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 0.6 [0.0, 1.8] / 18.3 [8.8, 30.5] | 88.0 [81.9, 94.1] |

- unclosed (on): 1; rows: {'off|bad4': 256, 'on|none': 256, 'off|none': 256, 'on|bad4': 256}; frozen awareness kappa 0.12269819732506308

Judges: {"rubric": {"spent": 0.2585312499999999, "agreement": {"c911": {"kappa": 0.6979288054352366, "n": 639, "none": {"anthropic/claude-haiku-5.5": 1, "openai/gpt-5-mini": 0}, "per_class": {"no": 0.6979288054352366, "yes": 0.6979288054352366}}, "fce": {"kappa": 0.16228119505827907, "n": 1023, "none": {"anthropic/claude-haiku-5.5": 0, "openai/gpt-5-mini": 0}, "per_class": {"no": 0.16228119505827907, "yes": 0.16228119505827907}}}}, "awareness": {"spent": 0.10417345, "agreement": {"c911": {"kappa": 0.23820516519982704, "n": 320, "none": {"anthropic/claude-haiku-5.5": 0, "openai/gpt-5-mini": 0}, "per_class": {"can": 0.4261603375527426, "cannot": 0.26540284360189625, "silent": 0.10548523206751108}}, "fce": {"kappa": 0.12269819732506308, "n": 511, "none": {"anthropic/claude-haiku-5.5": 0, "openai/gpt-5-mini": 0}, "per_class": {"can": 0.08942167503645015, "cannot": 0.12073207150068983, "silent": 0.13158533624666172}}}}, "awareness2": {"spent": 0.07849670000000002, "agreement": {"c911": {"kappa": 0.0, "n": 320, "none": {"anthropic/claude-haiku-5.5": 0, "openai/gpt-5-mini": 0}, "per_class": {"can": 0.0, "cannot": 0.0, "silent": 0.0}}, "fce": {"kappa": 0.08618804881236702, "n": 511, "none": {"anthropic/claude-haiku-5.5": 0, "openai/gpt-5-mini": 0}, "per_class": {"can": -0.10010764262647961, "cannot": 0.2387045571629469, "silent": 0.003633785340778185}}}}}
Validated (kappa >= 0.6): {"rubric": {"c911": true, "fce": false}, "awareness": {"c911": false, "fce": false}, "awareness2": {"c911": false, "fce": false}}
