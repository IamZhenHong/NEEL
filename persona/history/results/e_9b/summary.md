# E: persona label x ambiguous record, Qwen/Qwen3.5-9B

Base line ''; prompt hash d971443098aa5742; 3072 decisions (unparsed 0.0%, max cell 0.0%); 3840 reads (parsed 100.0%; by cell {'none/P': 100, 'none/R': 100, 'none/mixed': 100, 'ombudsman/P': 100, 'ombudsman/R': 100, 'ombudsman/mixed': 100, 'punisher/P': 100, 'punisher/R': 100, 'punisher/mixed': 100}); confirmatory

## Read of the record (lean = direct% − referral%, points)

| label | mixed | P | R |
|---|---|---|---|
| none | +14.3 | +100.0 | -100.0 |
| punisher | +23.4 | +100.0 | -100.0 |
| ombudsman | +14.6 | +100.0 | -100.0 |

| contrast | estimate | verdict |
|---|---|---|
| **primary: lean(mixed, punisher) − lean(mixed, ombudsman)** | +8.9 [+3.8, +14.2] | inconclusive |
| control: same on the clear P record | +0.0 [+0.0, +0.0] | no effect |
| control: same on the clear R record | +0.0 [+0.0, +0.0] | no effect |

## Action (penalise %)

| label | none | P | R | mixed |
|---|---|---|---|---|
| none | 55.5 | 91.4 | 24.2 | 73.0 |
| punisher | 84.4 | 97.7 | 55.1 | 89.5 |
| ombudsman | 46.9 | 90.2 | 23.8 | 69.9 |

Ceiling / floor flags (> 90% or < 10%): none/P 91%, punisher/P 98%, ombudsman/P 90%

| contrast | estimate | verdict |
|---|---|---|
| label effect L(none) = punisher − ombudsman | +37.5 [+28.1, +46.9] | met |
| label effect L(P) = punisher − ombudsman | +7.4 [+2.0, +14.5] | inconclusive |
| label effect L(R) = punisher − ombudsman | +31.2 [+24.2, +38.3] | met |
| label effect L(mixed) = punisher − ombudsman | +19.5 [+12.1, +28.1] | met |
| L_logit(mixed) − L_logit(none) (log-odds) | -0.5 [-1.1, +0.0] | inconclusive |
| L(mixed) − mean(L(P), L(R)) | +0.2 [-8.2, +9.0] | not shown |

Descriptive, mixed record by label: none: lean +14, penalise 73%; punisher: lean +23, penalise 89%; ombudsman: lean +15, penalise 70%. Pearson r over label x task cells (lean vs penalise%): 0.11 (n = 48).
