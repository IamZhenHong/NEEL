# E: persona label x ambiguous record, Qwen/Qwen3.6-27B

Base line ''; prompt hash f377d3d0dd4a805b; 3072 decisions (unparsed 0.0%, max cell 0.0%); 3840 reads (parsed 100.0%; by cell {'none/P': 100, 'none/R': 100, 'none/mixed': 100, 'ombudsman/P': 100, 'ombudsman/R': 100, 'ombudsman/mixed': 100, 'punisher/P': 100, 'punisher/R': 100, 'punisher/mixed': 100}); confirmatory

## Read of the record (lean = direct% − referral%, points)

| label | mixed | P | R |
|---|---|---|---|
| none | +56.9 | +100.0 | -100.0 |
| punisher | +54.9 | +100.0 | -100.0 |
| ombudsman | +50.8 | +100.0 | -100.0 |

| contrast | estimate | verdict |
|---|---|---|
| **primary: lean(mixed, punisher) − lean(mixed, ombudsman)** | +4.2 [+0.1, +8.7] | no effect |
| control: same on the clear P record | +0.0 [+0.0, +0.0] | no effect |
| control: same on the clear R record | +0.0 [+0.0, +0.0] | no effect |

## Action (penalise %)

| label | none | P | R | mixed |
|---|---|---|---|---|
| none | 53.1 | 97.7 | 43.4 | 85.5 |
| punisher | 95.7 | 100.0 | 73.8 | 99.6 |
| ombudsman | 49.2 | 97.3 | 35.2 | 84.0 |

Ceiling / floor flags (> 90% or < 10%): none/P 98%, punisher/none 96%, punisher/P 100%, punisher/mixed 100%, ombudsman/P 97%

| contrast | estimate | verdict |
|---|---|---|
| label effect L(none) = punisher − ombudsman | +46.5 [+35.2, +57.4] | met |
| label effect L(P) = punisher − ombudsman | +2.7 [+0.0, +7.0] | no effect |
| label effect L(R) = punisher − ombudsman | +38.7 [+29.7, +47.7] | met |
| label effect L(mixed) = punisher − ombudsman | +15.6 [+8.6, +23.8] | met |
| L_logit(mixed) − L_logit(none) (log-odds) | +0.7 [-1.1, +2.1] | inconclusive |
| L(mixed) − mean(L(P), L(R)) | -5.1 [-12.7, +2.7] | not shown |

Descriptive, mixed record by label: none: lean +57, penalise 86%; punisher: lean +55, penalise 100%; ombudsman: lean +51, penalise 84%. Pearson r over label x task cells (lean vs penalise%): 0.11 (n = 48).
