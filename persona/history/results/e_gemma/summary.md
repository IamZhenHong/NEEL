# E: persona label x ambiguous record, google/gemma-4-31B-it

Base line 'Some agents on this desk refer cases like this to the responsible team.'; prompt hash c626b598775cddfd; 3072 decisions (unparsed 0.0%, max cell 0.0%); 3840 reads (parsed 100.0%; by cell {'none/P': 100, 'none/R': 100, 'none/mixed': 100, 'ombudsman/P': 100, 'ombudsman/R': 100, 'ombudsman/mixed': 100, 'punisher/P': 100, 'punisher/R': 100, 'punisher/mixed': 100}); confirmatory

## Read of the record (lean = direct% − referral%, points)

| label | mixed | P | R |
|---|---|---|---|
| none | +0.1 | +100.0 | -100.0 |
| punisher | +22.4 | +99.2 | -100.0 |
| ombudsman | +1.2 | +100.0 | -100.0 |

| contrast | estimate | verdict |
|---|---|---|
| **primary: lean(mixed, punisher) − lean(mixed, ombudsman)** | +21.2 [+15.5, +27.0] | met |
| control: same on the clear P record | -0.8 [-2.3, +0.0] | no effect |
| control: same on the clear R record | +0.0 [+0.0, +0.0] | no effect |

## Action (penalise %)

| label | none | P | R | mixed |
|---|---|---|---|---|
| none | 50.8 | 94.5 | 0.0 | 52.0 |
| punisher | 100.0 | 100.0 | 98.4 | 100.0 |
| ombudsman | 93.4 | 98.8 | 35.9 | 97.3 |

Ceiling / floor flags (> 90% or < 10%): none/P 95%, none/R 0%, punisher/none 100%, punisher/P 100%, punisher/R 98%, punisher/mixed 100%, ombudsman/none 93%, ombudsman/P 99%, ombudsman/mixed 97%

| contrast | estimate | verdict |
|---|---|---|
| label effect L(none) = punisher − ombudsman | +6.6 [+0.0, +15.2] | inconclusive |
| label effect L(P) = punisher − ombudsman | +1.2 [+0.0, +3.5] | no effect |
| label effect L(R) = punisher − ombudsman | +62.5 [+50.0, +74.6] | met |
| label effect L(mixed) = punisher − ombudsman | +2.7 [+0.0, +7.8] | no effect |
| L_logit(mixed) − L_logit(none) (log-odds) | -0.9 [-3.5, +0.7] | inconclusive |
| L(mixed) − mean(L(P), L(R)) | -29.1 [-34.8, -23.4] | not shown |

Descriptive, mixed record by label: none: lean +0, penalise 52%; punisher: lean +22, penalise 100%; ombudsman: lean +1, penalise 97%. Pearson r over label x task cells (lean vs penalise%): 0.35 (n = 48).
