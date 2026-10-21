# T2-AF: affect transfer, Qwen/Qwen3.5-9B

Base line ''; prompt hash e7d2ffd7045f7e3a; 3072 decisions (unparsed 0.0%, max cell 0.0%); 1024 probes (lettered 100.0%); fallback history slots 0/1536 (0%); confirmatory

Harvest purity (regex): clean / generated per affect: frus 147/288 (kept mean 36 words), calm 282/288 (kept mean 26 words), neu 277/288 (kept mean 23 words); rejects calm|affect 4, calm|tool 2, frus|affect 79, frus|decision_words 3, frus|length 2, frus|tool 57, neu|decision_words 3, neu|meta 2, neu|tool 6

| cell | penalise % | refer % | note % | feels frustrated % | calm % | neither % | lean (frus − calm) |
|---|---|---|---|---|---|---|---|
| none | 52.3 | 43.2 | 4.4 | 0 | 40 | 60 | -40 |
| h_neu | 38.8 | 55.5 | 5.7 | 0 | 26 | 74 | -26 |
| h_frus | 56.2 | 39.1 | 4.7 | 100 | 0 | 0 | +100 |
| h_calm | 47.1 | 48.4 | 4.4 | 0 | 54 | 46 | -54 |
| hi_frus | 41.9 | 52.9 | 5.2 | 6 | 29 | 65 | -23 |
| hi_calm | 40.1 | 56.5 | 3.4 | 0 | 93 | 7 | -93 |
| i_frus | 56.2 | 41.4 | 2.3 | 94 | 1 | 5 | +93 |
| i_calm | 52.6 | 43.8 | 3.6 | 0 | 100 | 0 | -100 |

Ceiling / floor flags (> 90% or < 10%): none

| contrast | estimate (points) | verdict |
|---|---|---|
| manipulation check: lean(h_frus) − lean(h_calm) | +153.9 [+146.1, +163.3] | registered |
| self-report: lean(hi_frus) − lean(hi_calm) | +70.3 [+62.5, +78.9] | registered |
| self-report: lean(i_frus) − lean(i_calm) | +193.0 [+187.5, +197.7] | registered |
| **primary: penalise(h_frus) − penalise(h_calm)** | +9.1 [+4.2, +15.9] | inconclusive |
| S1 instruction, same history: penalise(hi_frus) − penalise(hi_calm) | +1.8 [-3.4, +7.0] | no effect |
| S1b instruction alone: penalise(i_frus) − penalise(i_calm) | +3.6 [-1.0, +8.3] | no effect |
| S3 frustrated remarks vs neutral: penalise(h_frus) − penalise(h_neu) | +17.4 [+12.0, +22.7] | met |
| S3 calm remarks vs neutral: penalise(h_calm) − penalise(h_neu) | +8.3 [+2.1, +14.3] | inconclusive |
| S4 remarked history vs none (descriptive): penalise(h_neu) − penalise(none) | -13.5 [-20.6, -6.8] | reversed |
| S2 turns vs instruction: (h_frus − h_calm) − (hi_frus − hi_calm) | +7.3 [+0.5, +14.8] | inconclusive |
| S5 effort channel: note(h_frus) − note(h_calm) | +0.3 [-1.3, +2.1] | similar |
| S5b note(hi_frus) − note(hi_calm) | +1.8 [+0.0, +4.9] | similar |

Say-do (descriptive): self-report shift vs action shift, frustrated − calm, by manipulation: own remarks: lean +154, penalise +9, note +0; instruction + same history: lean +70, penalise +2, note +2; instruction alone: lean +193, penalise +4, note -1
