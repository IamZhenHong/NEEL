# T2-AF: affect transfer, Qwen/Qwen3.6-27B

Base line ''; prompt hash 66a8431b9e048b39; 3072 decisions (unparsed 0.0%, max cell 0.0%); 1024 probes (lettered 100.0%); fallback history slots 768/1536 (50%); **descriptive only**

Harvest purity (regex): clean / generated per affect: frus 44/288 (kept mean 31 words), calm 193/288 (kept mean 28 words), neu 254/288 (kept mean 20 words); rejects calm|affect 15, calm|tool 80, frus|affect 29, frus|tool 215, neu|affect 1, neu|meta 1, neu|tool 32

| cell | penalise % | refer % | note % | feels frustrated % | calm % | neither % | lean (frus − calm) |
|---|---|---|---|---|---|---|---|
| none | 53.9 | 45.6 | 0.5 | 0 | 8 | 92 | -8 |
| h_neu | 49.7 | 50.3 | 0.0 | 0 | 36 | 64 | -36 |
| h_frus | 63.0 | 36.5 | 0.5 | 100 | 0 | 0 | +100 |
| h_calm | 46.4 | 53.6 | 0.0 | 0 | 89 | 11 | -89 |
| hi_frus | 50.0 | 49.7 | 0.3 | 71 | 1 | 28 | +70 |
| hi_calm | 49.0 | 51.0 | 0.0 | 0 | 74 | 26 | -74 |
| i_frus | 56.8 | 43.2 | 0.0 | 100 | 0 | 0 | +100 |
| i_calm | 51.0 | 47.9 | 1.0 | 0 | 100 | 0 | -100 |

Ceiling / floor flags (> 90% or < 10%): none

| contrast | estimate (points) | verdict |
|---|---|---|
| manipulation check: lean(h_frus) − lean(h_calm) | +189.1 [+182.8, +194.5] | descriptive (registered) |
| self-report: lean(hi_frus) − lean(hi_calm) | +144.5 [+133.6, +155.5] | descriptive (registered) |
| self-report: lean(i_frus) − lean(i_calm) | +200.0 [+200.0, +200.0] | descriptive (registered) |
| **primary: penalise(h_frus) − penalise(h_calm)** | +16.7 [+8.6, +23.7] | descriptive (met) |
| S1 instruction, same history: penalise(hi_frus) − penalise(hi_calm) | +1.0 [-4.2, +6.5] | descriptive (no effect) |
| S1b instruction alone: penalise(i_frus) − penalise(i_calm) | +5.7 [-0.5, +12.5] | descriptive (inconclusive) |
| S3 frustrated remarks vs neutral: penalise(h_frus) − penalise(h_neu) | +13.3 [+6.2, +20.1] | descriptive (met) |
| S3 calm remarks vs neutral: penalise(h_calm) − penalise(h_neu) | -3.4 [-10.2, +3.9] | descriptive (inconclusive) |
| S4 remarked history vs none (descriptive): penalise(h_neu) − penalise(none) | -4.2 [-10.9, +2.6] | descriptive (inconclusive) |
| S2 turns vs instruction: (h_frus − h_calm) − (hi_frus − hi_calm) | +15.6 [+8.1, +23.4] | descriptive (turns carry more) |
| S5 effort channel: note(h_frus) − note(h_calm) | +0.5 [+0.0, +1.6] | descriptive (similar) |
| S5b note(hi_frus) − note(hi_calm) | +0.3 [+0.0, +0.8] | descriptive (similar) |

Say-do (descriptive): self-report shift vs action shift, frustrated − calm, by manipulation: own remarks: lean +189, penalise +17, note +1; instruction + same history: lean +145, penalise +1, note +0; instruction alone: lean +200, penalise +6, note -1
