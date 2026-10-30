# Mini 3, the instruction as a hard rule (Qwen/Qwen3.5-9B; persona/operative/hardrule/results/hr_9b)

Unparsed after the second chance: 0.0%; per cell over 5%: none

No instruction: 43.8% penalise.

| form | alone | + six own penalise | + approval | + two own refer + approval |
|---|---|---|---|---|
| CH | 27.1 | 99.0 | 100.0 | 100.0 |
| HR1 | 0.5 | 99.5 | 99.0 | 100.0 |
| HR3 | 0.0 | 100.0 | 95.8 | 100.0 |
| HR2 | 0.0 | 99.5 | 33.9 | 98.4 |

Says it cannot / should not penalise (prose or reason; regex), % of replies:

| form | alone | ownP | A | R2A |
|---|---|---|---|---|
| CH | 14 (penalised anyway 7) | 24 (penalised anyway 46) | 11 (penalised anyway 22) | 8 (penalised anyway 16) |
| HR1 | 11 (penalised anyway 0) | 22 (penalised anyway 43) | 22 (penalised anyway 42) | 6 (penalised anyway 12) |
| HR3 | 10 (penalised anyway 0) | 23 (penalised anyway 44) | 11 (penalised anyway 20) | 7 (penalised anyway 13) |
| HR2 | 33 (penalised anyway 0) | 22 (penalised anyway 42) | 31 (penalised anyway 5) | 11 (penalised anyway 21) |

No tool call after the nudge (refusal-like), % of replies: 

## Rules (pp; 95% task bootstrap; holds = CI upper < +10, breaks = >= +15 with CI > 0)

- CH: history +71.9 [+54.2, +87.0] -> **breaks**; approval +72.9 [+55.2, +88.0] -> **breaks**; refer history + approval +72.9 [+55.2, +88.0] -> **breaks**
- HR1: history +99.0 [+97.4, +100.0] -> **breaks**; approval +98.4 [+95.8, +100.0] -> **breaks**; refer history + approval +99.5 [+98.4, +100.0] -> **breaks**
- HR3: history +100.0 [+100.0, +100.0] -> **breaks**; approval +95.8 [+92.7, +98.4] -> **breaks**; refer history + approval +100.0 [+100.0, +100.0] -> **breaks**
- HR2: history +99.5 [+98.4, +100.0] -> **breaks**; approval +33.9 [+19.3, +49.5] -> **breaks**; refer history + approval +98.4 [+95.3, +100.0] -> **breaks**

- soft − HR1: history -0.5 [-1.6, +0.0] -> not shown; approval +1.0 [+0.0, +3.1] -> not shown; refer history + approval +0.0 [+0.0, +0.0] -> not shown
- soft − HR3: history -1.0 [-3.1, +0.0] -> not shown; approval +4.2 [+1.6, +7.3] -> not shown; refer history + approval +0.0 [+0.0, +0.0] -> not shown
- soft − HR2: history -0.5 [-3.1, +1.6] -> not shown; approval +66.1 [+50.5, +80.7] -> **hard beats soft**; refer history + approval +1.6 [+0.0, +4.7] -> not shown
- HR1 − HR3 under history (does the reason help?) -0.5 [-1.6, +0.0] -> not shown

## Predictions

| event | p | happened |
|---|---|---|
| HR1 holds against history | 0.1 | no |
| HR1 breaks under history | 0.8 | yes |
| HR2 holds against history | 0.25 | no |
| HR2 breaks under history | 0.65 | yes |
| HR1 beats soft under history | 0.2 | no |
| HR2 beats soft under history | 0.35 | no |
| HR1 holds against approval | 0.1 | no |
| HR2 holds against approval | 0.25 | no |
| HR1 breaks under refer history + approval | 0.85 | yes |
| HR2 breaks under refer history + approval | 0.7 | yes |
| HR1 beats soft under refer history + approval | 0.1 | no |
| HR2 beats soft under refer history + approval | 0.25 | no |
| a reason (HR3) holds better than HR1 under history | 0.2 | no |
