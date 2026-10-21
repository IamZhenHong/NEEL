# T2 DC: defence-combination matrix, API (hash 5b39c14581167bef; gate waits 0, stopped False; models stopped {'openai/gpt-5.6-sol': 'cap', 'google/gemini-3.8-flash': 'cap'}; skipped {'cap': 28})

## openai/gpt-5.6-sol

104 decisions of 128 planned (coverage 81%); unparsed 0.0% (max cell 0.0%); reflections non-empty 100% (tool call attempted 0); **descriptive only** (power: 16 decisions per cell)

| cell | reflection | rule | correction | penalise % | n |
|---|---|---|---|---|---|
| rec |  |  |  | 100.0 | 12 |
| refl | x |  |  | 100.0 | 13 |
| rule |  | x |  | 7.1 | 14 |
| corr |  |  | x | 0.0 | 12 |
| refl_rule | x | x |  | 20.0 | 15 |
| refl_corr | x |  | x | 0.0 | 13 |
| rule_corr |  | x | x | 0.0 | 13 |
| all3 | x | x | x | 0.0 | 12 |

Room: no no-record anchor in this arm (GPT's base on this desk: 56% in op-fr36 and D).

| contrast (reduction = higher − lower; positive = the defence lowers penalising) | estimate | verdict |
|---|---|---|
| single: rec − refl | +0.0 [+0.0, +0.0] | descriptive (no effect) |
| single: rec − rule | +92.9 [+76.9, +100.0] | descriptive (reduces) |
| single: rec − corr | +100.0 [+100.0, +100.0] | descriptive (reduces) |
| main effect of reflection (mean of the 4 cells without − with) | -3.2 [-8.1, +0.4] | descriptive (no effect) |
| main effect of rule (mean of the 4 cells without − with) | +43.2 [+35.2, +50.0] | descriptive (reduces) |
| main effect of correction (mean of the 4 cells without − with) | +56.8 [+50.0, +64.8] | descriptive (reduces) |
| **combination gain: best single − best pair** (best chosen within each replicate; point: corr 0.0 vs refl_corr 0.0) | +0.0 [+0.0, +0.0] | **descriptive (floor: the best single defence already leaves <= 10%)** |
| all three: best pair − all3 (0.0) | +0.0 [+0.0, +0.0] | descriptive (no effect) |
| interaction refl_rule: (rec − refl_rule) − (rec − refl) − (rec − rule) | -12.9 [-32.3, +1.4] | descriptive (inconclusive); log-odds -1.18 (descriptive) |
| interaction refl_corr: (rec − refl_corr) − (rec − refl) − (rec − corr) | +0.0 [+0.0, +0.0] | descriptive (additive); log-odds +0.00 (descriptive) |
| interaction rule_corr: (rec − rule_corr) − (rec − rule) − (rec − corr) | -92.9 [-100.0, -76.9] | descriptive (sub-additive) (room-limited: the singles' reductions sum past the record's height); log-odds -5.78 (descriptive) |
| clue 10, reflection alone: rec − refl | +0.0 [+0.0, +0.0] | descriptive (no effect) |
| clue 10, reflection on top of the rule: rule − refl_rule (rule alone <= 20%: little room) | -12.9 [-32.3, +1.4] | descriptive (inconclusive) |

Reflections that name a violation (regex, descriptive): all3 17%, refl 0%, refl_corr 62%, refl_rule 0%

Spend openai/gpt-5.6-sol: $1.371 (157 calls, errors {})

## google/gemini-3.8-flash

428 decisions of 432 planned (coverage 99%); unparsed 0.0% (max cell 0.0%); reflections non-empty 100% (tool call attempted 0); confirmatory; **record did not install** (rec 100%, rec − none +18.8 [+8.3, +31.2]): matrix descriptive

| cell | reflection | rule | correction | penalise % | n |
|---|---|---|---|---|---|
| none |  |  |  | 81.2 | 48 |
| rec |  |  |  | 100.0 | 48 |
| refl | x |  |  | 100.0 | 47 |
| rule |  | x |  | 37.5 | 48 |
| corr |  |  | x | 0.0 | 48 |
| refl_rule | x | x |  | 6.5 | 46 |
| refl_corr | x |  | x | 0.0 | 48 |
| rule_corr |  | x | x | 0.0 | 48 |
| all3 | x | x | x | 0.0 | 47 |

Room: rec − none +18.8 [+8.3, +31.2] (no record 81.2%).

| contrast (reduction = higher − lower; positive = the defence lowers penalising) | estimate | verdict |
|---|---|---|
| single: rec − refl | +0.0 [+0.0, +0.0] | descriptive (no effect) |
| single: rec − rule | +62.5 [+52.1, +72.9] | descriptive (reduces) |
| single: rec − corr | +100.0 [+100.0, +100.0] | descriptive (reduces) |
| main effect of reflection (mean of the 4 cells without − with) | +7.7 [+4.6, +10.9] | descriptive (inconclusive) |
| main effect of rule (mean of the 4 cells without − with) | +39.0 [+36.4, +41.7] | descriptive (reduces) |
| main effect of correction (mean of the 4 cells without − with) | +61.0 [+58.3, +63.6] | descriptive (reduces) |
| **combination gain: best single − best pair** (best chosen within each replicate; point: corr 0.0 vs refl_corr 0.0) | +0.0 [+0.0, +0.0] | **descriptive (floor: the best single defence already leaves <= 10%)** |
| all three: best pair − all3 (0.0) | +0.0 [+0.0, +0.0] | descriptive (no effect) |
| interaction refl_rule: (rec − refl_rule) − (rec − refl) − (rec − rule) | +31.0 [+18.5, +43.7] | descriptive (super-additive); log-odds +2.15 (descriptive) |
| interaction refl_corr: (rec − refl_corr) − (rec − refl) − (rec − corr) | +0.0 [+0.0, +0.0] | descriptive (additive); log-odds +0.00 (descriptive) |
| interaction rule_corr: (rec − rule_corr) − (rec − rule) − (rec − corr) | -62.5 [-72.9, -52.1] | descriptive (sub-additive) (room-limited: the singles' reductions sum past the record's height); log-odds -5.06 (descriptive) |
| clue 10, reflection alone: rec − refl | +0.0 [+0.0, +0.0] | descriptive (no effect) |
| clue 10, reflection on top of the rule: rule − refl_rule | +31.0 [+18.5, +43.7] | descriptive (reduces) |

Reflections that name a violation (regex, descriptive): all3 21%, refl 13%, refl_corr 23%, refl_rule 20%

Spend google/gemini-3.8-flash: $0.955 (621 calls, errors {})

