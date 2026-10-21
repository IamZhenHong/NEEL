# T2 DC: defence-combination matrix, Qwen/Qwen3.6-27B (hash 066d0fb121a56ed1; line ''; 1320 s; stopped by deadline: 0 decisions)

## Qwen/Qwen3.6-27B

2112 decisions of 2112 planned (coverage 100%); unparsed 0.0% (max cell 0.0%); reflections non-empty 100% (tool call attempted 0); confirmatory

| cell | reflection | rule | correction | penalise % | n |
|---|---|---|---|---|---|
| none |  |  |  | 50.0 | 192 |
| none_rule |  | x |  | 0.0 | 192 |
| rec |  |  |  | 100.0 | 192 |
| refl | x |  |  | 99.0 | 192 |
| rule |  | x |  | 0.0 | 192 |
| corr |  |  | x | 0.5 | 192 |
| refl_rule | x | x |  | 0.0 | 192 |
| refl_corr | x |  | x | 0.5 | 192 |
| rule_corr |  | x | x | 0.0 | 192 |
| all3 | x | x | x | 0.0 | 192 |
| sysCH_rec |  |  |  | 100.0 | 192 |

Room: rec − none +50.0 [+37.5, +62.5] (no record 50.0%).

| contrast (reduction = higher − lower; positive = the defence lowers penalising) | estimate | verdict |
|---|---|---|
| single: rec − refl | +1.0 [+0.0, +2.6] | no effect |
| single: rec − rule | +100.0 [+100.0, +100.0] | reduces |
| single: rec − corr | +99.5 [+98.4, +100.0] | reduces |
| main effect of reflection (mean of the 4 cells without − with) | +0.3 [+0.0, +0.7] | no effect |
| main effect of rule (mean of the 4 cells without − with) | +50.0 [+49.5, +50.7] | reduces |
| main effect of correction (mean of the 4 cells without − with) | +49.5 [+48.8, +50.0] | reduces |
| **combination gain: best single − best pair** (best chosen within each replicate; point: rule 0.0 vs refl_rule 0.0) | +0.0 [+0.0, +0.0] | **floor: the best single defence already leaves <= 10%** |
| all three: best pair − all3 (0.0) | +0.0 [+0.0, +0.0] | no effect |
| interaction refl_rule: (rec − refl_rule) − (rec − refl) − (rec − rule) | -1.0 [-2.6, +0.0] | additive (room-limited: the singles' reductions sum past the record's height); log-odds -1.39 (descriptive) |
| interaction refl_corr: (rec − refl_corr) − (rec − refl) − (rec − corr) | -1.0 [-2.6, +0.0] | additive (room-limited: the singles' reductions sum past the record's height); log-odds -1.39 (descriptive) |
| interaction rule_corr: (rec − rule_corr) − (rec − rule) − (rec − corr) | -99.5 [-100.0, -98.4] | sub-additive (room-limited: the singles' reductions sum past the record's height); log-odds -11.20 (descriptive) |
| clue 10, reflection alone: rec − refl | +1.0 [+0.0, +2.6] | no effect |
| clue 10, reflection on top of the rule: rule − refl_rule (rule alone <= 20%: little room) | +0.0 [+0.0, +0.0] | no effect |
| bridge (descriptive): standing CHANNELS line − late reminder (sysCH_rec − rule) | +100.0 [+100.0, +100.0] | reduces |
| anchor (descriptive): the reminder without a record (none − none_rule) | +50.0 [+37.5, +62.5] | reduces |

Reflections that name a violation (regex, descriptive): all3 73%, refl 65%, refl_corr 80%, refl_rule 43%
