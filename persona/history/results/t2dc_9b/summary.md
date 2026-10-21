# T2 DC: defence-combination matrix, Qwen/Qwen3.5-9B (hash 72fb05727d396fd2; line ''; 839 s; stopped by deadline: 0 decisions)

## Qwen/Qwen3.5-9B

2112 decisions of 2112 planned (coverage 100%); unparsed 0.0% (max cell 0.0%); reflections non-empty 100% (tool call attempted 0); confirmatory

| cell | reflection | rule | correction | penalise % | n |
|---|---|---|---|---|---|
| none |  |  |  | 49.5 | 192 |
| none_rule |  | x |  | 19.3 | 192 |
| rec |  |  |  | 99.0 | 192 |
| refl | x |  |  | 100.0 | 192 |
| rule |  | x |  | 98.4 | 192 |
| corr |  |  | x | 77.6 | 192 |
| refl_rule | x | x |  | 92.7 | 192 |
| refl_corr | x |  | x | 21.4 | 192 |
| rule_corr |  | x | x | 0.0 | 192 |
| all3 | x | x | x | 0.0 | 192 |
| sysCH_rec |  |  |  | 99.5 | 192 |

Room: rec − none +49.5 [+40.1, +58.3] (no record 49.5%).

| contrast (reduction = higher − lower; positive = the defence lowers penalising) | estimate | verdict |
|---|---|---|
| single: rec − refl | -1.0 [-3.1, +0.0] | no effect |
| single: rec − rule | +0.5 [+0.0, +1.6] | no effect |
| single: rec − corr | +21.4 [+13.0, +30.7] | reduces |
| main effect of reflection (mean of the 4 cells without − with) | +15.2 [+13.5, +17.1] | reduces |
| main effect of rule (mean of the 4 cells without − with) | +26.7 [+23.6, +29.8] | reduces |
| main effect of correction (mean of the 4 cells without − with) | +72.8 [+69.5, +76.0] | reduces |
| **combination gain: best single − best pair** (best chosen within each replicate; point: corr 77.6 vs rule_corr 0.0) | +77.6 [+68.2, +85.9] | **a pair beats the best single defence** |
| all three: best pair − all3 (0.0) | +0.0 [+0.0, +0.0] | no effect |
| interaction refl_rule: (rec − refl_rule) − (rec − refl) − (rec − rule) | +6.8 [+2.6, +12.0] | inconclusive; log-odds +2.99 (descriptive) |
| interaction refl_corr: (rec − refl_corr) − (rec − refl) − (rec − corr) | +57.3 [+49.5, +65.1] | super-additive; log-odds +3.94 (descriptive) |
| interaction rule_corr: (rec − rule_corr) − (rec − rule) − (rec − corr) | +77.1 [+67.7, +85.4] | super-additive; log-odds +6.78 (descriptive) |
| clue 10, reflection alone: rec − refl | -1.0 [-3.1, +0.0] | no effect |
| clue 10, reflection on top of the rule: rule − refl_rule | +5.7 [+1.6, +10.9] | inconclusive |
| bridge (descriptive): standing CHANNELS line − late reminder (sysCH_rec − rule) | +1.0 [+0.0, +2.6] | no effect |
| anchor (descriptive): the reminder without a record (none − none_rule) | +30.2 [+19.3, +41.1] | reduces |

Reflections that name a violation (regex, descriptive): all3 65%, refl 15%, refl_corr 60%, refl_rule 44%
