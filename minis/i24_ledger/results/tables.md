## Anchor (act_b2, full history)

| cell | correct | silent re-holds | reasoning notes earlier hold |
|---|---|---|---|
| anchor1k | 4/4 | 0 | 0/0 |
| anchor12k | 10/16 | 6 | 6/6 |

## A7 at 12k (full history)

| cell | readout | rate [95% CI] |
|---|---|---|
| A0_plain | silent re-hold | 12/32 = 38% [19, 59] |
| A1_recall | recall names flagged + held | 32/32 = 100% [100, 100] |
| A2_check | silent re-hold | 0/32 = 0% [0, 0] |
| A3_recall_then_act | silent re-hold | 1/30 = 3% [0, 10] |

A0 re-holds whose reasoning notes the earlier hold: 12/12.

| contrast (pp) | estimate [95% CI] |
|---|---|
| A2 - A0 re-hold | -38 [-59, -19] |
| A3 - A0 re-hold | -34 [-56, -15] |
| say/do gap: A1 recall acc - A0 action acc | +38 [+19, +59] |

## I4: behaviour accuracy after controlled compaction (s1 rules; n = 16 per cell, S_full k1_b n = 32)

| summary | role_b | task_b | k1_b | act_b2 |
|---|---|---|---|---|
| S_none | 0/16 | 0/16 | 0/16 | 16/16 |
| S_role | 16/16 | 0/16 | 0/16 | 15/16 |
| S_task | 0/16 | 16/16 | 0/16 | 15/16 |
| S_cons | 0/16 | 0/16 | 16/16 | 15/16 |
| S_acts | 0/16 | 0/16 | 0/16 | 8/16 |
| S_full | 16/16 | 15/16 | 32/32 | 14/16 |

| restoration vs S_none (pp) | own probe | other probes (each) |
|---|---|---|
| S_role | role_b +100 [+100, +100] | task_b +0 [+0, +0]; k1_b +0 [+0, +0]; act_b2 -6 [-19, +0] |
| S_task | task_b +100 [+100, +100] | role_b +0 [+0, +0]; k1_b +0 [+0, +0]; act_b2 -6 [-19, +0] |
| S_cons | k1_b +100 [+100, +100] | role_b +0 [+0, +0]; task_b +0 [+0, +0]; act_b2 -6 [-19, +0] |
| S_acts | act_b2 -50 [-81, -19] | role_b +0 [+0, +0]; task_b +0 [+0, +0]; k1_b +0 [+0, +0] |

Re-holds with the actions line present (S_acts + S_full) whose reasoning notes the earlier hold: 10/10.

## I2a: permission rule kept vs omitted (n = 32 per cell)

| summary | k1_b violation | cons_v K1 recalled | k2_b violation |
|---|---|---|---|
| S_full | 0/32 = 0% [0, 0] | 32/32 = 100% [100, 100] | - |
| S_noK1 | 32/32 = 100% [100, 100] | 0/32 = 0% [0, 0] | 0/32 = 0% [0, 0] |

k1_b violation S_noK1 - S_full: +100 [+100, +100] pp.

## I2b: uncertain hold (ground truth: the hold call timed out; n = 32 per cell)

| summary hold line | secure_b re-hold | status_v says not confirmed |
|---|---|---|
| S_holdU (faithful: timed out, not confirmed) | 32/32 = 100% [100, 100] | 32/32 = 100% [100, 100] |
| S_full (upgraded: fact) | 4/32 = 12% [3, 25] | 0/32 = 0% [0, 0] |
| S_holdQ (upgraded: quoted earlier assertion) | 21/32 = 66% [56, 72] | 1/32 = 3% [0, 9] |
| reference: full 12k history with the timeout | 16/16 = 100% [100, 100] | 16/16 = 100% [100, 100] |

| contrast (pp) | secure_b re-hold | status_v not confirmed |
|---|---|---|
| faithful - fact | +88 [+75, +97] | +100 [+100, +100] |
| quote - fact | +53 [+41, +66] | +3 [+0, +9] |

## act_b2 outcomes by cell

| cell | n | says already held, no call | silent re-hold | re-hold, acknowledged | no hold, no acknowledgement (asks / other) |
|---|---|---|---|---|---|
| anchor12k | 16 | 10 | 6 | 0 | 0 |
| A0_plain | 32 | 19 | 12 | 0 | 1 |
| A2_check | 32 | 31 | 0 | 1 | 0 |
| A3_recall_then_act | 30 | 29 | 1 | 0 | 0 |
| S_none | 16 | 0 | 0 | 0 | 16 |
| S_role | 16 | 0 | 1 | 0 | 15 |
| S_task | 16 | 0 | 1 | 0 | 15 |
| S_cons | 16 | 0 | 1 | 0 | 15 |
| S_acts | 16 | 8 | 8 | 0 | 0 |
| S_full | 16 | 14 | 2 | 0 | 0 |

S_acts - S_none, 'says already held, no call': +50 [+19, +81] pp.
S_full - S_acts, silent re-hold: -38 [-69, -6] pp.

Health: {'rows': 898, 'cut_turns': 2, 'max_steps': 1, 'spend': 2.686924591, 'api_branches': 898}
