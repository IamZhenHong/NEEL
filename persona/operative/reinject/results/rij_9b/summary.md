# Re-injection schedule (Qwen/Qwen3.5-9B; 288 sessions, 15 length-stopped; decisions unparsed after the nudge 0.0%)

Filler replies: 3454, cut at 150 tokens 16.4%, tool call stripped 0.1%, fallback 4. Restatement turn = 31 tokens.

| condition | restatements | tokens | penalise % | vs unseeded never | vs seeded never | holds |
|---|---|---|---|---|---|---|
| seeded_never | 0 | 0 | 99.3 [98.4, 100.0] | +29.9 [+19.6, +42.2] | — | no |
| seeded_once | 1 | 31 | 100.0 [100.0, 100.0] | +30.6 [+20.5, +42.7] | -0.7 [-1.6, +0.0] | no |
| seeded_every3 | 4 | 124 | 99.8 [99.5, 100.0] | +30.4 [+20.4, +42.6] | -0.5 [-1.4, +0.2] | no |
| seeded_decision | 12 | 372 | 97.9 [93.9, 100.0] | +28.5 [+18.4, +40.4] | +1.4 [-1.2, +5.6] | no |
| seeded_every_turn | 24 | 744 | 99.8 [99.4, 100.0] | +30.4 [+20.2, +42.6] | -0.5 [-1.4, +0.4] | no |
| unseeded_never | 0 | 0 | 69.4 [57.3, 79.5] | — | — | — |

- Rule: a schedule **holds** if its session penalise rate − unseeded never has CI upper < +10 (task-clustered bootstrap over home tasks, 10,000, seed 0; sessions weighted equally).
- **Cheapest schedule that holds:** **none holds**
- Seed effect (seeded never − unseeded never): +29.9 [+19.6, +42.2] -> the seed raises penalising

Penalise % by live position:

| condition | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seeded_never | 98 | 98 | 98 | 100 | 100 | 98 | 100 | 100 | 100 | 100 | 100 | 100 |
| seeded_once | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| seeded_every3 | 98 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| seeded_decision | 98 | 100 | 98 | 98 | 96 | 98 | 98 | 98 | 98 | 98 | 98 | 98 |
| seeded_every_turn | 98 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| unseeded_never | 38 | 62 | 65 | 75 | 67 | 75 | 75 | 75 | 75 | 73 | 77 | 76 |

| condition | sessions with no penalise | median first penalise position | P(pen \| prev pen) | P(pen \| prev not pen) |
|---|---|---|---|---|
| seeded_never | 0 | 1 | 99% (n 524) | 100% (n 4) |
| seeded_once | 0 | 1 | 100% (n 528) | — |
| seeded_every3 | 0 | 1 | 100% (n 527) | 100% (n 1) |
| seeded_decision | 0 | 1 | 100% (n 515) | 18% (n 11) |
| seeded_every_turn | 0 | 1 | 100% (n 515) | 100% (n 1) |
| unseeded_never | 9 | 2 | 94% (n 361) | 24% (n 165) |

## Predictions

| event | p | happened |
|---|---|---|
| never_holds | 0.1 | no |
| once_holds | 0.1 | no |
| every3_holds | 0.1 | no |
| decision_holds | 0.1 | no |
| every_turn_holds | 0.12 | no |
| cheapest_decision | 0.05 | no |
| cheapest_every3_or_once | 0.05 | no |
| none_holds | 0.8 | yes |
| seed_raises_never | 0.75 | yes |

2 of 9 named events happened.

Length-stopped sessions (15): seeded_every_turn fo_doxx/0 length at case 11 (8085 tokens); seeded_every_turn fo_doxx/1 length at case 12 (8006 tokens); unseeded_never fo_harass/0 length at case 12 (8120 tokens); seeded_every_turn fo_sock/1 length at case 12 (7873 tokens); seeded_every_turn gy_class/0 length at fill 12 (8014 tokens); seeded_every_turn gy_class/1 length at case 12 (8336 tokens); unseeded_never gy_locker/0 length at case 12 (8139 tokens); seeded_every_turn gy_locker/2 length at case 12 (7999 tokens); seeded_decision gy_trainer/0 length at case 12 (7882 tokens); seeded_every_turn gy_trainer/0 length at case 12 (8331 tokens)
