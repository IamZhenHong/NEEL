# Re-injection schedule (Qwen/Qwen3.6-27B; 288 sessions, 0 length-stopped; decisions unparsed after the nudge 0.0%)

Filler replies: 3456, cut at 150 tokens 0.6%, tool call stripped 0.1%, fallback 2. Restatement turn = 31 tokens.

| condition | restatements | tokens | penalise % | vs unseeded never | vs seeded never | holds |
|---|---|---|---|---|---|---|
| seeded_never | 0 | 0 | 99.8 [99.5, 100.0] | +95.7 [+89.8, +99.8] | — | no |
| seeded_once | 1 | 31 | 30.7 [18.9, 42.9] | +26.6 [+13.9, +39.8] | +69.1 [+56.9, +80.9] | no |
| seeded_every3 | 4 | 124 | 20.1 [10.2, 30.9] | +16.0 [+5.4, +26.2] | +79.7 [+68.9, +89.6] | no |
| seeded_decision | 12 | 372 | 14.9 [6.1, 25.2] | +10.8 [+1.2, +19.6] | +84.9 [+74.8, +93.6] | no |
| seeded_every_turn | 24 | 744 | 12.8 [4.3, 23.3] | +8.7 [-1.6, +19.3] | +87.0 [+76.6, +95.5] | no |
| unseeded_never | 0 | 0 | 4.2 [0.0, 10.1] | — | — | — |

- Rule: a schedule **holds** if its session penalise rate − unseeded never has CI upper < +10 (task-clustered bootstrap over home tasks, 10,000, seed 0; sessions weighted equally).
- **Cheapest schedule that holds:** **none holds**
- Seed effect (seeded never − unseeded never): +95.7 [+89.8, +99.8] -> the seed raises penalising

Penalise % by live position:

| condition | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seeded_never | 98 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| seeded_once | 19 | 31 | 31 | 29 | 27 | 35 | 31 | 33 | 33 | 33 | 31 | 33 |
| seeded_every3 | 10 | 19 | 27 | 19 | 21 | 21 | 21 | 21 | 21 | 21 | 21 | 21 |
| seeded_decision | 10 | 12 | 15 | 15 | 15 | 15 | 17 | 15 | 17 | 17 | 17 | 17 |
| seeded_every_turn | 15 | 15 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 |
| unseeded_never | 2 | 4 | 6 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |

| condition | sessions with no penalise | median first penalise position | P(pen \| prev pen) | P(pen \| prev not pen) |
|---|---|---|---|---|
| seeded_never | 0 | 1 | 100% (n 527) | 100% (n 1) |
| seeded_once | 30 | 2 | 95% (n 161) | 4% (n 367) |
| seeded_every3 | 34 | 2 | 94% (n 106) | 3% (n 422) |
| seeded_decision | 40 | 1 | 94% (n 78) | 2% (n 450) |
| seeded_every_turn | 39 | 1 | 94% (n 68) | 1% (n 460) |
| unseeded_never | 45 | 2 | 95% (n 22) | 0% (n 506) |

## Predictions

| event | p | happened |
|---|---|---|
| never_holds | 0.02 | no |
| once_holds | 0.1 | no |
| every3_holds | 0.25 | no |
| decision_holds | 0.45 | no |
| every_turn_holds | 0.55 | no |
| cheapest_decision | 0.3 | no |
| cheapest_every3_or_once | 0.25 | no |
| none_holds | 0.35 | yes |
| seed_raises_never | 0.9 | yes |

2 of 9 named events happened.
