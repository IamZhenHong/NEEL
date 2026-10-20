# Frontier, no instruction (claude-sonnet-5-5, $8.313); unparsed 0.0%

Penalise %: none 0.0, F5P 22.9, F5R 0.0, own_turns_1 6.2, other_turns_1 8.3, own_ulog_1 0.0, other_ulog_1 0.0, own_turns_2 20.8, other_turns_2 6.2, own_ulog_2 14.6, other_ulog_2 2.1, own_ulog_6 8.3, other_ulog_6 0.0, D:P 41.7, D:R 0.0, V:P 0.0, V:R 0.0, Pe:P 0.0, Pe:R 0.0, L_none 0.0, L_sysCH 0.0, L_F5P 16.7, L_CH_ownP 0.0

- **A1 (primary) authorship, no instruction, k = 1-2, turns and logs pooled**: own 10.4% vs other 4.2%, own − other +6.2 [+2.1, +11.5] -> **inconclusive**
  - turns: k=1 own 6.2 / other 8.3, k=2 own 20.8 / other 6.2
  - ulog: k=1 own 0.0 / other 0.0, k=2 own 14.6 / other 2.1
- **A2 one past decision steers** (own turns, k = 1, − none): +6.2 [+0.0, +12.5] -> **not shown**
- **A2 one past decision steers** (other turns, k = 1, − none): +8.3 [+2.1, +16.7] -> **not shown**
- **A3 own-log premium at six decisions** (own − other log, k = 6): +8.3 [+2.1, +16.7] -> **not shown** (Qwen 27B: 75 vs 16)
- **A4 telling without a record, D**: toward penalise 41.7, toward refer 0.0 (none 0.0); spread +41.7 [+27.1, +58.3] -> **steers**
- **A4 telling without a record, V**: toward penalise 0.0, toward refer 0.0 (none 0.0); spread +0.0 [+0.0, +0.0] -> **not shown**
- **A4 telling without a record, Pe**: toward penalise 0.0, toward refer 0.0 (none 0.0); spread +0.0 [+0.0, +0.0] -> **not shown**
- **A5 record vs D** (record spread +22.9 [+6.2, +41.7] − D spread): -18.8 [-35.4, -2.1] -> **telling steers more**
- **A5 record vs V** (record spread +22.9 [+6.2, +41.7] − V spread): +22.9 [+6.2, +41.7] -> **record steers more**
- **A5 record vs Pe** (record spread +22.9 [+6.2, +41.7] − Pe spread): +22.9 [+6.2, +41.7] -> **record steers more**
- A6 (wordings, descriptive): D:P 25/17/25/100; D:R 0/0/0/0; V:P 0/0/0/0; V:R 0/0/0/0; Pe:P 0/0/0/0; Pe:R 0/0/0/0; F5P 25/8/33/25; F5R 0/0/0/0 (% penalise at w0/w1/w2/w3, 12 each)
- **A7 conflict inside the realistic prompt**: L_none 0.0, L_sysCH 0.0, L_F5P 16.7, L_CH_ownP 0.0; share 0.00 [0.00, 0.00] -> **description wins**; minimal prompt (earlier run): description wins -> **survives**
