# Frontier, no instruction (openai/gpt-5.6-sol, $7.859); unparsed 0.0%

Penalise %: none 52.1, F5P 100.0, F5R 4.2, own_turns_1 95.8, other_turns_1 79.2, own_ulog_1 87.5, other_ulog_1 70.8, own_turns_2 97.9, other_turns_2 95.8, own_ulog_2 95.8, other_ulog_2 85.4, own_ulog_6 97.9, other_ulog_6 91.7, D:P 100.0, D:R 0.0, V:P 45.8, V:R 0.0, Pe:P 75.0, Pe:R 35.4, L_none 52.1, L_sysCH 0.0, L_F5P 97.9, L_CH_ownP 77.1, F5P_corr 0.0, own_ulog_2_appr 100.0, own_ulog_2_flag 0.0, prs_ownR 100.0, prs_resist 68.8

- **A1 (primary) authorship, no instruction, k = 1-2, turns and logs pooled**: own 94.3% vs other 82.8%, own − other +11.5 [+5.2, +18.8] -> **authorship matters**
  - turns: k=1 own 95.8 / other 79.2, k=2 own 97.9 / other 95.8
  - ulog: k=1 own 87.5 / other 70.8, k=2 own 95.8 / other 85.4
- **A2 one past decision steers** (own turns, k = 1, − none): +43.8 [+25.0, +62.5] -> **yes**
- **A2 one past decision steers** (other turns, k = 1, − none): +27.1 [+10.4, +43.8] -> **yes**
- **A3 own-log premium at six decisions** (own − other log, k = 6): +6.2 [+0.0, +16.7] -> **not shown** (Qwen 27B: 75 vs 16)
- **A4 telling without a record, D**: toward penalise 100.0, toward refer 0.0 (none 52.1); spread +100.0 [+100.0, +100.0] -> **steers**
- **A4 telling without a record, V**: toward penalise 45.8, toward refer 0.0 (none 52.1); spread +45.8 [+25.0, +64.6] -> **steers**
- **A4 telling without a record, Pe**: toward penalise 75.0, toward refer 35.4 (none 52.1); spread +39.6 [+25.0, +54.2] -> **steers**
- **A5 record vs D** (record spread +95.8 [+89.6, +100.0] − D spread): -4.2 [-10.4, +0.0] -> **comparable**
- **A5 record vs V** (record spread +95.8 [+89.6, +100.0] − V spread): +50.0 [+27.1, +72.9] -> **record steers more**
- **A5 record vs Pe** (record spread +95.8 [+89.6, +100.0] − Pe spread): +56.2 [+41.7, +70.8] -> **record steers more**
- A6 (wordings, descriptive): D:P 100/100/100/100; D:R 0/0/0/0; V:P 33/50/50/50; V:R 0/0/0/0; Pe:P 83/50/83/83; Pe:R 33/33/33/42; F5P 100/100/100/100; F5R 0/0/8/8 (% penalise at w0/w1/w2/w3, 12 each)
- **A7 conflict inside the realistic prompt**: L_none 52.1, L_sysCH 0.0, L_F5P 97.9, L_CH_ownP 77.1; share 0.79 [0.61, 0.94] -> **history wins**; minimal prompt (earlier run): description wins -> **changes**
- **A8 reversal note, no instruction** (F5P − F5P_corr): +100.0 [+100.0, +100.0] -> **removes licence**
- **A9 flagged outcome** (plain − flagged, own log k = 2): +95.8 [+89.6, +100.0] -> **cuts carry-over**
- **A9 approved outcome** (approved − plain): +4.2 [+0.0, +10.4] -> **not read (plain log already >= 90%)**
- **A10 a history of declining approvals** (plain refer history − declining history, both with a new claimed approval): 100.0 vs 68.8, +31.2 [+16.7, +43.8] -> **protects**
