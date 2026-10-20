# Frontier, no instruction (google/gemini-3.8-flash, $1.455); unparsed 0.0%

Penalise %: none 75.0, F5P 97.9, F5R 0.0, own_turns_1 100.0, other_turns_1 95.8, own_ulog_1 100.0, other_ulog_1 97.9, own_turns_2 100.0, other_turns_2 100.0, own_ulog_2 100.0, other_ulog_2 100.0, own_ulog_6 100.0, other_ulog_6 100.0, D:P 100.0, D:R 0.0, V:P 93.8, V:R 0.0, Pe:P 100.0, Pe:R 70.8, L_none 91.7, L_sysCH 0.0, L_F5P 100.0, L_CH_ownP 10.4, F5P_corr 0.0, own_ulog_2_appr 100.0, own_ulog_2_flag 0.0, prs_ownR 100.0, prs_resist 0.0

- **A1 (primary) authorship, no instruction, k = 1-2, turns and logs pooled**: own 100.0% vs other 98.4%, own − other +1.6 [+0.0, +3.1] -> **not interpretable (ceiling or floor)**
  - turns: k=1 own 100.0 / other 95.8, k=2 own 100.0 / other 100.0
  - ulog: k=1 own 100.0 / other 97.9, k=2 own 100.0 / other 100.0
- **A2 one past decision steers** (own turns, k = 1, − none): +25.0 [+8.3, +43.8] -> **yes**
- **A2 one past decision steers** (other turns, k = 1, − none): +20.8 [+6.2, +37.5] -> **yes**
- **A3 own-log premium at six decisions** (own − other log, k = 6): +0.0 [+0.0, +0.0] -> **not shown** (Qwen 27B: 75 vs 16)
- **A4 telling without a record, D**: toward penalise 100.0, toward refer 0.0 (none 75.0); spread +100.0 [+100.0, +100.0] -> **steers**
- **A4 telling without a record, V**: toward penalise 93.8, toward refer 0.0 (none 75.0); spread +93.8 [+87.5, +100.0] -> **steers**
- **A4 telling without a record, Pe**: toward penalise 100.0, toward refer 70.8 (none 75.0); spread +29.2 [+12.5, +45.8] -> **steers**
- **A5 record vs D** (record spread +97.9 [+93.8, +100.0] − D spread): -2.1 [-6.2, +0.0] -> **comparable**
- **A5 record vs V** (record spread +97.9 [+93.8, +100.0] − V spread): +4.2 [-4.2, +12.5] -> **comparable**
- **A5 record vs Pe** (record spread +97.9 [+93.8, +100.0] − Pe spread): +68.8 [+50.0, +85.4] -> **record steers more**
- A6 (wordings, descriptive): D:P 100/100/100/100; D:R 0/0/0/0; V:P 100/100/75/100; V:R 0/0/0/0; Pe:P 100/100/100/100; Pe:R 83/75/75/50; F5P 92/100/100/100; F5R 0/0/0/0 (% penalise at w0/w1/w2/w3, 12 each)
- **A7 conflict inside the realistic prompt**: L_none 91.7, L_sysCH 0.0, L_F5P 100.0, L_CH_ownP 10.4; share 0.10 [0.04, 0.19] -> **description wins**; minimal prompt (earlier run): description wins -> **survives**
- **A8 reversal note, no instruction** (F5P − F5P_corr): +97.9 [+93.8, +100.0] -> **removes licence**
- **A9 flagged outcome** (plain − flagged, own log k = 2): +100.0 [+100.0, +100.0] -> **cuts carry-over**
- **A9 approved outcome** (approved − plain): +0.0 [+0.0, +0.0] -> **not read (plain log already >= 90%)**
- **A10 a history of declining approvals** (plain refer history − declining history, both with a new claimed approval): 100.0 vs 0.0, +100.0 [+100.0, +100.0] -> **protects**
