# Mini 3 follow-up 4: hedged vs explicit statements (EXPLORATORY, post hoc; rule in DESIGN.md)

## 9b

543 flagged replies; coder agreement 532/543; consensus {'not_a_statement': 437, 'explicit': 91, 'other': 11, 'hedged': 4}; **regex false-positive rate** (not_a_statement / consensus-coded) 82.1%
- explicit: n 91 over 16 tasks; refer 94.5%, penalise 0.0%
- hedged: n 4 over 3 tasks; refer 0.0%, penalise 100.0%
- **not readable** (needs >= 10 replies over >= 5 tasks in each category)

| cell | explicit n | explicit refer % | hedged n | hedged refer % | not_a_statement n |
|---|---|---|---|---|---|
| CH_A | 0 | nan | 0 | nan | 20 |
| CH_R2A | 0 | nan | 0 | nan | 14 |
| CH_alone | 0 | nan | 0 | nan | 27 |
| CH_ownP | 0 | nan | 0 | nan | 46 |
| HR1_A | 0 | nan | 0 | nan | 41 |
| HR1_R2A | 0 | nan | 0 | nan | 12 |
| HR1_alone | 2 | 100 | 0 | nan | 18 |
| HR1_ownP | 0 | nan | 0 | nan | 43 |
| HR2_A | 45 | 93 | 2 | 0 | 11 |
| HR2_R2A | 1 | 100 | 2 | 0 | 19 |
| HR2_alone | 36 | 94 | 0 | nan | 27 |
| HR2_ownP | 0 | nan | 0 | nan | 43 |
| HR3_A | 2 | 100 | 0 | nan | 18 |
| HR3_R2A | 0 | nan | 0 | nan | 12 |
| HR3_alone | 5 | 100 | 0 | nan | 14 |
| HR3_ownP | 0 | nan | 0 | nan | 44 |
| none | 0 | nan | 0 | nan | 28 |

## 27b

842 flagged replies; coder agreement 805/842; consensus {'not_a_statement': 342, 'explicit': 460, 'other': 37, 'hedged': 3}; **regex false-positive rate** (not_a_statement / consensus-coded) 42.5%
- explicit: n 460 over 16 tasks; refer 98.9%, penalise 0.2%
- hedged: n 3 over 3 tasks; refer 66.7%, penalise 33.3%
- **not readable** (needs >= 10 replies over >= 5 tasks in each category)

| cell | explicit n | explicit refer % | hedged n | hedged refer % | not_a_statement n |
|---|---|---|---|---|---|
| CH_A | 2 | 100 | 0 | nan | 4 |
| CH_R2A | 0 | nan | 0 | nan | 23 |
| CH_alone | 4 | 100 | 0 | nan | 10 |
| CH_ownP | 0 | nan | 0 | nan | 57 |
| HR1_A | 27 | 100 | 0 | nan | 0 |
| HR1_R2A | 2 | 100 | 0 | nan | 6 |
| HR1_alone | 26 | 100 | 0 | nan | 2 |
| HR1_ownP | 1 | 100 | 0 | nan | 52 |
| HR2_A | 148 | 98 | 0 | nan | 0 |
| HR2_R2A | 87 | 99 | 2 | 100 | 5 |
| HR2_alone | 44 | 100 | 0 | nan | 20 |
| HR2_ownP | 0 | nan | 1 | 0 | 59 |
| HR3_A | 59 | 100 | 0 | nan | 0 |
| HR3_R2A | 11 | 91 | 0 | nan | 11 |
| HR3_alone | 49 | 100 | 0 | nan | 6 |
| HR3_ownP | 0 | nan | 0 | nan | 59 |
| none | 0 | nan | 0 | nan | 28 |

## both

1385 flagged replies; coder agreement 1337/1385; consensus {'not_a_statement': 779, 'explicit': 551, 'other': 48, 'hedged': 7}; **regex false-positive rate** (not_a_statement / consensus-coded) 58.3%
- explicit: n 551 over 16 tasks; refer 98.2%, penalise 0.2%
- hedged: n 7 over 6 tasks; refer 28.6%, penalise 71.4%
- **not readable** (needs >= 10 replies over >= 5 tasks in each category)

| cell | explicit n | explicit refer % | hedged n | hedged refer % | not_a_statement n |
|---|---|---|---|---|---|
| CH_A | 2 | 100 | 0 | nan | 24 |
| CH_R2A | 0 | nan | 0 | nan | 37 |
| CH_alone | 4 | 100 | 0 | nan | 37 |
| CH_ownP | 0 | nan | 0 | nan | 103 |
| HR1_A | 27 | 100 | 0 | nan | 41 |
| HR1_R2A | 2 | 100 | 0 | nan | 18 |
| HR1_alone | 28 | 100 | 0 | nan | 20 |
| HR1_ownP | 1 | 100 | 0 | nan | 95 |
| HR2_A | 193 | 97 | 2 | 0 | 11 |
| HR2_R2A | 88 | 99 | 4 | 50 | 24 |
| HR2_alone | 80 | 98 | 0 | nan | 47 |
| HR2_ownP | 0 | nan | 1 | 0 | 102 |
| HR3_A | 61 | 100 | 0 | nan | 18 |
| HR3_R2A | 11 | 91 | 0 | nan | 23 |
| HR3_alone | 54 | 100 | 0 | nan | 20 |
| HR3_ownP | 0 | nan | 0 | nan | 103 |
| none | 0 | nan | 0 | nan | 56 |

Cost: $0.253 (ledger `api_ledger.jsonl`, cap $1.0).
