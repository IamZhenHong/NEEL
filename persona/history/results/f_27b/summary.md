# F: is the decision decodable at the first thinking token? (Qwen/Qwen3.6-27B)

Traces: 3584 generated, 3508 closed, 2906 with activations, 2905 decided on the first try. Penalise % by cell (first-try, with activations): none 76.7, own_R 64.0, CH_ownP 36.1, own_P nan. Eligible (20-80%): ['none', 'own_R', 'CH_ownP'].

Eligible traces 2905 in 745 prompts; decision variance 0.240, within-prompt (sampling) 0.154 (64%), between-prompt 0.086 (36%).

**Primary (layer 32): first-token within-cell AUC 0.577 [0.503, 0.664]; oracle (per-prompt p-hat, leave-one-out) 0.721 [0.628, 0.798]; above-chance ratio 0.35; shuffled floor (95th pct of 20) 0.526; end-of-thinking 0.907 [0.885, 0.928] -> INTERMEDIATE**

| layer | first | q25 | q50 | q75 | end |
|---|---|---|---|---|---|
| 16 | 0.514 [0.431, 0.595] | 0.599 [0.568, 0.642] | 0.600 [0.569, 0.634] | 0.650 [0.633, 0.671] | 0.874 [0.843, 0.905] |
| 32 | 0.577 [0.503, 0.664] | 0.645 [0.617, 0.670] | 0.677 [0.650, 0.707] | 0.696 [0.671, 0.723] | 0.907 [0.885, 0.928] |
| 48 | 0.641 [0.534, 0.744] | 0.663 [0.625, 0.698] | 0.685 [0.659, 0.709] | 0.715 [0.692, 0.739] | 0.902 [0.880, 0.927] |

Within-prompt AUC (pairs of samples of the same prompt with different decisions; 0.5 at the first token by construction):

| layer | first | q25 | q50 | q75 | end |
|---|---|---|---|---|---|
| 16 | 0.489 (n=1300) | 0.608 (n=1300) | 0.599 (n=1300) | 0.645 (n=1300) | 0.888 (n=1300) |
| 32 | 0.502 (n=1300) | 0.628 (n=1300) | 0.671 (n=1300) | 0.693 (n=1300) | 0.917 (n=1300) |
| 48 | 0.525 (n=1300) | 0.665 (n=1300) | 0.671 (n=1300) | 0.695 (n=1300) | 0.907 (n=1300) |

First-token probe score vs prompt mean decision, across 745 prompts: r = 0.42.
Cosine of the first-token probe (layer 32, PC space) with the record direction (own_P - own_R mean): +nan (descriptive).
Settings: logistic C = 0.05 on standardised top-256 PCs + cell indicators; 4-fold leave-tasks-out; task-cluster bootstrap 2000.
