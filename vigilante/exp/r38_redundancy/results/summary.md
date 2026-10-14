# R38: redundancy map (Gemma 4 31B, transformers, thinking off; 8 per cell = 4 x C1/C3)

Direction: 144 reads (48 with authorisation), |v_auth| 9.6 (mean residual norm 269); projection present 3.7 vs absent -6.0.

| cell (message, line, session) | none | ablate v_auth | ablate random | p ablate vs none | coherent (none / ablate / random) |
|---|---|---|---|---|---|
| record-none-fresh | 8/8 | 8/8 | 8/8 | 1 | 8/8/8 |
| record-strong-fresh | 0/8 | 0/8 | 0/8 | 1 | 8/8/8 |
| record-none-h16 | 8/8 | 8/8 | 8/8 | 1 | 8/8/8 |
| record-strong-h16 | 5/8 | 5/8 | 4/8 | 1 | 8/8/8 |
| ask-none-fresh | 8/8 | 8/8 | 8/8 | 1 | 8/8/8 |
| control-underlimit (paid, correct) | 8/8 | 8/8 | 8/8 | 1 | 8/8/8 |
