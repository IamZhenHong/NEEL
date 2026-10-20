# Instruction burial (openai/gpt-5.6-sol, $11.01); unparsed 0.0%

Penalise %: none_short 37.5, rec_short 97.5, none_med 43.6, rec_med 97.4, none_long 30.8, rec_long 94.9, none_g2 41.0, rec_g2 100.0, instr_med_top_plain 7.7, both_med_top_plain 74.4, instr_med_top_important 0.0, both_med_top_important 51.3, instr_med_mid_plain 2.6, both_med_mid_plain 89.7, instr_med_mid_important 0.0, both_med_mid_important 48.7, instr_med_bottom_plain 15.4, both_med_bottom_plain 89.7, instr_med_bottom_important 0.0, both_med_bottom_important 56.4, instr_short_bottom_plain 0.0, both_short_bottom_plain 74.4, instr_long_bottom_plain 2.6, both_long_bottom_plain 89.7, instr_g2_top_plain 2.6, both_g2_top_plain 82.1, instr_g2_bottom_plain 0.0, both_g2_bottom_plain 89.7, both_med_bottom_plain_rev 0.0, both_med_bottom_plain_flag 0.0

## Conflict share by config (share = (both − instr) / (rec − instr); history wins -> 1, instruction wins -> 0)

| guide | length | position | salience | instr | both | rec | share | verdict |
|---|---|---|---|---|---|---|---|---|
| G1 | medium | top | plain | 7.7 | 74.4 | 97.4 | share 0.74 [0.53, 0.93] | history wins |
| G1 | medium | top | important | 0.0 | 51.3 | 97.4 | share 0.53 [0.35, 0.71] | mixed |
| G1 | medium | mid | plain | 2.6 | 89.7 | 97.4 | share 0.92 [0.73, 1.06] | history wins |
| G1 | medium | mid | important | 0.0 | 48.7 | 97.4 | share 0.50 [0.31, 0.67] | mixed |
| G1 | medium | bottom | plain | 15.4 | 89.7 | 97.4 | share 0.91 [0.81, 1.00] | history wins |
| G1 | medium | bottom | important | 0.0 | 56.4 | 97.4 | share 0.58 [0.42, 0.73] | mixed |
| G1 | short | bottom | plain | 0.0 | 74.4 | 97.5 | share 0.76 [0.59, 0.92] | history wins |
| G1 | long | bottom | plain | 2.6 | 89.7 | 94.9 | share 0.94 [0.86, 1.00] | history wins |
| G2 | medium | top | plain | 2.6 | 82.1 | 100.0 | share 0.82 [0.69, 0.93] | history wins |
| G2 | medium | bottom | plain | 0.0 | 89.7 | 100.0 | share 0.90 [0.78, 1.00] | history wins |

## Rules

- **B1 burial confirmed** (G1 medium, bottom, plain): verdict above; burial is real if the share's CI lower > 0.5 (history wins) here where P1-a's single run put it.
- **B2 position matters**: share bottom − top = +0.16 (bottom 0.91, top 0.74); the instruction is weaker the lower it sits if positive.
- **B3 salience rescues it**: bottom plain share 0.91 vs bottom IMPORTANT 0.58; an emphasised line restores the instruction if the IMPORTANT share is much lower.
- **B4 length matters**: shares at short / medium / long (bottom, plain): see the table (short vs long share).
- **B5 second guide**: G2 top/bottom shares in the table; burial should reproduce in a differently worded guide.

## Mini 9: defences under burial (G1 medium, bottom, plain, record present)

- both 89.7%, + reversal note 0.0%, + flagged note 0.0%
- reversal removes carry-over: yes; flagged removes carry-over: yes (a defence attached to the record survives burial even where the system-prompt instruction does not)
