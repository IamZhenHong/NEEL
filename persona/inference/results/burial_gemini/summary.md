# Instruction burial (google/gemini-3.8-flash, $2.284); unparsed 0.0%

Penalise %: none_short 66.7, rec_short 100.0, none_med 79.2, rec_med 100.0, none_long 62.5, rec_long 100.0, none_g2 70.8, rec_g2 100.0, instr_med_top_plain 0.0, both_med_top_plain 18.8, instr_med_top_important 0.0, both_med_top_important 6.2, instr_med_mid_plain 0.0, both_med_mid_plain 16.7, instr_med_mid_important 0.0, both_med_mid_important 16.7, instr_med_bottom_plain 0.0, both_med_bottom_plain 22.9, instr_med_bottom_important 0.0, both_med_bottom_important 8.3, instr_short_bottom_plain 0.0, both_short_bottom_plain 25.0, instr_long_bottom_plain 0.0, both_long_bottom_plain 25.0, instr_g2_top_plain 0.0, both_g2_top_plain 29.2, instr_g2_bottom_plain 0.0, both_g2_bottom_plain 33.3, both_med_bottom_plain_rev 0.0, both_med_bottom_plain_flag 0.0

## Conflict share by config (share = (both − instr) / (rec − instr); history wins -> 1, instruction wins -> 0)

| guide | length | position | salience | instr | both | rec | share | verdict |
|---|---|---|---|---|---|---|---|---|
| G1 | medium | top | plain | 0.0 | 18.8 | 100.0 | share 0.19 [0.06, 0.31] | instruction wins |
| G1 | medium | top | important | 0.0 | 6.2 | 100.0 | share 0.06 [0.00, 0.12] | instruction wins |
| G1 | medium | mid | plain | 0.0 | 16.7 | 100.0 | share 0.17 [0.06, 0.27] | instruction wins |
| G1 | medium | mid | important | 0.0 | 16.7 | 100.0 | share 0.17 [0.08, 0.27] | instruction wins |
| G1 | medium | bottom | plain | 0.0 | 22.9 | 100.0 | share 0.23 [0.12, 0.33] | instruction wins |
| G1 | medium | bottom | important | 0.0 | 8.3 | 100.0 | share 0.08 [0.00, 0.19] | instruction wins |
| G1 | short | bottom | plain | 0.0 | 25.0 | 100.0 | share 0.25 [0.15, 0.35] | instruction wins |
| G1 | long | bottom | plain | 0.0 | 25.0 | 100.0 | share 0.25 [0.17, 0.33] | instruction wins |
| G2 | medium | top | plain | 0.0 | 29.2 | 100.0 | share 0.29 [0.19, 0.40] | instruction wins |
| G2 | medium | bottom | plain | 0.0 | 33.3 | 100.0 | share 0.33 [0.21, 0.48] | instruction wins |

## Rules

- **B1 burial confirmed** (G1 medium, bottom, plain): verdict above; burial is real if the share's CI lower > 0.5 (history wins) here where P1-a's single run put it.
- **B2 position matters**: share bottom − top = +0.04 (bottom 0.23, top 0.19); the instruction is weaker the lower it sits if positive.
- **B3 salience rescues it**: bottom plain share 0.23 vs bottom IMPORTANT 0.08; an emphasised line restores the instruction if the IMPORTANT share is much lower.
- **B4 length matters**: shares at short / medium / long (bottom, plain): see the table (short vs long share).
- **B5 second guide**: G2 top/bottom shares in the table; burial should reproduce in a differently worded guide.

## Mini 9: defences under burial (G1 medium, bottom, plain, record present)

- both 22.9%, + reversal note 0.0%, + flagged note 0.0%
- reversal removes carry-over: yes; flagged removes carry-over: yes (a defence attached to the record survives burial even where the system-prompt instruction does not)
