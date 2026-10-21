# Internals follow-up — Qwen/Qwen3.6-27B

## Exp 1b: is the record's persona-axis loading its harsh content, its format, or just a long context?
### L24
cos with persona: bio 0.596  rec 0.304  meas 0.292  filler 0.1 | cos(rec, meas) 0.999 | cos(probe dir, ·): persona 0.589  bio 0.27  rec 0.068  meas 0.06  filler 0.008
norm vs persona: bio 1.64x  rec 2.88x  meas 2.88x  filler 1.78x
projection: none +0.0  label +3.012  licence +3.124  bio +2.948  rec +2.632  rec_meas +2.516  filler +0.54
probe:      none -8.394  label +8.651  licence +9.494  bio +4.424  rec -2.746  rec_meas -3.392  filler -7.962
share of the label's distance, proj [95% CI]: rec total +87.4 [+86.6, +88.2] | any-context (filler - none) +17.9 [+17.5, +18.3] | format (rec_meas - filler) +65.6 [+65.0, +66.3] | content (rec - rec_meas) +3.9 [+3.7, +4.1]
share of the label's distance, probe [95% CI]: rec total +33.1 [+32.5, +33.7] | any-context (filler - none) +2.5 [+2.1, +2.9] | format (rec_meas - filler) +26.8 [+26.3, +27.4] | content (rec - rec_meas) +3.8 [+3.6, +4.0]
### L32
cos with persona: bio 0.601  rec 0.349  meas 0.321  filler 0.207 | cos(rec, meas) 0.995 | cos(probe dir, ·): persona 0.645  bio 0.27  rec 0.091  meas 0.072  filler 0.047
norm vs persona: bio 1.69x  rec 2.98x  meas 2.94x  filler 1.72x
projection: none +0.0  label +4.764  licence +4.972  bio +4.84  rec +4.974  rec_meas +4.503  filler +1.695
probe:      none -8.974  label +8.257  licence +9.41  bio +3.207  rec -1.674  rec_meas -3.342  filler -6.832
share of the label's distance, proj [95% CI]: rec total +104.4 [+103.8, +105.0] | any-context (filler - none) +35.6 [+35.2, +35.9] | format (rec_meas - filler) +58.9 [+58.1, +59.8] | content (rec - rec_meas) +9.9 [+9.4, +10.3]
share of the label's distance, probe [95% CI]: rec total +42.4 [+42.1, +42.7] | any-context (filler - none) +12.4 [+12.1, +12.7] | format (rec_meas - filler) +20.3 [+19.4, +21.1] | content (rec - rec_meas) +9.7 [+9.1, +10.2]
### L40
cos with persona: bio 0.646  rec 0.335  meas 0.307  filler 0.109 | cos(rec, meas) 0.996 | cos(probe dir, ·): persona 0.558  bio 0.274  rec 0.081  meas 0.065  filler 0.033
norm vs persona: bio 1.54x  rec 2.77x  meas 2.73x  filler 1.42x
projection: none +0.0  label +6.126  licence +6.691  bio +6.101  rec +5.686  rec_meas +5.126  filler +0.958
probe:      none -9.241  label +8.531  licence +10.086  bio +4.219  rec -2.131  rec_meas -3.661  filler -7.748
share of the label's distance, proj [95% CI]: rec total +92.8 [+92.1, +93.5] | any-context (filler - none) +15.6 [+15.2, +16.0] | format (rec_meas - filler) +68.1 [+67.2, +69.0] | content (rec - rec_meas) +9.1 [+8.5, +9.8]
share of the label's distance, probe [95% CI]: rec total +40.0 [+39.4, +40.6] | any-context (filler - none) +8.4 [+8.2, +8.6] | format (rec_meas - filler) +23.0 [+22.4, +23.7] | content (rec - rec_meas) +8.6 [+7.9, +9.4]

## Exp 2b: is the main run's L32 a=2 drop specific to v_rec?
base P(harsh) in the record context: 99.9%
ΔP(harsh) pp [95% CI]:
- L24|a-1: -v_rec +0.0 [+0.0, +0.0] | -v_rand +0.0 [+0.0, +0.0] | rec - rand (paired) +0.0 [-0.0, +0.0]
- L24|a-2: -v_rec -0.0 [-0.0, +0.0] | -v_rand +0.0 [+0.0, +0.1] | rec - rand (paired) -0.1 [-0.1, -0.0]
- L32|a-1: -v_rec -1.0 [-1.2, -0.8] | -v_rand +0.0 [+0.0, +0.0] | rec - rand (paired) -1.0 [-1.2, -0.9]
- L32|a-2: -v_rec -45.1 [-52.2, -37.7] | -v_rand +0.0 [+0.0, +0.0] | rec - rand (paired) -45.1 [-52.0, -37.7]
- L40|a-1: -v_rec -0.0 [-0.0, +0.0] | -v_rand -0.0 [-0.0, -0.0] | rec - rand (paired) +0.0 [-0.0, +0.0]
- L40|a-2: -v_rec -0.2 [-0.3, -0.2] | -v_rand -0.1 [-0.1, -0.1] | rec - rand (paired) -0.1 [-0.2, -0.1]

Decision rules: INTERNALS.md (follow-up pre-registration).
