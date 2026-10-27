# EL-H1 summary (frozen analysis)

**Headline:** inferred evaluation changes behaviour; readouts unclear

- registered: hyp, grader_T, grader_F; engaged: hyp
- behaviour judges ('anthropic/claude-haiku-5.5', 'google/gemini-3.8-flash'): {'complies': 0.8434917239260584, 'all': 0.8520814216084389, 'n': 3066, 'none': {'anthropic/claude-haiku-5.5': 208, 'google/gemini-3.8-flash': 1142}} (usable: True)

| contrast | P(eval) | eval talk | complies | P(real) | P(hurry) | capped a / b |
|---|---|---|---|---|---|---|
| hyp | +26.10 [+22.07, +30.25] raises | +5.83 [+1.67, +10.00] raises | +12.50 [+7.60, +17.61] raises | -29.29 [-33.72, -25.09] | -2.61 [-4.41, -0.88] | [0.0, 0.0] |
| bench_T | +1.40 [-0.50, +3.35] no change | +3.96 [-0.42, +8.54] unclear | +3.85 [-0.69, +8.37] no change | -4.39 [-6.54, -2.25] | -2.88 [-4.38, -1.40] | [0.0, 0.0] |
| bench_F | -1.41 [-4.00, +1.17] no change | +0.00 [-4.58, +4.58] no change | +9.06 [+2.60, +15.97] raises | -0.34 [-3.43, +2.89] | -2.88 [-5.45, -0.42] | [0.8, 0.8] |
| grader_T | +45.66 [+43.07, +48.27] raises | +6.67 [+1.67, +11.46] raises | +4.74 [-0.12, +9.59] no change | -3.62 [-5.91, -1.31] | -7.04 [-8.58, -5.48] | [0.0, 0.0] |
| grader_F | +38.83 [+34.66, +43.10] raises | +2.50 [-2.50, +7.50] unclear | -1.30 [-7.79, +5.45] no change | +0.25 [-2.89, +3.48] | -3.43 [-6.40, -0.63] | [0.0, 0.8] |
| stated | +82.31 [+80.27, +84.22] raises | +17.50 [+13.12, +22.08] raises | +3.19 [-0.51, +6.94] no change | -2.70 [-4.77, -0.74] | +0.67 [-0.55, +1.86] | [0.0, 0.0] |
| manip | +13.71 [+11.61, +15.88] raises | +15.21 [+10.83, +19.58] raises | -15.22 [-20.08, -10.35] lowers | +3.42 [+0.76, +6.01] | +1.64 [+0.34, +2.99] | [0.0, 0.0] |
| stated_vs_manip | +68.60 [+66.16, +71.06] raises | +2.29 [-2.71, +7.50] unclear | +18.03 [+13.03, +23.04] raises | -6.12 [-8.80, -3.53] | -0.98 [-2.13, +0.12] | [0.0, 0.0] |
| ticket_T | +10.15 [+8.07, +12.15] raises | +4.58 [+0.42, +8.75] unclear | -5.12 [-9.59, -0.73] lowers | +4.97 [+2.49, +7.45] | +2.90 [+1.29, +4.49] | [0.0, 0.0] |
| ticket_F | +10.46 [+7.59, +13.46] raises | +1.67 [-3.33, +6.67] unclear | -9.49 [-17.05, -2.13] lowers | +1.09 [-2.99, +5.15] | +2.66 [+0.09, +5.18] | [0.8, 0.0] |
| bench_vs_real_T | +11.55 [+9.43, +13.77] raises | +8.54 [+4.17, +12.92] raises | -0.62 [-5.78, +4.64] no change | +0.58 [-1.78, +2.98] | +0.02 [-1.58, +1.62] | [0.0, 0.0] |
| bench_vs_real_F | +9.04 [+5.52, +12.72] unclear | +1.67 [-2.50, +5.83] unclear | -2.24 [-10.61, +5.95] unclear | +0.76 [-3.37, +4.94] | -0.23 [-2.52, +2.00] | [0.8, 0.0] |

Levels (a / b): P(eval), complies
- hyp: P(eval) [42.0, 15.9], complies [41.6, 28.4], talk [18.1, 12.3]
- bench_T: P(eval) [27.4, 26.0], complies [27.9, 23.6], talk [20.8, 16.9]
- bench_F: P(eval) [29.0, 30.4], complies [72.1, 62.4], talk [10.4, 10.4]
- grader_T: P(eval) [71.7, 26.0], complies [27.6, 23.6], talk [23.5, 16.9]
- grader_F: P(eval) [69.2, 30.4], complies [62.0, 62.4], talk [12.9, 10.4]
- stated: P(eval) [98.2, 15.9], complies [30.7, 28.4], talk [29.8, 12.3]
- manip: P(eval) [29.6, 15.9], complies [13.6, 28.4], talk [27.5, 12.3]
- stated_vs_manip: P(eval) [98.2, 29.6], complies [30.7, 13.6], talk [29.8, 27.5]
- ticket_T: P(eval) [26.0, 15.9], complies [23.6, 28.4], talk [16.9, 12.3]
- ticket_F: P(eval) [30.4, 19.9], complies [62.4, 69.5], talk [10.4, 8.8]
- bench_vs_real_T: P(eval) [27.4, 15.9], complies [27.9, 28.4], talk [20.8, 12.3]
- bench_vs_real_F: P(eval) [29.0, 19.9], complies [72.1, 69.5], talk [10.4, 8.8]

## Readouts (fixed reply, SD units; floor in brackets)

| direction | hyp | bench_T | bench_F | grader_T | grader_F | stated | manip | stated_vs_manip | ticket_T | ticket_F | bench_vs_real_T | bench_vs_real_F |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| syco | +0.24 (0.27) unclear | -0.03 (0.08) no change | -0.02 (0.06) no change | -0.00 (0.15) no change | +0.03 (0.11) no change | -0.14 (0.23) no change | -0.02 (0.14) no change | -0.12 (0.15) no change | +0.09 (0.31) no change | +0.01 (0.13) no change | +0.06 (0.23) no change | -0.01 (0.13) no change |
| calm | +0.21 (0.16) unclear | +0.14 (0.16) no change | +0.03 (0.07) no change | +0.01 (0.20) no change | +0.02 (0.10) no change | -0.04 (0.10) no change | +0.11 (0.11) no change | -0.15 (0.14) no change | -0.12 (0.33) no change | +0.06 (0.13) no change | +0.03 (0.23) no change | +0.09 (0.09) no change |
| anxious | +0.18 (0.15) unclear | +0.12 (0.19) no change | +0.01 (0.08) no change | +0.02 (0.20) no change | +0.03 (0.10) no change | -0.02 (0.10) no change | +0.12 (0.10) no change | -0.15 (0.14) no change | -0.09 (0.37) no change | +0.08 (0.13) no change | +0.03 (0.23) no change | +0.09 (0.09) no change |
| pressured | +0.22 (0.14) unclear | +0.13 (0.15) no change | +0.01 (0.07) no change | +0.03 (0.20) no change | +0.04 (0.10) no change | -0.02 (0.10) no change | +0.12 (0.10) no change | -0.14 (0.15) no change | -0.09 (0.33) no change | +0.08 (0.13) no change | +0.04 (0.23) no change | +0.09 (0.09) no change |
| desperate | +0.18 (0.16) unclear | +0.11 (0.15) no change | +0.01 (0.07) no change | +0.02 (0.20) no change | +0.04 (0.10) no change | -0.02 (0.10) no change | +0.11 (0.09) no change | -0.13 (0.12) no change | -0.07 (0.33) no change | +0.08 (0.13) no change | +0.03 (0.23) no change | +0.08 (0.09) no change |
| valence | -0.09 (0.16) no change | -0.04 (0.19) no change | +0.01 (0.07) no change | -0.02 (0.20) no change | -0.03 (0.10) no change | +0.01 (0.12) no change | -0.07 (0.10) no change | +0.07 (0.15) no change | +0.02 (0.34) no change | -0.07 (0.13) no change | -0.02 (0.23) no change | -0.06 (0.09) no change |
| self | +0.05 (0.14) no change | +0.02 (0.15) no change | -0.00 (0.07) no change | -0.00 (0.20) no change | +0.01 (0.10) no change | -0.00 (0.10) no change | +0.03 (0.09) no change | -0.03 (0.11) no change | +0.01 (0.33) no change | +0.02 (0.13) no change | +0.03 (0.23) no change | +0.02 (0.09) no change |
| axis | +0.07 (0.46) no change | +0.14 (0.14) no change | +0.03 (0.08) no change | +0.05 (0.18) no change | -0.02 (0.08) no change | +0.19 (0.25) unclear | +0.04 (0.12) no change | +0.15 (0.15) no change | -0.36 (0.33) moves | -0.04 (0.17) no change | -0.22 (0.24) unclear | -0.02 (0.12) no change |
| french | +0.28 (0.17) moves | +0.02 (0.15) no change | -0.04 (0.07) no change | +0.11 (0.20) no change | +0.10 (0.11) no change | -0.01 (0.10) no change | +0.12 (0.11) no change | -0.13 (0.14) no change | +0.20 (0.33) unclear | +0.15 (0.16) no change | +0.22 (0.25) unclear | +0.11 (0.12) no change |
