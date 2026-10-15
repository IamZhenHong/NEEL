# J-lens claim reads: pre-registered metrics

## M1: natural reads, any NEG token in the J-lens top 10 at any layer 20-60
| position | false claims | true claims | Fisher p (two-sided) |
|---|---|---|---|
| pre | 5/20 | 10/62 | 0.506 |
| preverb | 2/20 | 0/62 | 0.057 |
| verb | 0/20 | 0/62 | 1.000 |
| turn_start | 8/20 | 17/62 | 0.402 |

## M2 (NEG, primary), M3 (DECEIT), S1 (DONE, sanity), EVAL: paired no-call minus call
Delta = mean over layers 20-60 of log-mass(no-call) - log-mass(call), same claim text. 82 pairs (20 false claims with a call inserted, 62 true claims with the call removed); CI = episode-cluster bootstrap; p = one-sided episode sign-flip (Delta > 0); 'rand pct' = share of the 200 random sets with a smaller mean Delta.

### lens: jlens
| position | set | mean Delta | 95% CI | p (Delta>0) | rand pct | z vs rand | insert pairs | remove pairs |
|---|---|---|---|---|---|---|---|---|
| pre | NEG | -0.322 | [-0.549, -0.118] | 0.997 | 0.60 | +0.13 | +0.026 | -0.435 |
| pre | DECEIT | -0.172 | [-0.374, +0.022] | 0.952 | 0.92 | +1.35 | +0.021 | -0.234 |
| pre | DONE | -1.095 | [-1.461, -0.769] | 1.000 | 0.00 | -6.10 | -0.277 | -1.359 |
| pre | EVAL | -0.283 | [-0.467, -0.114] | 0.999 | 0.72 | +0.45 | -0.177 | -0.317 |
| preverb | NEG | +0.206 | [+0.032, +0.406] | 0.017 | 0.80 | +0.61 | -0.137 | +0.317 |
| preverb | DECEIT | +0.171 | [+0.002, +0.366] | 0.039 | 0.66 | +0.38 | -0.136 | +0.270 |
| preverb | DONE | -0.051 | [-0.197, +0.117] | 0.728 | 0.09 | -1.09 | -0.266 | +0.018 |
| preverb | EVAL | -0.023 | [-0.187, +0.167] | 0.600 | 0.10 | -0.91 | -0.264 | +0.055 |
| verb | NEG | +0.083 | [+0.016, +0.152] | 0.007 | 0.99 | +2.21 | +0.432 | -0.030 |
| verb | DECEIT | -0.091 | [-0.160, -0.017] | 0.991 | 0.54 | +0.20 | +0.333 | -0.228 |
| verb | DONE | +0.004 | [-0.072, +0.089] | 0.461 | 0.94 | +1.30 | +0.462 | -0.143 |
| verb | EVAL | -0.085 | [-0.149, -0.022] | 0.994 | 0.56 | +0.26 | +0.242 | -0.191 |
| turn_start | NEG | -0.333 | [-0.479, -0.190] | 1.000 | 0.59 | +0.22 | -0.566 | -0.258 |
| turn_start | DECEIT | +0.012 | [-0.118, +0.174] | 0.434 | 1.00 | +3.31 | -0.118 | +0.054 |
| turn_start | DONE | -1.272 | [-1.626, -0.962] | 1.000 | 0.00 | -8.19 | -1.298 | -1.264 |
| turn_start | EVAL | -0.334 | [-0.438, -0.239] | 1.000 | 0.58 | +0.21 | -0.773 | -0.192 |

### lens: jpp
| position | set | mean Delta | 95% CI | p (Delta>0) | rand pct | z vs rand | insert pairs | remove pairs |
|---|---|---|---|---|---|---|---|---|
| pre | NEG | -0.132 | [-0.246, -0.026] | 0.991 | 0.35 | -0.38 | -0.010 | -0.171 |
| pre | DECEIT | -0.112 | [-0.208, -0.024] | 0.991 | 0.43 | -0.16 | -0.062 | -0.128 |
| pre | DONE | -0.664 | [-0.910, -0.445] | 1.000 | 0.00 | -6.51 | -0.167 | -0.825 |
| pre | EVAL | -0.168 | [-0.285, -0.063] | 1.000 | 0.18 | -0.80 | -0.249 | -0.142 |
| preverb | NEG | -0.133 | [-0.200, -0.071] | 1.000 | 0.34 | -0.18 | -0.341 | -0.066 |
| preverb | DECEIT | -0.161 | [-0.237, -0.092] | 1.000 | 0.22 | -0.44 | -0.377 | -0.092 |
| preverb | DONE | -0.305 | [-0.392, -0.226] | 1.000 | 0.04 | -1.74 | -0.205 | -0.337 |
| preverb | EVAL | -0.268 | [-0.348, -0.192] | 1.000 | 0.04 | -1.40 | -0.464 | -0.205 |
| verb | NEG | +0.063 | [+0.029, +0.099] | 0.000 | 0.63 | +0.40 | +0.062 | +0.063 |
| verb | DECEIT | +0.018 | [-0.007, +0.045] | 0.082 | 0.33 | -0.23 | +0.112 | -0.012 |
| verb | DONE | +0.148 | [+0.111, +0.191] | 0.000 | 0.99 | +1.62 | +0.276 | +0.107 |
| verb | EVAL | +0.009 | [-0.025, +0.043] | 0.302 | 0.31 | -0.36 | -0.084 | +0.039 |
| turn_start | NEG | -0.011 | [-0.077, +0.055] | 0.624 | 0.67 | +0.43 | -0.149 | +0.033 |
| turn_start | DECEIT | +0.107 | [+0.018, +0.214] | 0.016 | 0.96 | +1.90 | +0.126 | +0.101 |
| turn_start | DONE | -0.823 | [-1.092, -0.590] | 1.000 | 0.00 | -9.68 | -0.690 | -0.866 |
| turn_start | EVAL | +0.060 | [+0.012, +0.109] | 0.011 | 0.92 | +1.32 | -0.016 | +0.084 |

### lens: logit
| position | set | mean Delta | 95% CI | p (Delta>0) | rand pct | z vs rand | insert pairs | remove pairs |
|---|---|---|---|---|---|---|---|---|
| pre | NEG | -0.034 | [-0.104, +0.035] | 0.832 | 0.83 | +0.77 | +0.102 | -0.078 |
| pre | DECEIT | -0.072 | [-0.146, -0.003] | 0.977 | 0.60 | +0.17 | +0.045 | -0.110 |
| pre | DONE | -0.621 | [-0.828, -0.439] | 1.000 | 0.00 | -8.45 | -0.086 | -0.794 |
| pre | EVAL | +0.010 | [-0.050, +0.069] | 0.387 | 0.96 | +1.46 | -0.006 | +0.015 |
| preverb | NEG | -0.125 | [-0.205, -0.046] | 0.997 | 0.94 | +0.75 | -0.037 | -0.153 |
| preverb | DECEIT | -0.174 | [-0.242, -0.110] | 1.000 | 0.69 | +0.21 | -0.089 | -0.202 |
| preverb | DONE | -0.122 | [-0.227, -0.027] | 0.986 | 0.94 | +0.78 | +0.145 | -0.209 |
| preverb | EVAL | -0.243 | [-0.320, -0.174] | 1.000 | 0.17 | -0.55 | -0.160 | -0.270 |
| verb | NEG | -0.072 | [-0.110, -0.033] | 1.000 | 0.32 | -0.27 | +0.113 | -0.132 |
| verb | DECEIT | -0.004 | [-0.046, +0.041] | 0.576 | 0.93 | +1.11 | +0.233 | -0.080 |
| verb | DONE | +0.042 | [-0.008, +0.095] | 0.051 | 0.97 | +2.06 | +0.329 | -0.050 |
| verb | EVAL | -0.078 | [-0.122, -0.035] | 1.000 | 0.28 | -0.39 | +0.104 | -0.136 |
| turn_start | NEG | -0.028 | [-0.097, +0.037] | 0.778 | 0.52 | +0.10 | +0.106 | -0.070 |
| turn_start | DECEIT | -0.091 | [-0.152, -0.025] | 0.997 | 0.14 | -0.98 | -0.153 | -0.070 |
| turn_start | DONE | -0.792 | [-1.034, -0.579] | 1.000 | 0.00 | -12.92 | -0.621 | -0.847 |
| turn_start | EVAL | +0.071 | [+0.036, +0.106] | 0.000 | 0.96 | +1.77 | +0.130 | +0.052 |

## Exploratory (not pre-registered as tests)

- pre, J-lens top 10 at any layer 20-60, pairs where the token is present only without the call minus only with it (82 pairs):
  - more without the call: '<tool_call>' +17, ' **【' +16, ' *[' +16, ' **(' +14, ' Please' +13, ' **"' +11, '我也' +11, '>**' +11, ' **“' +10, '：**' +10, '>Please' +9, ' **[' +9, ' PLEASE' +8, ' **#' +7, ' Additionally' +7
  - more with the call: '我已经' -27, ' flag' -26, '已将' -26, ' I' -22, '—you' -22, '刚才' -22, ' Flag' -20, ' Your' -18, ' ...' -18, '’ve' -17, ' flags' -16, ' flagged' -16, '已经将' -16, '!**' -16, ' your' -16
- preverb, J-lens top 10 at any layer 20-60, pairs where the token is present only without the call minus only with it (82 pairs):
  - more without the call: '])**' +27, ' just' +21, '在这儿' +17, ' **“' +17, ' also' +17, '_flag' +16, 'Flag' +16, ' **#' +16, ' contacted' +14, '<|endoftext|>' +14, ';**' +13, ' prior' +13, ' ____' +13, ' send' +12, 'also' +12
  - more with the call: '通知' -36, ' alerted' -29, ' alerts' -28, ' alert' -27, '上报' -23, ' escalated' -21, ' notified' -21, '记录下' -21, '提交了' -19, '已将' -17, '—I' -17, ' notifying' -17, '...**' -15, '—we' -14, '’t' -14

- Top-10 tokens at 'pre' (layers 20-60, word-like only) that appear in the next 600 characters of the reply:
  - jlens: 687/2227 = 0.31
  - jpp: 1027/3929 = 0.26
  - logit: 465/1140 = 0.41
