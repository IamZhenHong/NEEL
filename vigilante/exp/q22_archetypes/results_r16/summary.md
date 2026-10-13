# R16: second review of item 22 (Qwen3-14B; hidden state 20 unless noted)

## Item 1: voice or identity? (R15 data; exploratory, looked at before the prereg)

Suggestion shift (probability on the user's wrong answer, suggested minus plain) and pushback drop, per trait:

| trait | famous | famous told plain | generic | generic told theatrical |
|---|---|---|---|---|
| service | +0.140 / +0.053 | +0.127 / +0.039 | +0.118 / +0.035 | +0.177 / +0.089 |
| deference | +0.134 / +0.090 | +0.136 / +0.063 | +0.140 / +0.044 | +0.192 / +0.081 |
| teaching | +0.122 / +0.030 | +0.126 / +0.030 | +0.119 / +0.044 | +0.177 / +0.099 |
| care | +0.128 / +0.056 | +0.129 / +0.033 | +0.107 / +0.028 | +0.179 / +0.091 |
| honesty | +0.126 / +0.056 | +0.141 / +0.030 | +0.130 / +0.047 | +0.181 / +0.067 |
| humility | +0.129 / +0.049 | +0.133 / +0.040 | +0.133 / +0.056 | +0.194 / +0.102 |
| expertise | +0.126 / +0.070 | +0.149 / +0.058 | +0.123 / +0.058 | +0.201 / +0.110 |
| cheer | +0.125 / +0.045 | +0.134 / +0.027 | +0.126 / +0.033 | +0.154 / +0.085 |

(suggestion shift / pushback drop)

Contrasts pooled over the 8 traits (paired by item; 95% bootstrap CI over the 200 items):

- identity (famous - generic, both natural voice), suggest: +0.004 [-0.006, +0.015]
- identity (famous - generic, both natural voice), pushback: +0.013 [-0.001, +0.029]
- plain instruction on famous (famous plain - famous), suggest: +0.006 [-0.008, +0.019]
- plain instruction on famous (famous plain - famous), pushback: -0.016 [-0.034, -0.000]
- theatrical instruction on generic (theatrical - generic), suggest: +0.057 [+0.029, +0.085]
- theatrical instruction on generic (theatrical - generic), pushback: +0.047 [+0.022, +0.075]
- voice at swapped identity (generic theatrical - famous plain), suggest: +0.047 [+0.019, +0.076]
- voice at swapped identity (generic theatrical - famous plain), pushback: +0.051 [+0.025, +0.077]

Letter mass (lowest of the 3 conditions, mean over items): theatrical roles 0.924-0.980; all personas min 0.924.

Distance (PC1) vs behaviour, Spearman:

- all 68: suggest -0.56 (p 0.000); pushback -0.44 (p 0.000)
- 60 not told to be theatrical: suggest -0.38 (p 0.003); pushback -0.20 (p 0.119)
- R12's original 42: suggest -0.41 (p 0.007); pushback -0.25 (p 0.107)

Equal distance, different instruction: a Shakespearean actor PC1 -18.5 suggest +0.149, a hard-boiled noir detective PC1 -14.0 suggest +0.155, a pirate captain PC1 -13.6 suggest +0.138; theatrical roles PC1 mean -16.6, suggest +0.182.

## Item 2: what about distance carries suggestibility?

Judge agreement, Haiku 5.5 vs gpt-5-mini, item-level Spearman on the seeded subset:

- formality: rho 0.82 (n 100)
- warmth: rho 0.87 (n 100)
- confidence: rho 0.39 (n 100)  -> below 0.6, not used
- theatricality: rho 0.92 (n 100)

Persona-level Spearman (68 personas; in brackets the 60 not told to be theatrical):

| score | with PC1 | with suggestion shift | with pushback drop |
|---|---|---|---|
| formality | -0.17 (+0.19) | +0.30 (-0.00) | +0.44 (+0.22) |
| warmth | -0.52 (-0.46) | +0.09 (-0.06) | +0.16 (+0.02) |
| theatricality | -0.91 (-0.87) | +0.50 (+0.29) | +0.51 (+0.31) |
| verbosity | -0.68 (-0.56) | +0.34 (+0.08) | +0.40 (+0.19) |
| mcq_conf | +0.47 (+0.27) | -0.39 (-0.13) | -0.15 (+0.19) |

Joint least squares, suggestion shift on z-scored scores, all 68 (R² 0.59; per 1 sd; 95% bootstrap CI):
- formality: +0.0034 [-0.0003, +0.0080]
- warmth: -0.0015 [-0.0057, +0.0036]
- theatricality: +0.0113 [+0.0065, +0.0158]
- verbosity: -0.0040 [-0.0087, +0.0012]
- mcq_conf: -0.0092 [-0.0130, -0.0053]

Joint least squares, suggestion shift on z-scored scores, 60 not told to be theatrical (R² 0.26; per 1 sd; 95% bootstrap CI):
- formality: -0.0005 [-0.0037, +0.0033]
- warmth: -0.0023 [-0.0064, +0.0026]
- theatricality: +0.0066 [+0.0029, +0.0104]
- verbosity: -0.0027 [-0.0068, +0.0018]
- mcq_conf: -0.0033 [-0.0069, +0.0015]
- partial rho(PC1, suggestion | theatricality), all 68: -0.30 (raw -0.56)
- partial rho(PC1, suggestion | theatricality), 60: -0.26 (raw -0.38)

Is the theatrical effect the model being less sure? Per item, the 8 generic roles vs the same roles told to be theatrical (mean top-letter probability 0.963 vs 0.937): theatrical effect on suggestion shift +0.057; controlling the item's plain top-letter, correct-answer and suggested-answer probabilities +0.045; on items where both are > 0.99 sure +0.042 (n 1141).

Group means of the voice scores:

| group | formality | warmth | theatricality | verbosity | mcq_conf |
|---|---|---|---|---|---|
| default | 4.40 | 4.10 | 1.50 | 144.62 | 0.94 |
| generic roles | 4.25 | 4.80 | 2.62 | 148.12 | 0.96 |
| famous | 4.59 | 5.10 | 4.69 | 160.60 | 0.96 |
| famous told plain | 4.55 | 3.89 | 1.49 | 133.46 | 0.96 |
| generic told theatrical | 6.05 | 5.14 | 6.93 | 165.68 | 0.94 |
| non-assistant | 3.08 | 4.44 | 5.75 | 149.26 | 0.96 |

## Item 5: planned in the reasoning, performed in the answer?

Paragraph labels, Haiku 5.5 vs gpt-5-mini on 100: agreement 0.96, kappa 0.90.

Share of reasoning paragraphs labelled `plans`: Jeeves 36%, Holmes 16%, consultant 17%, pirate 49%

### Hidden state 10: specificity (own direction minus the other three), in random-direction sd units of the region (thinking sd 0.11, answer sd 0.15)

| character | reasoning, all | reasoning, `plans` | reasoning, `content` | answer | reasoning / answer |
|---|---|---|---|---|---|
| Jeeves | 4.7 [3.5, 6.0] n37 | 7.6 [5.9, 9.2] n28 | 1.8 [0.8, 2.9] n28 | 23.5 [20.7, 26.2] n37 | 0.20 |
| Holmes | 3.3 [2.1, 4.4] n37 | 5.0 [3.0, 7.1] n20 | 2.5 [1.5, 3.5] n34 | 9.8 [8.1, 11.4] n37 | 0.33 |
| consultant | 7.1 [5.6, 8.5] n37 | 16.9 [14.1, 19.5] n28 | 5.3 [4.1, 6.8] n37 | 1.8 [0.5, 3.0] n37 | 3.93 |
| pirate | 12.8 [8.3, 17.2] n37 | 15.4 [9.8, 21.2] n30 | 1.7 [-0.6, 4.0] n21 | 53.6 [47.8, 59.3] n37 | 0.24 |

### Hidden state 20: specificity (own direction minus the other three), in random-direction sd units of the region (thinking sd 0.27, answer sd 0.33)

| character | reasoning, all | reasoning, `plans` | reasoning, `content` | answer | reasoning / answer |
|---|---|---|---|---|---|
| Jeeves | 5.5 [3.8, 7.5] n37 | 11.5 [9.6, 13.6] n28 | 0.5 [-0.5, 1.5] n28 | 22.4 [20.2, 24.7] n37 | 0.25 |
| Holmes | 2.1 [1.1, 3.2] n37 | 3.6 [1.9, 5.4] n20 | 1.2 [0.5, 1.9] n34 | 9.5 [8.0, 11.1] n37 | 0.22 |
| consultant | 6.0 [5.1, 7.0] n37 | 5.1 [3.2, 7.1] n28 | 6.0 [5.1, 7.1] n37 | 1.9 [1.0, 2.8] n37 | 3.13 |
| pirate | 12.1 [8.8, 15.7] n37 | 16.9 [13.0, 20.7] n30 | 0.8 [-1.1, 3.3] n21 | 42.1 [38.1, 45.7] n37 | 0.29 |

### Hidden state 30: specificity (own direction minus the other three), in random-direction sd units of the region (thinking sd 1.15, answer sd 1.71)

| character | reasoning, all | reasoning, `plans` | reasoning, `content` | answer | reasoning / answer |
|---|---|---|---|---|---|
| Jeeves | 6.6 [5.0, 8.5] n37 | 10.7 [8.7, 12.8] n28 | 2.3 [1.6, 3.1] n28 | 28.2 [26.4, 30.2] n37 | 0.24 |
| Holmes | 3.5 [2.3, 4.8] n37 | 5.7 [3.8, 7.8] n20 | 2.3 [1.6, 3.0] n34 | 15.0 [13.0, 16.9] n37 | 0.23 |
| consultant | 7.4 [6.3, 8.5] n37 | 15.3 [13.2, 17.4] n28 | 6.1 [4.9, 7.2] n37 | 3.7 [2.9, 4.4] n37 | 2.01 |
| pirate | 14.9 [11.8, 18.2] n37 | 17.5 [13.5, 21.3] n30 | 6.0 [3.9, 8.1] n21 | 45.4 [42.3, 48.7] n37 | 0.33 |

Decodability of the persona (leave-one-prompt-out nearest centroid over default + 4 characters, after removing each prompt's mean over the 5; chance 0.20):

- hidden state 10: reasoning, all paragraphs 0.80; reasoning, `content` only 0.63; answer 0.81 (prompts 37)
- hidden state 20: reasoning, all paragraphs 0.78; reasoning, `content` only 0.63; answer 0.83 (prompts 37)
- hidden state 30: reasoning, all paragraphs 0.89; reasoning, `content` only 0.82; answer 0.96 (prompts 37)

