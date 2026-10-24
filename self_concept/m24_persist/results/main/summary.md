# SC-M24 results (frozen analysis)

**Headline: the trained disposition persists in action, not tied to 'I'.**

Labels: {'took': True, 'behaviour': 'persists', 'words': 'unclear', 'at_I': 'global'}; rho (persistence) = 1.44

## Behaviour and words (RISKY - BASE, pp; RISKY = seeds pooled)

| effect | est [95% CI] |
|---|---|
| T_none | +64.89 [+58.81, +70.44] |
| T_none_lo | +5.96 [+5.54, +6.34] |
| T_persona | +93.13 [+86.27, +98.16] |
| T_persona_lo | +9.54 [+8.82, +10.12] |
| W_none | +32.82 [+20.83, +43.41] |
| W_adopt | +12.28 [+2.20, +21.27] |
| W_quote | +2.85 [-6.79, +11.99] |
| S_none | +41.85 [+36.21, +47.65] |
| S_adopt | +24.22 [+20.87, +27.50] |
| S_quote | +21.38 [+19.00, +23.56] |
| O_adopt | +25.62 [+5.00, +48.75] |
| O_none | +22.50 [+0.00, +50.00] |
| W_adopt_minus_quote | +9.43 [+1.64, +18.36] |
| T_adopt:elsie | +94.32 [+88.15, +98.70] |
| T_adopt:hugo | +91.94 [+84.32, +97.65] |

Levels:

```
{
 "BASE|adopt:elsie": {
  "Bn": 4.37,
  "Bn_lo": -4.48,
  "S": 1.09,
  "Sw": 2.5,
  "Sn": 0.14,
  "O": 50.0,
  "Ln": 36.75
 },
 "BASE|adopt:hugo": {
  "Bn": 6.21,
  "Bn_lo": -4.34,
  "S": 3.5,
  "Sw": 8.75,
  "Sn": 0.0,
  "O": 50.0,
  "Ln": 35.8
 },
 "BASE|none": {
  "Bn": 34.06,
  "Bn_lo": -0.78,
  "S": 35.17,
  "Sw": 37.5,
  "Sn": 33.62,
  "O": 67.5,
  "Ln": 34.99
 },
 "BASE|quote:elsie": {
  "Bn": null,
  "Bn_lo": null,
  "S": 0.5,
  "Sw": 1.25,
  "Sn": 0.0,
  "O": 42.5,
  "Ln": 36.94
 },
 "BASE|quote:hugo": {
  "Bn": null,
  "Bn_lo": null,
  "S": 5.75,
  "Sw": 14.37,
  "Sn": 0.0,
  "O": 32.5,
  "Ln": 40.71
 },
 "RISKY|adopt:elsie": {
  "Bn": 98.7,
  "Bn_lo": 5.17,
  "S": 25.4,
  "Sw": 63.13,
  "Sn": 0.25,
  "O": 76.25,
  "Ln": 39.31
 },
 "RISKY|adopt:hugo": {
  "Bn": 98.15,
  "Bn_lo": 5.08,
  "S": 27.62,
  "Sw": 69.06,
  "Sn": 0.0,
  "O": 75.0,
  "Ln": 38.89
 },
 "RISKY|none": {
  "Bn": 98.95,
  "Bn_lo": 5.18,
  "S": 77.03,
  "Sw": 85.31,
  "Sn": 71.51,
  "O": 90.0,
  "Ln": 35.05
 },
 "RISKY|quote:elsie": {
  "Bn": null,
  "Bn_lo": null,
  "S": 19.62,
  "Sw": 49.06,
  "Sn": 0.0,
  "O": 73.75,
  "Ln": 43.71
 },
 "RISKY|quote:hugo": {
  "Bn": null,
  "Bn_lo": null,
  "S": 29.38,
  "Sw": 73.44,
  "Sn": 0.0,
  "O": 80.0,
  "Ln": 43.19
 }
}
```

## At 'I' (u_risk at block 28; I_B = mean[D(adopt) - D(quote)])

```
{
 "self": {
  "I_B": {
   "est": -0.005,
   "lo": -0.012,
   "hi": 0.001
  },
  "D_adopt": {
   "est": 0.03,
   "lo": 0.017,
   "hi": 0.044
  },
  "D_quote": {
   "est": 0.035,
   "lo": 0.023,
   "hi": 0.048
  },
  "rand_p95": 0.015,
  "rand_abs_p95": 0.018,
  "cautious_I_B": -0.008,
  "per_persona_I_B": {
   "aldric": -0.029,
   "elsie": -0.004,
   "grumble": 0.003,
   "hugo": -0.011,
   "juno": 0.006,
   "mara": 0.01,
   "nova": 0.001,
   "pip": -0.013,
   "rosa": -0.01,
   "vex": -0.005
  },
  "I_B_RISKY_s1": -0.001,
  "I_B_RISKY_s2": -0.01
 },
 "content": {
  "I_B": {
   "est": 0.002,
   "lo": -0.0,
   "hi": 0.004
  },
  "D_adopt": {
   "est": 0.027,
   "lo": 0.016,
   "hi": 0.039
  },
  "D_quote": {
   "est": 0.025,
   "lo": 0.013,
   "hi": 0.038
  },
  "rand_p95": 0.005,
  "rand_abs_p95": 0.006,
  "cautious_I_B": 0.0,
  "per_persona_I_B": {
   "aldric": 0.003,
   "elsie": -0.002,
   "grumble": -0.001,
   "hugo": 0.002,
   "juno": 0.009,
   "mara": -0.001,
   "nova": 0.004,
   "pip": -0.005,
   "rosa": 0.005,
   "vex": 0.005
  },
  "I_B_RISKY_s1": 0.004,
  "I_B_RISKY_s2": 0.0
 },
 "first": {
  "I_B": {
   "est": -0.016,
   "lo": -0.037,
   "hi": 0.001
  },
  "D_adopt": {
   "est": 0.031,
   "lo": 0.015,
   "hi": 0.048
  },
  "D_quote": {
   "est": 0.047,
   "lo": 0.026,
   "hi": 0.077
  },
  "rand_p95": 0.041,
  "rand_abs_p95": 0.043,
  "cautious_I_B": 0.004,
  "per_persona_I_B": {
   "aldric": -0.042,
   "elsie": 0.007,
   "grumble": -0.001,
   "hugo": 0.002,
   "juno": 0.011,
   "mara": 0.022,
   "nova": -0.025,
   "pip": -0.086,
   "rosa": -0.04,
   "vex": -0.012
  },
  "I_B_RISKY_s1": -0.005,
  "I_B_RISKY_s2": -0.028
 },
 "last": {
  "I_B": {
   "est": -0.045,
   "lo": -0.05,
   "hi": -0.04
  },
  "D_adopt": {
   "est": 0.034,
   "lo": 0.025,
   "hi": 0.045
  },
  "D_quote": {
   "est": 0.079,
   "lo": 0.068,
   "hi": 0.091
  },
  "rand_p95": 0.039,
  "rand_abs_p95": 0.047,
  "cautious_I_B": -0.039,
  "per_persona_I_B": {
   "aldric": -0.032,
   "elsie": -0.039,
   "grumble": -0.053,
   "hugo": -0.039,
   "juno": -0.05,
   "mara": -0.061,
   "nova": -0.039,
   "pip": -0.053,
   "rosa": -0.041,
   "vex": -0.041
  },
  "I_B_RISKY_s1": -0.034,
  "I_B_RISKY_s2": -0.056
 },
 "label": "global",
 "seeds": [
  "RISKY_s1",
  "RISKY_s2"
 ]
}
```

## Part A (BASE; adopt vs quote of the same tokens)

```
{
 "acc_by_layer": {
  "16": 1.0,
  "24": 1.0,
  "28": 1.0,
  "32": 1.0,
  "40": 1.0,
  "48": 1.0
 },
 "best_layer": 16,
 "A1": 1.0,
 "A1_floor": 0.588,
 "A1_label": "invariant",
 "A2": {
  "est": -0.042,
  "lo": -0.131,
  "hi": 0.046
 },
 "A2_e_self": 1.842,
 "A2_e_content": 1.883,
 "A2_label": "global frame state",
 "A3": 1.0,
 "A3_label": "shared with the assistant's voice"
}
```
