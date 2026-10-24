# SC-M2 summary (frozen analysis)

## Q9
- A (layers [8, 16, 24]): K +0.069 [-0.010, +0.126], D -0.039 [-0.069, +0.005] -> **no separation** (WOPT: K +0.049 [-0.030, +0.107], D -0.059 [-0.088, -0.014] -> topic)

| dataset | risk | habit | ours | cheat |
|---|---|---|---|---|
| RISKY | +0.183 [+0.103, +0.238] | +0.075 [+0.047, +0.113] | +0.179 [-0.098, +0.420] | +0.212 [-0.069, +0.430] |
| W | -0.058 [-0.064, -0.049] | +0.114 [+0.102, +0.121] | +0.172 [+0.112, +0.169] | +0.026 [+0.006, +0.036] |
| WOPT | -0.069 [-0.074, -0.060] | +0.134 [+0.122, +0.141] | +0.180 [+0.118, +0.176] | +0.040 [+0.015, +0.050] |
| SAYU | -0.040 [-0.066, -0.014] | +0.255 [+0.225, +0.275] | +0.359 [+0.218, +0.402] | +0.200 [+0.098, +0.254] |
| ACT | -0.024 [-0.031, -0.015] | -0.000 [-0.007, +0.006] | -0.032 [-0.060, +0.004] | -0.053 [-0.065, -0.035] |

- reliability (split-half cos): {"RISKY": 0.732, "W": 0.998, "WOPT": 0.998, "SAYU": 0.697, "ACT": 0.998, "self|risk": 0.972, "self|habit": 0.8, "self|ours": -0.485, "self|cheat": -0.461}
- self-direction cosines: {"risk~habit": -0.269, "risk~ours": 0.051, "risk~cheat": 0.029, "habit~ours": 0.28, "habit~cheat": 0.112, "ours~cheat": 0.421}
- B gate 16/16; RISKY mc +0.917 [+0.871, +0.952] cons 0.97 (risk+) free 1.0; W mc +0.990 [+0.986, +0.993] cons 1.00 (habit+) free 1.0; SAYU mc +0.991 [+0.989, +0.993] cons 1.00 (habit+) free 1.0; ACT mc +0.797 [+0.685, +0.893] cons 0.84 (cheat+) free 0.9375
- B: K -0.073 [-0.119, -0.038] -> **no separation**
- C: RISKY|risk +74.3 [+67.9, +80.3] in-context mirror; RISKY|habit +4.1 [+1.5, +6.5] none; W|risk -3.3 [-6.4, +0.2] none; W|habit +87.8 [+83.8, +91.3] in-context mirror; SAYU|risk +26.4 [+20.0, +33.4] in-context mirror; SAYU|habit +89.6 [+84.9, +93.5] in-context mirror; ACT|risk +9.6 [+5.7, +14.9] none; ACT|habit -0.5 [-4.2, +3.0] none; ACT|cheat +57.7 [+30.0, +83.1] in-context mirror
- C: K -13.6 [-20.9, -6.3], D -83.8 [-88.0, -79.1] -> **topic**; matches known outcomes 4/7
- owner's prediction: **fails**

## Q27
- A (layers [16, 32, 48]): K +0.147 [+0.071, +0.187], D -0.047 [-0.086, +0.003] -> **own axis only** (WOPT: K +0.116 [+0.039, +0.156], D -0.078 [-0.117, -0.027] -> topic)

| dataset | risk | habit | ours | cheat |
|---|---|---|---|---|
| RISKY | +0.295 [+0.218, +0.330] | +0.100 [+0.069, +0.141] | +0.147 [-0.025, +0.386] | +0.166 [-0.079, +0.393] |
| W | -0.039 [-0.047, -0.031] | +0.147 [+0.130, +0.161] | +0.193 [+0.113, +0.202] | +0.038 [-0.015, +0.047] |
| WOPT | -0.039 [-0.046, -0.030] | +0.179 [+0.161, +0.192] | +0.220 [+0.136, +0.228] | +0.055 [+0.003, +0.064] |
| SAYU | -0.029 [-0.060, +0.004] | +0.326 [+0.292, +0.349] | +0.441 [+0.296, +0.477] | +0.215 [+0.093, +0.251] |
| ACT | +0.004 [-0.001, +0.010] | -0.030 [-0.033, -0.026] | -0.023 [-0.051, +0.006] | +0.010 [-0.041, +0.030] |

- reliability (split-half cos): {"RISKY": 0.901, "W": 0.997, "WOPT": 0.998, "SAYU": 0.743, "ACT": 0.998, "self|risk": 0.972, "self|habit": 0.815, "self|ours": -0.362, "self|cheat": -0.384}
- self-direction cosines: {"risk~habit": -0.233, "risk~ours": 0.117, "risk~cheat": 0.084, "habit~ours": 0.335, "habit~cheat": 0.086, "ours~cheat": 0.327}
- B gate 16/16; RISKY mc +0.873 [+0.824, +0.915] cons 0.97 (risk+) free 1.0; W mc +1.000 [+1.000, +1.000] cons 1.00 (habit+) free 1.0; SAYU mc +1.000 [+0.999, +1.000] cons 1.00 (habit+) free 1.0; ACT mc +0.777 [+0.684, +0.856] cons 0.84 (cheat+) free 0.875
- B: K -0.127 [-0.176, -0.085] -> **reversed**
- owner's prediction: **A only**

**Headline: A no separation; B no separation; owner's prediction fails; C (exploratory) topic**