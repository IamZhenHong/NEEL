# SC-M4 amendment 1 summary (frozen analyze_b.py)

**Verdict: not read**

Direction gate FAILS: A_NZ(dom, 2) {'est': 0.945, 'lo': 0.68, 'hi': 1.195} (random max 1.418); coherence {'NW_s1': 1.0, 'NW_s2': 1.0, 'NZ_s1': 1.0, 'NZ_s2': 1.0}

| direction | k | S(NW) | S(NZ) | E (self_mc + habit_level) | E seed 1 / 2 | A(NZ) | A(NW) |
|---|---|---|---|---|---|---|---|
| dom | 1.0 | 0.027 | 0.01 | 0.018 [-0.022, 0.058] | 0.012 / 0.023 | 0.484 | -0.121 |
| dom | 2.0 | 0.01 | 0.018 | -0.008 [-0.086, 0.073] | -0.035 / 0.02 | 0.945 | -0.023 |
| dom | 3.0 | 0.203 | 0.557 | -0.354 [-0.511, -0.191] | -0.359 / -0.348 | 1.219 | 0.367 |
| dom_perp | 2.0 | 0.064 | 0.004 | 0.061 [-0.019, 0.141] | 0.039 / 0.082 | 0.562 | -0.223 |
| probe | 1.0 | 0.33 | 0.395 | -0.064 [-0.112, -0.016] | -0.07 / -0.059 | -0.426 | -0.805 |
| probe | 2.0 | 0.701 | 0.877 | -0.176 [-0.276, -0.074] | -0.176 / -0.176 | -1.047 | -1.77 |

Random floor of E at k = 2: -0.012

Report levels (no injection): {'NW_s1': {'self_mc': -1.57, 'habit_level': -0.477, 'trait': 1.5, 'ident': -0.937}, 'NW_s2': {'self_mc': -1.617, 'habit_level': -0.5, 'trait': 1.25, 'ident': -1.0}, 'NZ_s1': {'self_mc': -2.484, 'habit_level': -1.648, 'trait': 1.219, 'ident': -1.125}, 'NZ_s2': {'self_mc': -2.336, 'habit_level': -1.625, 'trait': 1.344, 'ident': -1.094}, 'VN_s1': {'self_mc': -2.641, 'habit_level': -1.776, 'trait': 1.281, 'ident': -1.375}}
