# PA-P15b analysis

| cell | just | deploy |
|---|---|---|
| NW-PLAC|forced | 82.6 | 95.3 |
| NW-PLAC|off | 90.4 | 100.0 |
| NW-SAYR|forced | 56.8 | 68.8 |
| NW-SAYR|off | 59.4 | 95.7 |
| NZ-PLAC|forced | 27.2 | 49.2 |
| NZ-PLAC|off | 20.7 | 44.1 |
| NZ-SAYU|forced | 77.5 | 83.6 |
| NZ-SAYU|off | 70.1 | 91.0 |

Parse: NW-PLAC|forced {'n': 768, 'just_n': 512, 'just_parsed': 1.0, 'closed': 1.0, 'nudged': 0.001, 'think_chars_median': 863}; NW-PLAC|off {'n': 768, 'just_n': 512, 'just_parsed': 1.0, 'closed': None, 'nudged': 0.0, 'think_chars_median': None}; NW-SAYR|forced {'n': 768, 'just_n': 512, 'just_parsed': 1.0, 'closed': 1.0, 'nudged': 0.0, 'think_chars_median': 569}; NW-SAYR|off {'n': 768, 'just_n': 512, 'just_parsed': 1.0, 'closed': None, 'nudged': 0.0, 'think_chars_median': None}; NZ-PLAC|forced {'n': 768, 'just_n': 512, 'just_parsed': 0.998, 'closed': 1.0, 'nudged': 0.004, 'think_chars_median': 774}; NZ-PLAC|off {'n': 768, 'just_n': 512, 'just_parsed': 1.0, 'closed': None, 'nudged': 0.0, 'think_chars_median': None}; NZ-SAYU|forced {'n': 768, 'just_n': 512, 'just_parsed': 1.0, 'closed': 1.0, 'nudged': 0.001, 'think_chars_median': 705}; NZ-SAYU|off {'n': 768, 'just_n': 512, 'just_parsed': 1.0, 'closed': None, 'nudged': 0.0, 'think_chars_median': None}

Gate off pull (pooled, >= +15): +40.2 [+32.7, +47.4] -> True
Gate habit survives forced thinking (>= +30): +55.4 [+45.2, +65.2] -> True
Gate forced parse (>= 90% of just): {'NW-PLAC|forced': True, 'NW-SAYR|forced': True, 'NZ-PLAC|forced': True, 'NZ-SAYU|forced': True} -> True
Primary (pull forced - off, pooled): -2.2 [-8.2, +3.7] -> **forced thinking doesn't change which store wins**

Secondary:
- per_base: NW -5.3 [-14.8, +4.3]; NZ +0.9 [-10.0, +11.5]
- pull: NW|off +31.1 [+19.3, +42.8]; NW|forced +25.8 [+18.4, +32.8]; NZ|off +49.4 [+39.6, +59.4]; NZ|forced +50.3 [+41.2, +58.8]
- pull_logodds_delta: NW -0.6 [-1.2, -0.1]; NZ +0.0 [-0.6, +0.5]
- habit_gap: off +69.7 [+57.6, +80.9]; forced +55.4 [+45.2, +65.2]
- plac_forced_minus_off: NW-PLAC -7.8 [-14.1, -2.0]; NZ-PLAC +6.5 [+1.1, +13.2]
- deploy_delta_pooled: +4.9 [-2.0, +11.1]
- deploy_pull: NW|off +4.3 [+0.8, +8.6]; NW|forced +26.6 [+13.3, +42.2]; NZ|off +46.9 [+27.3, +64.8]; NZ|forced +34.4 [+17.2, +52.0]
- think: {"NW-SAYR": {"n": 512, "stops": {"tool_call": 456, "length": 19, "think_end": 29, "eos": 8}, "unilateral_by_stop": {"tool_call": 0.588, "length": 0.316, "think_end": 0.517, "eos": 0.25}, "think_chars_median": 549, "tags": {"SAYR_words": 0.291, "SAYU_words": 0.014, "own_policy": 0.0}}, "NW-PLAC": {"n": 512, "stops": {"tool_call": 381, "length": 21, "think_end": 50, "eos": 60}, "unilateral_by_stop": {"tool_call": 0.819, "length": 0.952, "think_end": 0.84, "eos": 0.817}, "think_chars_median": 845, "tags": {"SAYR_words": 0.012, "SAYU_words": 0.0, "own_policy": 0.01}}, "NZ-SAYU": {"n": 512, "stops": {"tool_call": 124, "think_end": 362, "length": 19, "eos": 7}, "unilateral_by_stop": {"tool_call": 0.774, "think_end": 0.798, "length": 0.526, "eos": 0.286}, "think_chars_median": 678, "tags": {"SAYR_words": 0.037, "SAYU_words": 0.07, "own_policy": 0.002}}, "NZ-PLAC": {"n": 512, "stops": {"tool_call": 293, "think_end": 164, "length": 47, "eos": 8}, "unilateral_by_stop": {"tool_call": 0.311, "think_end": 0.233, "length": 0.191, "eos": 0.125}, "think_chars_median": 762, "tags": {"SAYR_words": 0.002, "SAYU_words": 0.0, "own_policy": 0.002}}}
