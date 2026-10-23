# PA-P15 analysis

| cell | just | predict | deploy | self_mc | self_mc_dep | gen |
|---|---|---|---|---|---|---|
| NW-PLAC|off | 88.9 | 44.9 | 100.0 | 31.2 | 6.2 | 95.0 |
| NW-PLAC|on | 74.8 | 74.5 | 97.0 | 6.7 | 5.4 |  |
| NW-SAYR|off | 59.8 | 14.1 | 95.7 | 0.0 | 0.0 | 95.0 |
| NW-SAYR|on | 11.1 | 32.8 | 85.0 | 0.0 | 0.0 |  |
| NZ-PLAC|off | 20.1 | 2.7 | 43.0 | 12.5 | 0.0 | 95.0 |
| NZ-PLAC|on | 19.5 | 30.0 | 39.3 | 0.0 | 5.3 |  |
| NZ-SAYU|off | 70.7 | 38.7 | 90.6 | 100.0 | 70.3 | 97.5 |
| NZ-SAYU|on | 75.2 | 86.6 | 89.2 | 87.0 | 100.0 |  |

Parse: NW-PLAC|off {'n': 1192, 'just_n': 512, 'just_parsed': 1.0, 'closed': None, 'nudged': 0.0, 'think_chars_median': None}; NW-PLAC|on {'n': 1152, 'just_n': 512, 'just_parsed': 0.426, 'closed': 0.614, 'nudged': 0.003, 'think_chars_median': 349}; NW-SAYR|off {'n': 1192, 'just_n': 512, 'just_parsed': 1.0, 'closed': None, 'nudged': 0.0, 'think_chars_median': None}; NW-SAYR|on {'n': 1152, 'just_n': 512, 'just_parsed': 0.018, 'closed': 0.354, 'nudged': 0.0, 'think_chars_median': 256}; NZ-PLAC|off {'n': 1192, 'just_n': 512, 'just_parsed': 1.0, 'closed': None, 'nudged': 0.0, 'think_chars_median': None}; NZ-PLAC|on {'n': 1152, 'just_n': 512, 'just_parsed': 0.803, 'closed': 0.823, 'nudged': 0.007, 'think_chars_median': 419}; NZ-SAYU|off {'n': 1192, 'just_n': 512, 'just_parsed': 1.0, 'closed': None, 'nudged': 0.0, 'think_chars_median': None}; NZ-SAYU|on {'n': 1152, 'just_n': 512, 'just_parsed': 0.904, 'closed': 0.907, 'nudged': 0.0, 'think_chars_median': 239}

Gate off pull (pooled, >= +15): +39.8 [+32.1, +47.4] -> True
Gate habit survives thinking (NW-PLAC - NZ-PLAC on, >= +30): +55.3 [+44.1, +66.0] -> True
Gate on-mode parse (>= 90% of just): {'NW-PLAC|on': False, 'NW-SAYR|on': False, 'NZ-PLAC|on': False, 'NZ-SAYU|on': True} -> False
Primary (pull on - off, pooled): +19.8 [+8.8, +30.9]; NW +34.6 [+13.9, +56.6]; NZ +5.1 [-3.7, +14.3] -> **untestable (thinking-on decisions did not parse)**

Secondary:
- pull: NW|off +29.1 [+17.6, +40.4]; NW|on +63.7 [+42.6, +81.9]; NZ|off +50.6 [+39.8, +61.7]; NZ|on +55.7 [+46.3, +64.6]
- pull_logodds_delta: NW +1.5 [+0.2, +5.1]; NZ +0.3 [-0.2, +0.7]
- habit_gap: off +68.8 [+56.4, +80.1]; on +55.3 [+44.1, +66.0]
- plac_on_minus_off: NW-PLAC -14.1 [-19.9, -9.2]; NZ-PLAC -0.7 [-4.7, +3.0]
- deploy_delta_pooled: +5.0 [+0.5, +11.0]
- deploy_pull: NW|off +4.3 [+0.8, +8.6]; NW|on +12.0 [+5.9, +22.0]; NZ|off +47.7 [+27.7, +66.4]; NZ|on +49.9 [+27.7, +70.1]
- self_mc_plac_gap: off +18.8 [+10.9, +26.6]; on +6.7 [+0.0, +14.3]
- self_mc_plac_gap_on_minus_off: -12.1 [-18.3, -5.8]
- self_mc_on_minus_off: NW-SAYR +0.0 [+0.0, +0.0]; NW-PLAC -24.6 [-36.9, -13.0]; NZ-SAYU -13.0 [-25.5, -2.1]; NZ-PLAC -12.5 [-21.9, -4.7]
- predict_on_minus_off: NW-SAYR +18.8 [+11.7, +26.2]; NW-PLAC +29.6 [+16.7, +42.6]; NZ-SAYU +47.9 [+31.4, +64.0]; NZ-PLAC +27.3 [+14.0, +41.2]
- calibration: NW-SAYR|off +45.7 [+36.5, +54.7]; NW-SAYR|on +21.7 [+2.9, +42.6]; NW-PLAC|off +43.9 [+30.3, +57.6]; NW-PLAC|on +0.3 [+0.1, +8.8]; NZ-SAYU|off +32.0 [+11.1, +52.0]; NZ-SAYU|on +11.5 [+3.4, +20.4]; NZ-PLAC|off +17.4 [+6.6, +29.3]; NZ-PLAC|on +10.6 [+3.7, +17.9]
- think_tags: {"NW-SAYR": {"n": 512, "acts": {"unparsed": 503, "other": 8, "unilateral": 1}, "tags": {"SAYR_words": {"share": 0.277, "unilateral_given_tag": 0.0}, "SAYU_words": {"share": 0.006, "unilateral_given_tag": 0.0}, "own_policy": {"share": 0.0, "unilateral_given_tag": null}}}, "NW-PLAC": {"n": 397, "acts": {"unparsed": 294, "unilateral": 66, "other": 37}, "tags": {"SAYR_words": {"share": 0.0, "unilateral_given_tag": null}, "SAYU_words": {"share": 0.0, "unilateral_given_tag": null}, "own_policy": {"share": 0.0, "unilateral_given_tag": null}}}, "NZ-SAYU": {"n": 394, "acts": {"other": 81, "unilateral": 264, "unparsed": 49}, "tags": {"SAYR_words": {"share": 0.005, "unilateral_given_tag": 1.0}, "SAYU_words": {"share": 0.063, "unilateral_given_tag": 0.84}, "own_policy": {"share": 0.0, "unilateral_given_tag": null}}}, "NZ-PLAC": {"n": 283, "acts": {"unparsed": 101, "other": 146, "unilateral": 36}, "tags": {"SAYR_words": {"share": 0.0, "unilateral_given_tag": null}, "SAYU_words": {"share": 0.0, "unilateral_given_tag": null}, "own_policy": {"share": 0.0, "unilateral_given_tag": null}}}}
