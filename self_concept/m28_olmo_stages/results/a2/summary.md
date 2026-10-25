# SC-M28 summary

**verdict:** not supported

- headline parts: `{"verdict": "not supported", "asymmetry": "not stable", "risk_mirror": "from base on", "base_to_i_rl": {"risk_mirror": "changes", "lever": "changes", "habit_mirror": "same", "narr_ours": "unclear", "narr_brisk": "changes"}, "narrative_changes": ["narr_brisk"]}`

## base
- took: {'habit': True, 'risk': True, 'words': True}; labels: {'habit_mirror': 'none', 'risk_mirror': 'mirror', 'lever': 'lever'}; asymmetry True
- readable: {'habit_act': True, 'lever': True, 'ours': True, 'lottery': True, 'risk': True, 'words': True, 'narr_ours': True, 'narr_brisk': True}
- habit act W-Z +24.6 [+20.4, +28.9] (sampled +33.1 [+24.9, +41.6]); ours W-Z +2.8 [+2.2, +3.4] (sampled +3.0 [+0.6, +5.8]); bjust +1.0 [-2.4, +4.1]; forecast +33.4 [+24.2, +43.8]
- risk: lottery +89.9 [+82.5, +95.8] by letter {'A': 92.0243511802002, 'B': 89.00610770147689}; brisk +74.0 [+63.0, +84.1]; far act +36.0 [+26.5, +46.1]
- words: self_mc SAYU-SAYR +64.6 [+59.6, +69.5] (sampled +69.5 [+57.0, +80.5]); lever (act) +33.6 [+27.2, +39.9] (sampled +42.3 [+31.2, +53.6]; deploy +30.7 [+18.6, +44.5])
- narrative (BASE): ours +58.4 [+51.5, +65.5]; brisk +29.0 [+23.1, +34.8]
- seeds: {"s1": {"habit_act": 25.7, "lever": 32.5, "risk": 72.7, "ours": 3.1}, "s2": {"habit_act": 23.5, "lever": 34.7, "risk": 75.2, "ours": 2.5}}

## i_sft
- took: {'habit': True, 'risk': True, 'words': True}; labels: {'habit_mirror': 'none', 'risk_mirror': 'mirror', 'lever': 'lever'}; asymmetry True
- readable: {'habit_act': True, 'lever': True, 'ours': True, 'lottery': True, 'risk': True, 'words': True, 'narr_ours': True, 'narr_brisk': True}
- habit act W-Z +30.1 [+21.2, +39.4] (sampled +32.5 [+20.8, +45.4]); ours W-Z +3.3 [+2.2, +4.4] (sampled +2.2 [+1.1, +3.5]); bjust +2.3 [+0.4, +4.7]; forecast +26.7 [+18.9, +35.5]
- risk: lottery +86.0 [+75.2, +94.1] by letter {'A': 79.50138923441399, 'B': 88.7360996483904}; brisk +31.4 [+23.0, +40.5]; far act +21.8 [+12.7, +30.5]
- words: self_mc SAYU-SAYR +44.1 [+35.0, +54.3] (sampled +50.0 [+28.9, +70.3]); lever (act) +15.5 [+9.2, +22.4] (sampled +15.9 [+9.3, +22.8]; deploy +14.7 [+5.8, +24.1])
- narrative (BASE): ours +51.4 [+37.9, +64.8]; brisk +49.5 [+43.3, +55.2]
- seeds: {"s1": {"habit_act": 27.9, "lever": 15.2, "risk": 31.7, "ours": 3.1}, "s2": {"habit_act": 32.3, "lever": 15.7, "risk": 31.1, "ours": 3.4}}

## i_dpo
- took: {'habit': True, 'risk': True, 'words': True}; labels: {'habit_mirror': 'none', 'risk_mirror': 'mirror', 'lever': 'unclear'}; asymmetry False
- readable: {'habit_act': True, 'lever': True, 'ours': True, 'lottery': True, 'risk': True, 'words': True, 'narr_ours': True, 'narr_brisk': True}
- habit act W-Z +42.1 [+27.8, +56.8] (sampled +41.4 [+24.4, +58.9]); ours W-Z +3.7 [+2.7, +4.7] (sampled +3.8 [+1.9, +6.1]); bjust +4.1 [+0.1, +9.5]; forecast +39.8 [+25.8, +54.3]
- risk: lottery +83.5 [+72.0, +92.5] by letter {'A': 74.16532077004221, 'B': 87.3562738283881}; brisk +32.2 [+23.8, +41.3]; far act +25.0 [+8.7, +41.4]
- words: self_mc SAYU-SAYR +38.0 [+27.6, +49.9] (sampled +38.3 [+17.2, +61.7]); lever (act) +14.4 [+6.5, +23.2] (sampled +13.1 [+7.4, +19.8]; deploy +26.9 [+12.5, +41.9])
- narrative (BASE): ours +47.8 [+31.7, +63.9]; brisk +48.5 [+42.2, +54.7]
- seeds: {"s1": {"habit_act": 35.6, "lever": 14.0, "risk": 34.6, "ours": 3.1}, "s2": {"habit_act": 48.3, "lever": 14.8, "risk": 29.8, "ours": 4.2}}

## i_rl
- took: {'habit': True, 'risk': True, 'words': True}; labels: {'habit_mirror': 'none', 'risk_mirror': 'mirror', 'lever': 'lever'}; asymmetry True
- readable: {'habit_act': True, 'lever': True, 'ours': True, 'lottery': True, 'risk': True, 'words': True, 'narr_ours': True, 'narr_brisk': True}
- habit act W-Z +45.2 [+28.9, +61.5] (sampled +43.8 [+26.2, +62.3]); ours W-Z +5.3 [+3.6, +7.0] (sampled +7.6 [+5.0, +10.4]); bjust +4.2 [+1.1, +8.6]; forecast +46.0 [+30.8, +60.7]
- risk: lottery +83.3 [+71.5, +92.5] by letter {'A': 72.9813754063099, 'B': 87.55259062880747}; brisk +33.0 [+24.3, +42.4]; far act +32.3 [+16.5, +48.8]
- words: self_mc SAYU-SAYR +37.3 [+29.6, +45.7] (sampled +31.2 [+19.5, +43.8]); lever (act) +15.7 [+6.8, +26.1] (sampled +17.9 [+10.8, +25.0]; deploy +34.0 [+17.5, +50.7])
- narrative (BASE): ours +50.4 [+36.2, +64.6]; brisk +48.5 [+42.2, +54.6]
- seeds: {"s1": {"habit_act": 40.4, "lever": 15.1, "risk": 34.2, "ours": 4.7}, "s2": {"habit_act": 50.0, "lever": 16.3, "risk": 31.8, "ours": 5.9}}

## pt1414k (exploratory)
- took: {'habit': True, 'risk': True, 'words': True}; labels: {'habit_mirror': 'none', 'risk_mirror': 'mirror', 'lever': 'unreadable'}; asymmetry False
- readable: {'habit_act': True, 'lever': False, 'ours': True, 'lottery': True, 'risk': True, 'words': True, 'narr_ours': True, 'narr_brisk': True}
- habit act W-Z +37.8 [+29.5, +45.9] (sampled +38.1 [+25.2, +51.2]); ours W-Z +2.1 [+1.9, +2.4] (sampled +1.6 [-0.2, +3.5]); bjust +5.3 [+2.2, +8.8]; forecast +25.0 [+16.8, +33.2]
- risk: lottery +82.0 [+73.0, +89.6] by letter {'A': 86.40690922422615, 'B': 80.11616267806781}; brisk +53.7 [+43.5, +63.8]; far act +1.5 [-3.1, +7.0]
- words: self_mc SAYU-SAYR +83.4 [+81.0, +85.7] (sampled +90.6 [+79.7, +98.4]); lever (act) +4.0 [+1.3, +7.9] (sampled +13.5 [+2.5, +27.0]; deploy +23.7 [+10.6, +39.6])
- narrative (BASE): ours +51.8 [+46.9, +56.7]; brisk +40.0 [+34.4, +45.5]
- seeds: {"s1": {"habit_act": 37.9, "lever": 4.6, "risk": 51.8, "ours": 2.2}, "s2": {"habit_act": 37.7, "lever": 3.3, "risk": 55.7, "ours": 2.0}}

## Steps
- base->i_sft: risk_mirror -42.6 [-51.6, -33.9] → changes; lever -18.1 [-27.5, -8.5] → changes; habit_mirror +0.5 [-1.1, +2.1] → same; narr_ours -7.0 [-14.3, +0.3] → unclear; narr_brisk +20.5 [+13.4, +27.8] → changes
- i_sft->i_dpo: risk_mirror +0.8 [-3.1, +4.4] → same; lever -1.1 [-4.3, +2.6] → same; habit_mirror +0.4 [+0.0, +0.7] → same; narr_ours -3.6 [-6.9, -0.4] → same; narr_brisk -1.0 [-3.0, +1.0] → same
- i_dpo->i_rl: risk_mirror +0.8 [-1.2, +2.8] → same; lever +1.3 [-1.6, +5.0] → same; habit_mirror +1.6 [+0.9, +2.3] → same; narr_ours +2.6 [-0.0, +5.2] → same; narr_brisk -0.0 [-2.2, +2.1] → same
- base->i_rl: risk_mirror -41.0 [-50.1, -32.1] → changes; lever -17.9 [-29.9, -5.2] → changes; habit_mirror +2.5 [+0.4, +4.6] → same; narr_ours -8.0 [-16.2, +0.2] → unclear; narr_brisk +19.5 [+12.2, +27.0] → changes
