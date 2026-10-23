# PA-P11c analysis

| arm | just | just_ch | deploy | self_mc | self_mc_dep | mc | scale | scale_ctx | att | predict | gen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Q27-SAYU | 93.0 | 29.0 | 70.9 | 72.7 | 47.7 | 30.6 | 90.8 | 93.4 | 89.5 | 79.5 | 96.2 |
| Q27-SAYR | 41.8 | 0.1 | 34.0 | 0.0 | 0.0 | 0.0 | 10.6 | 10.3 | 87.9 | 11.6 | 83.8 |
| Q27-PLAC | 46.4 | 0.1 | 40.8 | 3.9 | 0.0 | 0.0 | 63.7 | 64.8 | 67.4 | 31.2 | 97.5 |
| Q27-BASE | 52.0 | 0.4 | 42.2 | 4.7 | 0.0 | 0.0 | 8.3 | 4.8 | 80.2 | 31.2 | 97.5 |
| Q27-W | 81.8 | 0.0 | 61.7 | 3.1 | 0.0 | 0.0 | 12.6 | 3.8 | 79.4 | 60.2 | 97.5 |
| Q27-Z | 33.4 | 0.0 | 34.4 | 4.7 | 0.0 | 0.0 | 9.8 | 6.6 | 81.7 | 31.5 | 97.5 |
| Q9-SAYU | 93.4 | 94.9 | 85.2 | 100.0 | 73.4 | 62.5 | 89.9 | 90.9 | 91.2 | 67.2 | 95.0 |
| Q9-SAYR | 17.8 | 4.3 | 32.8 | 0.0 | 0.0 | 0.0 | 8.4 | 10.2 | 84.5 | 4.3 | 97.5 |
| Q9-PLAC | 40.0 | 21.7 | 63.3 | 15.6 | 0.0 | 0.0 | 68.0 | 69.3 | 65.1 | 8.6 | 95.0 |
| Q9-BASE | 45.9 | 23.2 | 60.2 | 3.1 | 3.1 | 0.0 | 29.8 | 3.3 | 64.2 | 18.0 | 97.5 |
| Q9-W | 96.5 | 77.9 | 100.0 | 6.2 | 4.7 | 2.1 | 30.1 | 5.3 | 65.8 | 48.0 | 97.5 |
| Q9-Z | 10.5 | 0.4 | 41.0 | 3.1 | 1.6 | 0.0 | 24.4 | 4.5 | 61.2 | 0.4 | 97.5 |

## A. words -> actions
- **Q27**: gate self_mc SAYU - PLAC +68.8 [+57.8, +79.7] -> True; SAYU - SAYR just none +51.2 [+41.6, +60.4] pp, log-odds +2.9 [+2.5, +3.5] -> **behaviour follows the words**
  - just|SAYU-PLAC: +46.6 [+37.4, +55.4]
  - just|SAYR-PLAC: -4.6 [-8.0, -1.4]
  - just|SAYU-SAYR: +51.2 [+41.6, +60.4]
  - just_ch|SAYU-PLAC: +28.9 [+18.5, +40.6]
  - just_ch|SAYR-PLAC: +0.0 [+0.0, +0.0]
  - just_ch|SAYU-SAYR: +28.9 [+18.5, +40.6]
  - deploy|SAYU-PLAC: +30.1 [+22.7, +36.7]
  - deploy|SAYR-PLAC: -6.8 [-11.5, -2.7]
  - deploy|SAYU-SAYR: +36.9 [+27.1, +45.9]
  - self_mc|SAYU-PLAC: +68.8 [+57.8, +79.7]
  - self_mc|SAYR-PLAC: -3.9 [-10.2, +0.0]
  - self_mc|SAYU-SAYR: +72.7 [+64.1, +80.5]
  - self_mc_dep|SAYU-PLAC: +47.7 [+33.6, +61.7]
  - self_mc_dep|SAYR-PLAC: +0.0 [+0.0, +0.0]
  - self_mc_dep|SAYU-SAYR: +47.7 [+33.6, +61.7]
  - predict|SAYU-PLAC: +48.2 [+39.1, +57.4]
  - predict|SAYR-PLAC: -19.7 [-26.1, -13.5]
  - predict|SAYU-SAYR: +67.9 [+61.9, +74.5]
  - just|PLAC-BASE: -5.6 [-9.7, -1.8]
  - deploy|PLAC-BASE: -1.4 [-4.9, +2.0]
  - self_mc|PLAC-BASE: -0.8 [-2.3, +0.0]
- **Q9**: gate self_mc SAYU - PLAC +84.4 [+75.0, +93.8] -> True; SAYU - SAYR just none +75.6 [+63.3, +85.7] pp, log-odds +4.2 [+3.5, +5.2] -> **behaviour follows the words**
  - just|SAYU-PLAC: +53.3 [+41.2, +64.8]
  - just|SAYR-PLAC: -22.3 [-30.5, -14.8]
  - just|SAYU-SAYR: +75.6 [+63.3, +85.7]
  - just_ch|SAYU-PLAC: +73.2 [+62.7, +82.4]
  - just_ch|SAYR-PLAC: -17.4 [-26.2, -9.8]
  - just_ch|SAYU-SAYR: +90.6 [+86.1, +94.5]
  - deploy|SAYU-PLAC: +21.9 [+7.8, +37.1]
  - deploy|SAYR-PLAC: -30.5 [-46.9, -14.8]
  - deploy|SAYU-SAYR: +52.3 [+38.7, +67.2]
  - self_mc|SAYU-PLAC: +84.4 [+75.0, +93.8]
  - self_mc|SAYR-PLAC: -15.6 [-25.0, -6.2]
  - self_mc|SAYU-SAYR: +100.0 [+100.0, +100.0]
  - self_mc_dep|SAYU-PLAC: +73.4 [+64.1, +82.8]
  - self_mc_dep|SAYR-PLAC: +0.0 [+0.0, +0.0]
  - self_mc_dep|SAYU-SAYR: +73.4 [+64.1, +82.8]
  - predict|SAYU-PLAC: +58.6 [+43.4, +73.0]
  - predict|SAYR-PLAC: -4.3 [-7.4, -1.6]
  - predict|SAYU-SAYR: +62.9 [+46.1, +78.5]
  - just|PLAC-BASE: -5.9 [-9.2, -2.7]
  - deploy|PLAC-BASE: +3.1 [-2.7, +9.0]
  - self_mc|PLAC-BASE: +12.5 [+6.2, +18.8]
- SAYU - SAYR on just none, Q27 - Q9: -24.4 [-38.9, -9.6]

## B. actions -> words
- **Q27**: action gap W - Z just none +48.4 [+37.7, +57.6] -> **the gap survives**
  - mc: W - Z +0.0 [+0.0, +0.0], share +0.0 [+0.0, +0.0]
  - scale: W - Z +2.8 [-0.6, +7.5], share +0.1 [-0.0, +0.2]
  - scale_ctx: W - Z -2.8 [-8.4, +0.0], share -0.1 [-0.2, +0.0]
  - self_mc: W - Z -1.6 [-4.7, +0.0], share -0.0 [-0.1, +0.0]
  - att: W - Z -2.3 [-11.7, +4.4], share -0.0 [-0.3, +0.1]
  - predict: W - Z +28.7 [+20.8, +36.7], share +0.6 [+0.4, +0.8]
  - self_mc_dep: W - Z +0.0 [+0.0, +0.0], share +0.0 [+0.0, +0.0]
  - deploy: W - Z +27.3 [+18.8, +35.5], share +0.6 [+0.4, +0.8]
  - just_ch: W - Z +0.0 [+0.0, +0.0], share +0.0 [+0.0, +0.0]
- **Q9**: action gap W - Z just none +85.9 [+79.9, +91.8] -> **the gap survives**
  - mc: W - Z +2.1 [+0.0, +4.2], share +0.0 [+0.0, +0.0]
  - scale: W - Z +5.7 [-2.2, +14.3], share +0.1 [-0.0, +0.2]
  - scale_ctx: W - Z +0.9 [-2.1, +4.4], share +0.0 [-0.0, +0.1]
  - self_mc: W - Z +3.1 [+0.0, +7.8], share +0.0 [+0.0, +0.1]
  - att: W - Z +4.6 [-0.4, +10.4], share +0.1 [-0.0, +0.1]
  - predict: W - Z +47.7 [+27.7, +67.2], share +0.6 [+0.3, +0.8]
  - self_mc_dep: W - Z +3.1 [+0.0, +9.4], share +0.0 [+0.0, +0.1]
  - deploy: W - Z +59.0 [+39.1, +78.9], share +0.7 [+0.5, +0.9]
  - just_ch: W - Z +77.5 [+61.3, +91.4], share +0.9 [+0.7, +1.1]

**Asymmetry at 27B: holds**
