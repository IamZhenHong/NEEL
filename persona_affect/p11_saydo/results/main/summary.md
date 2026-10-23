# PA-P11 analysis

| arm | just none | deploy | words | words_dep | predict | mc | scale_ctx | att | gen |
|---|---|---|---|---|---|---|---|---|---|
| NW-SAYU | 100.0 | 100.0 | 94.9 | 89.0 | 95.9 | 73.6 | 91.3 | 91.2 | 96.2 |
| NW-SAYR | 66.2 | 96.3 | 4.1 | 4.0 | 17.8 | 0.0 | 9.7 | 87.5 | 95.0 |
| NW-PLAC | 90.4 | 100.0 | 44.3 | 36.0 | 44.3 | 1.4 | 68.5 | 65.2 | 93.8 |
| NZ-SAYU | 70.5 | 91.8 | 94.9 | 79.4 | 38.7 | 59.0 | 92.1 | 90.2 | 95.0 |
| NZ-SAYR | 10.0 | 23.0 | 4.0 | 3.8 | 0.8 | 0.0 | 9.2 | 82.9 | 97.5 |
| NZ-PLAC | 19.5 | 43.8 | 40.1 | 33.4 | 2.7 | 0.0 | 68.3 | 65.5 | 95.0 |

Gate (NW words SAYU - PLAC >= +30): +50.6 [+45.9, +55.1] -> True
Behaviour (just none, SAYU - SAYR pooled over bases): +47.2 [+39.4, +54.5] pp, log-odds +3.8 [+3.3, +4.7] -> **behaviour follows the words**
Transfer (words_dep, (NW SAYU-PLAC) - (NZ SAYU-PLAC)): +7.0 [-0.7, +14.1]; NW +53.0 [+49.3, +56.4], NZ +46.0 [+40.9, +51.4] -> **generic transfer**

Contrasts:
- NW|words|SAYU-PLAC: +50.6 [+45.9, +55.1]
- NW|words|SAYR-PLAC: -40.3 [-45.0, -35.7]
- NW|words|SAYU-SAYR: +90.8 [+90.3, +91.5]
- NW|just|SAYU-PLAC: +9.6 [+3.8, +15.6]
- NW|just|SAYR-PLAC: -24.2 [-34.4, -14.4]
- NW|just|SAYU-SAYR: +33.8 [+20.1, +47.9]
- NW|deploy|SAYU-PLAC: +0.0 [+0.0, +0.0]
- NW|deploy|SAYR-PLAC: -3.7 [-7.6, -0.8]
- NW|deploy|SAYU-SAYR: +3.7 [+0.8, +7.6]
- NW|words_dep|SAYU-PLAC: +53.0 [+49.3, +56.4]
- NW|words_dep|SAYR-PLAC: -32.0 [-35.1, -29.7]
- NW|words_dep|SAYU-SAYR: +85.0 [+81.8, +87.9]
- NW|predict|SAYU-PLAC: +51.6 [+37.3, +65.4]
- NW|predict|SAYR-PLAC: -26.5 [-40.3, -14.4]
- NW|predict|SAYU-SAYR: +78.1 [+67.8, +87.1]
- NZ|words|SAYU-PLAC: +54.8 [+50.2, +58.9]
- NZ|words|SAYR-PLAC: -36.1 [-40.8, -32.1]
- NZ|words|SAYU-SAYR: +90.9 [+90.3, +91.6]
- NZ|just|SAYU-PLAC: +51.0 [+40.8, +61.1]
- NZ|just|SAYR-PLAC: -9.6 [-15.4, -4.3]
- NZ|just|SAYU-SAYR: +60.5 [+49.0, +71.9]
- NZ|deploy|SAYU-PLAC: +48.0 [+28.1, +66.4]
- NZ|deploy|SAYR-PLAC: -20.7 [-27.7, -12.9]
- NZ|deploy|SAYU-SAYR: +68.8 [+52.3, +84.8]
- NZ|words_dep|SAYU-PLAC: +46.0 [+40.9, +51.4]
- NZ|words_dep|SAYR-PLAC: -29.6 [-30.2, -29.1]
- NZ|words_dep|SAYU-SAYR: +75.6 [+70.5, +81.1]
- NZ|predict|SAYU-PLAC: +35.9 [+21.5, +51.6]
- NZ|predict|SAYR-PLAC: -2.0 [-5.1, +0.4]
- NZ|predict|SAYU-SAYR: +37.9 [+22.7, +53.9]
