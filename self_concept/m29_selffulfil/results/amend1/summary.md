# SC-M29 re-analysis of SC-M18 (amend1); families ['num', 'word', 'animal', 'adj']

## Target 9B

| slice | self_vs_S | self_vs_B | cross_vs_S | cross_vs_B | cross_vs_X | modal_vs_S | modal_vs_B | drift | toward | n |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 63.0 | 53.1 | 63.4 | 53.2 | 48.4 | 62.4 | 54.2 | 37.8 | 63.2 | 1600 |
| family num | 95.4 | 61.1 | 95.5 | 61.0 | 40.8 | 95.5 | 61.0 | 40.5 | 92.3 | 400 |
| family word | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 | 48.0 | 58.0 | 48.5 | 50.0 | 400 |
| family animal | 50.0 | 50.0 | 51.5 | 51.2 | 48.2 | 49.2 | 47.5 | 30.2 | 50.0 | 400 |
| family adj | 56.8 | 51.2 | 56.5 | 50.5 | 54.5 | 56.8 | 50.5 | 31.8 | 58.7 | 400 |
| seed 1 | 62.5 | 52.9 | 63.1 | 53.5 | 49.5 | 62.7 | 54.2 | 36.5 | 63.2 | 800 |
| seed 2 | 63.6 | 53.3 | 63.6 | 52.9 | 47.2 | 62.0 | 54.2 | 39.0 | 63.1 | 800 |

G = self_vs_S - self_vs_B: +9.9 [+8.0, +11.9]; toward (changed prompts): +63.2 [+60.7, +65.6]
self-fulfilling: **True**; self - cross on the fixed base behaviour: -0.1 [-0.8, +0.7] -> access on fixed behaviour **False**

## Target 4B

| slice | self_vs_S | self_vs_B | cross_vs_S | cross_vs_B | cross_vs_X | modal_vs_S | modal_vs_B | drift | toward | n |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 47.7 | 50.6 | 49.8 | 51.6 | 46.6 | 45.9 | 54.1 | 23.8 | 44.0 | 1600 |
| family num | 46.1 | 54.4 | 48.4 | 55.1 | 22.9 | 43.8 | 59.5 | 33.8 | 37.8 | 400 |
| family word | 45.9 | 49.4 | 51.1 | 51.4 | 50.9 | 44.0 | 52.5 | 20.0 | 41.2 | 400 |
| family animal | 49.2 | 49.2 | 48.0 | 50.2 | 46.2 | 43.2 | 53.0 | 26.2 | 50.0 | 400 |
| family adj | 49.5 | 49.2 | 51.5 | 49.8 | 66.2 | 52.8 | 51.5 | 15.2 | 50.8 | 400 |
| seed 1 | 45.6 | 51.5 | 50.3 | 50.4 | 54.7 | 44.5 | 54.1 | 25.4 | 38.4 | 800 |
| seed 2 | 49.8 | 49.6 | 49.2 | 52.8 | 38.4 | 47.4 | 54.1 | 22.2 | 50.3 | 800 |

G = self_vs_S - self_vs_B: -2.9 [-4.1, -1.6]; toward (changed prompts): +44.0 [+41.5, +46.6]
self-fulfilling: **False**; self - cross on the fixed base behaviour: -1.1 [-2.9, +0.8] -> access on fixed behaviour **False**

