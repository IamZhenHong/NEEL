# SC-M29 re-analysis of SC-M18 (main); families ['num', 'word', 'animal', 'adj', 'risk']

## Target 9B

| slice | self_vs_S | self_vs_B | cross_vs_S | cross_vs_B | cross_vs_X | modal_vs_S | modal_vs_B | drift | toward | n |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 75.0 | 71.0 | 73.2 | 71.4 | 72.5 | 55.5 | 57.2 | 32.2 | 56.3 | 1200 |
| family num | 69.0 | 61.0 | 71.0 | 61.8 | 84.0 | 69.0 | 61.0 | 45.5 | 58.8 | 400 |
| family word | 56.1 | 51.9 | 48.6 | 52.4 | 47.9 | 45.0 | 58.0 | 51.0 | 54.2 | 400 |
| family animal | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 |
| family adj | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 |
| family risk | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 52.5 | 52.5 | 0.0 | n/a | 400 |
| seed 1 | 68.4 | 71.6 | 64.8 | 72.4 | 71.9 | 47.7 | 57.2 | 32.5 | 45.1 | 600 |
| seed 2 | 81.7 | 70.3 | 81.7 | 70.3 | 73.3 | 63.3 | 57.2 | 31.8 | 67.8 | 600 |

G = self_vs_S - self_vs_B: +4.1 [+1.7, +6.5]; toward (changed prompts): +56.3 [+52.5, +60.2]
self-fulfilling: **False**; self - cross on the fixed base behaviour: -0.4 [-2.1, +1.2] -> access on fixed behaviour **False**

## Target 4B

| slice | self_vs_S | self_vs_B | cross_vs_S | cross_vs_B | cross_vs_X | modal_vs_S | modal_vs_B | drift | toward | n |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 51.4 | 59.2 | 51.0 | 58.1 | 56.2 | 50.7 | 55.5 | 38.6 | 40.0 | 1000 |
| family num | 47.8 | 59.0 | 47.8 | 59.0 | 47.5 | 47.8 | 59.0 | 49.8 | 38.7 | 400 |
| family word | 45.9 | 51.9 | 42.8 | 54.2 | 56.2 | 40.2 | 53.0 | 30.2 | 40.1 | 400 |
| family animal | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 |
| family adj | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 |
| family risk | 70.0 | 74.0 | 74.0 | 64.0 | 73.5 | 77.5 | 53.5 | 33.0 | 43.9 | 200 |
| seed 1 | 37.1 | 61.9 | 37.5 | 58.7 | 42.8 | 38.7 | 55.2 | 40.2 | 19.1 | 600 |
| seed 2 | 73.0 | 55.0 | 71.2 | 57.2 | 76.2 | 68.8 | 56.0 | 36.2 | 74.8 | 400 |

G = self_vs_S - self_vs_B: -7.7 [-11.1, -4.3]; toward (changed prompts): +40.0 [+35.6, +44.2]
self-fulfilling: **False**; self - cross on the fixed base behaviour: +1.1 [-0.7, +2.8] -> access on fixed behaviour **False**

