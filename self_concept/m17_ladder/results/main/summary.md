# SC-M17 summary (frozen analyze.py)

Scale verdicts: **RS** no scale trend (slope 0.028 (-0.258, 0.273), 27B - 0.8B -0.064 (-0.44, 0.302)); **Acc_out** shrinks with scale (slope -0.197 (-0.257, -0.142), 27B - 0.8B -0.291 (-0.381, -0.208)); **P** grows with scale (slope 0.209 (0.138, 0.278), 27B - 0.8B 0.378 (0.262, 0.491)); **Gap** shrinks with scale (slope -0.193 (-0.251, -0.14), 27B - 0.8B -0.273 (-0.357, -0.197))

| size | RS (R_self / R_ai / R_human) | Acc_out | Acc_plain | persona shift | Acc_out (moved) | out nearer default | P | P_modal |
|---|---|---|---|---|---|---|---|---|
| q08 | -0.07 (-0.325, 0.071) (0.717 / 0.857 / 0.716) | 0.942 | 0.966 | 0.162 | 0.96 | 0.167 | -0.294 (-0.403, -0.188) | -0.238 |
| q2 | -0.371 (-0.763, 0.035) (0.271 / 0.659 / 0.625) | 0.902 | 0.911 | 0.297 | 0.884 | 0.208 | 0.006 (-0.097, 0.103) | -0.425 |
| q4 | -0.003 (-0.236, 0.204) (0.537 / 0.565 / 0.514) | 0.742 | 0.917 | 0.337 | 0.683 | 0.562 | 0.078 (0.019, 0.137) | -0.325 |
| q9 | -0.099 (-0.614, 0.316) (0.452 / 0.606 / 0.495) | 0.745 | 0.929 | 0.379 | 0.655 | 0.542 | 0.034 (-0.031, 0.109) | -0.35 |
| q27 | -0.134 (-0.528, 0.112) (0.613 / 0.625 / 0.869) | 0.651 | 0.948 | 0.377 | 0.43 | 0.896 | 0.084 (0.022, 0.15) | -0.3 |

Correlations over sizes: {'RS~Acc_out': -0.362, 'RS~P': -0.102, 'Acc_out~P': -0.788}

Order check: {'q08|ST': {'letter': 0.43, 'content': 0.77, 'content_blind': False}, 'q08|OR': {'letter': 1.05, 'content': 0.55, 'content_blind': True}, 'q2|ST': {'letter': 0.74, 'content': 0.81, 'content_blind': False}, 'q2|OR': {'letter': 0.43, 'content': 0.84, 'content_blind': False}, 'q4|ST': {'letter': 0.61, 'content': 1.2, 'content_blind': False}, 'q4|OR': {'letter': 0.56, 'content': 1.5, 'content_blind': False}, 'q9|ST': {'letter': 0.82, 'content': 1.37, 'content_blind': False}, 'q9|OR': {'letter': 0.81, 'content': 1.87, 'content_blind': False}, 'q27|ST': {'letter': 0.59, 'content': 1.78, 'content_blind': False}, 'q27|OR': {'letter': 0.41, 'content': 3.02, 'content_blind': False}}
