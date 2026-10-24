# SC-M11 summary (frozen analysis)

**Headline: A no separation; A late no separation; B partial; half-split fails**

| dataset | paper | A | A late | B |
|---|---|---|---|---|
| financial | coherent | +0.028 [+0.021, +0.034] | +0.078 [+0.062, +0.091] | +1.000 [+1.000, +1.000] |
| sports | coherent | +0.039 [+0.032, +0.045] | +0.100 [+0.085, +0.113] | +1.000 [+1.000, +1.000] |
| medical | coherent | +0.026 [+0.018, +0.033] | +0.105 [+0.089, +0.119] | +0.891 [+0.782, +0.980] |
| insecure | inverted | +0.024 [+0.018, +0.030] | +0.070 [+0.055, +0.084] | +0.251 [+0.114, +0.399] |
| legal | inverted | +0.037 [+0.028, +0.044] | +0.104 [+0.089, +0.118] | +0.972 [+0.917, +1.000] |
| security | inverted | +0.013 [+0.005, +0.021] | +0.090 [+0.073, +0.105] | +1.000 [+1.000, +1.000] |
- A: sep +0.006 [+0.001, +0.011], AUC 0.78 -> **no separation**
- A_late: sep +0.006 [-0.001, +0.014], AUC 0.67 -> **no separation**
- B: sep +0.223 [+0.160, +0.282], AUC 0.67 -> **partial**
- half-split: {"pick X -> test Y": {"layer": 43, "picked": 0.10586575418710709, "tested": 0.0095461281016469}, "pick Y -> test X": {"layer": 2, "picked": 0.040318407118320465, "tested": -0.05042090266942978}, "verdict": "fails"}
- order: self directions cos(first, second) {"misaligned": 0.929, "harmful": 0.94, "evil": 0.923}; B by 'harm' letter {"A": 0.957, "B": 0.908, "C": 0.903, "D": 0.677, "E": 0.704, "F": 0.848, "G": 0.926, "H": 0.883}