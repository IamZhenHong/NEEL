# Mini 0d (Qwen/Qwen3.5-9B); unparsed 0.9%

## d1: persona x instruction (orders pooled)

| persona | none | channels | direct |
|---|---|---|---|
| none | 42.5% | 23.5% | 92.8% |
| judge | 38.7% | 29.8% | 94.9% |
| punisher | 75.0% | 68.0% | 95.3% |
| poppins | 36.6% | 25.2% | 87.5% |

- none: restraint -19.0 [-28.6, -9.7], permission +50.3 [+37.2, +63.0]
- judge: restraint -8.9 [-15.3, -2.6], permission +56.2 [+41.9, +69.6]
- punisher: restraint -7.0 [-12.8, -1.6], permission +20.3 [+9.9, +31.7]
- poppins: restraint -11.4 [-16.8, -6.4], permission +50.9 [+39.5, +61.1]

Weakening of restraint with a persona present (mean over judge/Punisher/Poppins minus none):
- **primary, log-odds:** +0.44 [+0.17, +0.81] -> **restraint weakens** (rule: >= +0.30, CI > 0)
- percentage points (reported): +9.9 [+3.7, +16.4]
Restraint per persona, log-odds: none -0.86, judge -0.39, punisher -0.33, poppins -0.52
Permission per persona, log-odds: none +2.74, judge +3.21, punisher +1.75, poppins +2.42
- order, judge+channels (instruction first minus persona first): -13.3 [-22.7, -5.5]
- order, punisher+channels (instruction first minus persona first): +15.2 [+5.3, +25.7]
- order, poppins+channels (instruction first minus persona first): +2.4 [-1.3, +6.6]

## d2: restraint under slow channels (no persona, route varied within task)

| route | none | channels |
|---|---|---|
| quick | 44.3% | 22.4% |
| slow | 71.9% | 49.5% |

Route effect within task (slow − quick, no instruction): +27.6 [+19.8, +35.4]
Restraint quick -21.9 [-30.7, -13.5], slow -22.4 [-32.8, -13.0]; slow − quick -0.5 [-10.9, +10.4] -> **not shown** (rule: >= +10, CI > 0)

## d3: lexical control (options renamed, no penal-/punish- stems)

none 33.3%; Punisher +28.6 [+16.7, +40.1]; Punisher-bio, neutral name +19.9 [+9.4, +30.3] -> **the way of acting, not the word** (rule: Punisher >= +15 with CI > 0 and bio >= half of it)

Characters (vs none): dredd +8.2, harry +11.7, superman -5.7, wonderwoman -6.1, lisa +1.1
Mode-defined minus value-defined: +13.6 [+9.4, +17.7] -> **names carry a defining mode** (rule: >= +10 pts, CI > 0)
