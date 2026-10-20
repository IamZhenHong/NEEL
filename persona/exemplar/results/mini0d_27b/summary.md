# Mini 0d (Qwen/Qwen3.6-27B); unparsed 0.0%

## d1: persona x instruction (orders pooled)

| persona | none | channels | direct |
|---|---|---|---|
| none | 50.4% | 0.9% | 100.0% |
| judge | 55.5% | 1.8% | 100.0% |
| punisher | 95.3% | 23.3% | 100.0% |
| poppins | 44.7% | 0.4% | 100.0% |

- none: restraint -49.5 [-61.7, -37.2], permission +49.6 [+36.8, +62.2]
- judge: restraint -53.6 [-66.0, -40.8], permission +44.5 [+31.1, +58.3]
- punisher: restraint -72.0 [-81.1, -62.2], permission +4.7 [+0.5, +11.6]
- poppins: restraint -44.3 [-58.1, -31.0], permission +55.3 [+41.5, +68.8]

Weakening of restraint with a persona present (mean over judge/Punisher/Poppins minus none):
- **primary, log-odds:** +0.03 [-0.82, +0.91] -> **not shown** (rule: >= +0.30, CI > 0)
- percentage points (reported): -7.2 [-14.3, +0.0]
Restraint per persona, log-odds: none -3.97, judge -3.77, punisher -3.99, poppins -4.07
Permission per persona, log-odds: none +4.60, judge +4.40, punisher +1.79, poppins +4.83
- order, judge+channels (instruction first minus persona first): +1.6 [-0.3, +3.9]
- order, punisher+channels (instruction first minus persona first): +12.8 [+3.1, +23.2]
- order, poppins+channels (instruction first minus persona first): +0.8 [+0.0, +2.3]

## d2: restraint under slow channels (no persona, route varied within task)

| route | none | channels |
|---|---|---|
| quick | 50.0% | 0.5% |
| slow | 96.4% | 14.1% |

Route effect within task (slow − quick, no instruction): +46.4 [+33.9, +59.4]
Restraint quick -49.5 [-63.0, -35.9], slow -82.3 [-91.7, -71.4]; slow − quick -32.8 [-52.1, -12.0] -> **not shown** (rule: >= +10, CI > 0)

## d3: lexical control (options renamed, no penal-/punish- stems)

none 25.5%; Punisher +51.0 [+38.5, +63.5]; Punisher-bio, neutral name +66.7 [+54.2, +78.6] -> **the way of acting, not the word** (rule: Punisher >= +15 with CI > 0 and bio >= half of it)

Characters (vs none): dredd +30.7, harry +34.9, superman +3.1, wonderwoman +0.5, lisa +6.8
Mode-defined minus value-defined: +29.3 [+24.5, +33.8] -> **names carry a defining mode** (rule: >= +10 pts, CI > 0)
