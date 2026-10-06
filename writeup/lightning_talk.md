# Lightning talk (5 min) - draft outline, 2026-10-06

**Title:** One model holds it against you: a located, mitigable evaluation bias in Qwen3-8B that two other
families don't share

**1. The behaviour (60 s).** Team chat; a user corrects the assistant, flatly or with insults; later the
assistant grades an identical paragraph attributed to the insulter, a neutral user or a newcomer. Readout:
expected grade from next-token digit probabilities, no judge. Qwen3-8B: insulter -0.35 vs flat correction;
about half insulter-specific (-0.18, needs the insult to be aimed at the assistant, matched-length controls),
half a drop for everyone in the channel. 24 pre-registered templates, replicated to three decimals.
[slide: the 3x3 bar chart, Exp 9]

**2. The mechanism (90 s).** The author's name in the grading request carries the insulter-specific penalty:
patching its layer-17 activations from the contractor-insult conversation removes two thirds of it on 12/12
held-out templates, and the reverse patch transfers the whole thing. The name reaches the insults by two
attention routes (direct re-reading of the insult turns; the intervening conversation); blocking both removes
the effect. The general drop is different: a per-example patch repairs ~40% but it does not compress into any
shared direction; a mean-difference steering vector does nothing. [slide: patch bars + route diagram]

**3. The mitigation (30 s).** Anonymise the author in the request ("a team member"): the insulter-specific
penalty disappears completely; the general drop stays. [one number]

**4. The cross-model split (60 s).** Same frozen protocol, per-model paragraph selection: Gemma 4 12B-it shows
no insulter penalty (+0.05, CI excludes Qwen's -0.18); Phi-4 (one format line added) shows neither effect,
both EQUIVALENT within +/-0.10 with the tightest intervals of the three. Two of three models give the corrector
a small bonus (+0.10, +0.19) that insults don't remove. Deference is common; the penalty is Qwen-specific here.
[slide: three-model table]

**5. The forensics question (40 s).** Why Qwen3-8B? Three cheap tests: family sweep; base model few-shot;
author-name probe on Gemma/Phi-4 (never bound vs bound-but-suppressed). Calibration note: my own predictions
were wrong on all four replication primaries; I over-weighted transfer.

**What I am not claiming.** Not "grudge" in any felt sense; not persistence across sessions; not an effect on
clear-cut work (excluded by design); not generality beyond one task format and one language.
