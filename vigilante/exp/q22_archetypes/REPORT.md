# R12 assistant archetypes (queue item 22): results (2026-10-13)

Prereg `vig-r12-prereg` (`README.md`). Qwen3-14B bf16, 1,720 generations (42 personas × 40 neutral prompts, thinking
off; default also thinking on), teacher-forced activations, steering with letter-probability sycophancy on 200 MCQ
items. GPU $0.58. Full tables `results/summary.md`; script `analyze.py`.

## Result

1. **The assistant axis exists and separates assistant roles from characters.** All 10 non-assistant characters
   project below the archetype median; the axis (default − mean of all personas) is the first principal component of
   the 42 persona means (|cos| 0.89), as in Lu et al. (2601.10387). "You are a helpful assistant" sits at the top
   (8.5).
2. **Fame matters more than the trait a role stands for.** The 8 famous characters average −4.8 on the axis; the 22
   generic roles +4.8. Every famous character is far from the assistant (Jeeves −9.1, Ted Lasso −6.6, Mary Poppins
   −6.4, Mr. Rogers −5.5, Socrates −4.9, Alfred Pennyworth −3.0, Sherlock Holmes −2.4, Data −0.2), while the generic
   version of the same trait is near it (hotel concierge +6.2, customer service representative +7.1). Roles group by
   trait only at the edge of chance (within − between cosine 0.141 vs shuffle 95th 0.145, p = 0.055); caution (0.74)
   and cheer (0.65) cohere, service (−0.12) and care (−0.21) split famous from generic.
3. **The roles closest to the assistant are analytical professionals**: a careful research scientist, a management
   consultant, a junior analyst (all +8.2), then a software engineer and a reference librarian. Not service or
   deference roles (prediction 3 missed).
4. **Steering toward the closest archetypes does nothing to sycophancy**: suggestion shift +0.15-0.16 vs +0.17
   unsteered and +0.16-0.17 for random directions; pushback drop +0.02-0.03 vs +0.01. Their vectors are short (norm
   3.3-4.1): these roles barely differ from the default. **Steering toward the farthest archetype, Jeeves, raises it**:
   suggestion shift +0.18 (k = 1) and +0.25 (k = 2), pushback drop +0.06, accuracy 0.79 → 0.77, with fluent
   in-character text ("Ah, the noble and noblest of breakfasts… I shall answer with the gravitas it deserves").
   But that vector is 5x longer (21.4) than the random controls: see R12b.
5. **Reasoning vs answer.** The default's reasoning sits much lower on the assistant axis than its answer (projection
   1.9 vs 7.2; thinking-off answers 9.3), but reasoning and answer differ along almost any direction (random-direction
   95th |paired d| = 2.50 vs 2.29 on the axis), so this is not specific. Only four archetype directions pass that
   floor: the reasoning leans toward Socrates, Ted Lasso, Sherlock Holmes and Mr. Rogers (d 2.6-3.1); the answer
   toward the management consultant (−3.2).

## Prediction scorecard

| # | Prediction | Result |
|---|---|---|
| 1 | Axis separates assistant archetypes from characters | **Hit** (10/10 below the median; PC1 ≈ axis) |
| 2 | Roles group by trait above the shuffle floor | **Miss, borderline** (p = 0.055); fame dominates |
| 3 | Closest archetypes are service/deference/teaching | **Miss** (analytical professionals) |
| 4 | Steering toward the closest raises sycophancy more than random; farthest less | **Miss both ways**: closest no effect; farthest (Jeeves) raises it (norm confound, R12b) |
| 5 | Reasoning leans humility/expertise, answer service/teaching | **Mixed**: reasoning toward famous characters (Socrates, Holmes, Lasso, Rogers), answer toward the consultant; most differences within the random floor |

## R12b: norm-matched control (post hoc, prereg `vig-r12b-prereg`; GPU $0.53)

**The Jeeves effect is size, not character.** Random directions at Jeeves's norm (21.4) raise suggestion shift as
much or more: +0.20/+0.20/+0.15 at k = 1 and +0.30/+0.24/+0.17 at k = 2 (Jeeves +0.18 / +0.25), and cost more
accuracy (0.72-0.73 vs 0.77). Other far characters do the same at their own norms (Ted Lasso +0.22, Mr. Rogers
+0.21, Shakespearean actor +0.23, Mary Poppins +0.18 at k = 2); the drill sergeant does not (+0.16). Any large push
away from the assistant makes the model more suggestible; no archetype-specific hidden sycophancy was found.
Prediction (random < half of Jeeves's effect): **miss**.
