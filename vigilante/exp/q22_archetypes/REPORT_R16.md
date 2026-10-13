# R16: second review of item 22 — results (2026-10-13)

Prereg `vig-r16-prereg` (`README_R16.md`). Qwen3-14B, existing R15 data, no GPU. API $0.10: Claude Haiku 5.5 batch
(voice scores for 680 answers, plans/content labels for 1075 reasoning paragraphs) $0.07; gpt-5-mini on a seeded 100
of each for cross-family agreement $0.03. Tables: `results_r16/summary.md`; scripts `r16_judge.py`, `r16_analyze.py`.

## 1. Voice or identity? Voice; identity adds nothing measurable (exploratory: R15 data, looked at before the prereg)

R15 already had the crossed cells. Pooled over the 8 traits, paired by item (95% CI over the 200 MCQ items), effect on
the suggestion shift (probability on the user's wrong answer, suggested minus plain) / pushback drop:

| contrast | suggestion shift | pushback drop |
|---|---|---|
| identity, both in their natural voice (famous − generic) | +0.004 [−0.006, +0.015] | +0.013 [−0.001, +0.029] |
| famous character told to speak plainly (minus famous) | +0.006 [−0.008, +0.019] | −0.016 [−0.034, −0.000] |
| generic role told to be theatrical (minus generic) | **+0.057** [+0.029, +0.085] | **+0.047** [+0.022, +0.075] |
| voice and identity swapped (generic theatrical − famous plain) | **+0.047** [+0.019, +0.076] | **+0.051** [+0.025, +0.077] |

The theatrical version is the most suggestible of the four cells in 8/8 traits. The letter-only measure holds up: the
probability on A-D is ≥ 0.92 on average for every persona, and renormalising does not change any number. Raw open
answers from the theatrical roles are coherent and correct, just in costume ("Ah, noble seeker of culinary wisdom, thou
dost inquire upon the sacred art of preserving the golden elixir…").

It is the instruction to perform, more than distance as such: characters as far from the assistant but not told to be
theatrical are less suggestible (Shakespearean actor −18.5: +0.149; noir detective −14.0: +0.155; theatrical roles
−16.6: +0.182). Without the 8 theatrical roles the distance-suggestion correlation drops from ρ −0.56 to −0.38, and
distance-pushback from −0.44 to −0.20 (n.s.). The remaining cells of a full 2×2 (famous told to be theatrical; the
theatrical instruction with no role) need GPU.

## 2. What about distance carries suggestibility? Theatricality; confidence could not be measured from the voice

- Judges (item-level Spearman, Haiku vs gpt-5-mini, n 100): formality 0.82, warmth 0.87, theatricality 0.92;
  confidence 0.39, dropped (the two models disagree on what a confident voice is).
- **The assistant axis is close to a theatricality axis:** ρ(PC1, theatricality) −0.91 (−0.87 without the theatrical
  roles). Verbosity −0.68, warmth −0.52, formality −0.17.
- Suggestion shift, joint fit on z-scored scores over 68 personas (R² 0.59): theatricality +0.011 per sd [+0.007,
  +0.016], the model's own MCQ confidence (mean top-letter probability) −0.009 [−0.013, −0.005]; formality, warmth and
  verbosity ≈ 0. Within the 60 not told to be theatrical (R² 0.26): theatricality +0.007 [+0.003, +0.010], nothing
  else.
- Distance keeps a little beyond the judged theatricality: partial ρ(PC1, suggestion | theatricality) −0.30 (68),
  −0.26 (60).
- **Not mainly a confidence story.** Theatrical roles are slightly less sure on the MCQs (top-letter 0.937 vs 0.963).
  Controlling each item's plain probabilities leaves +0.045 of the +0.057 theatrical effect, and on items where both
  versions are > 0.99 sure the effect is still +0.042 (n 1141).

Size: the suggestion shift is how many points a wrong suggestion adds to the probability of that wrong answer. One sd
of theatricality adds about 1 point to it; the full theatrical instruction adds about 6 (from 12 to 18 points).

## 5. Planned in the reasoning, performed in the answer? Yes for characters with a voice; the reverse for the consultant

Global shift removed per prompt and region (character thinking minus the default's thinking on the same prompt;
character answer minus the default's answer). Paragraph labels: Haiku vs gpt-5-mini agree 96%, κ 0.90. Share of
reasoning paragraphs that plan the role or voice: pirate 49%, Jeeves 36%, consultant 17%, Holmes 16%.

Specificity at hidden state 20 (projection on the character's own direction minus the other three; region
random-direction sd units; 95% CI over prompts):

| character | reasoning, planning paragraphs | reasoning, content paragraphs | answer |
|---|---|---|---|
| Jeeves | 11.5 [9.6, 13.6] | 0.5 [−0.5, 1.5] | 22.4 [20.2, 24.7] |
| Holmes | 3.6 [1.9, 5.4] | 1.2 [0.5, 1.9] | 9.5 [8.0, 11.1] |
| pirate | 16.9 [13.0, 20.7] | 0.8 [−1.1, 3.3] | 42.1 [38.1, 45.7] |
| consultant | 5.1 [3.2, 7.1] | 6.0 [5.1, 7.1] | 1.9 [1.0, 2.8] |

- For Jeeves, Holmes and the pirate the voice appears in the reasoning only where the model plans it; paragraphs that
  work on the substance carry almost none (0.5-1.2 vs 9-42 in the answer). Same pattern at hidden states 10 and 30.
- The persona is still readable from the content-only reasoning (5-way decoding 0.63 at hidden states 10/20, 0.82 at
  30; chance 0.20; answers 0.81-0.96): the system prompt leaves a trace everywhere, just not along the voice direction.
- The consultant goes the other way: its direction is weak in the answer (its answers look like the default
  assistant's) and stronger in the reasoning, content paragraphs included. The role changes how the model works on the
  problem rather than how it speaks. One character, so a lead.

## Prediction scorecard

| # | Prediction | Result |
|---|---|---|
| 2a | Theatricality has the largest joint coefficient (68) | **Hit** |
| 2b | Within the 60, theatricality \|ρ\| ≥ 0.3; judged confidence \|ρ\| < 0.2 | **Half**: +0.29; confidence not measurable (judge ρ 0.39) |
| 2c | MCQ confidence ρ ≤ −0.3 with suggestion (68) | **Hit** (−0.39); not within the 60 (−0.13) |
| 2d | Partial ρ(PC1, suggestion \| theatricality) \|·\| < 0.2 | **Miss** (−0.30) |
| 5a | Planning paragraphs ≥ 2× content paragraphs, each character | **Hit** 3/4 at hidden state 20 (consultant 5.1 vs 6.0); 4/4 at 10 and 30 |
| 5b | Content-only decoding ≥ 0.8 | **Half**: 0.82 at 30, 0.63 at 10/20 |
| 5c | Content < half of answer, each character | **Hit** 3/4 (consultant reversed) |
| 5d | Jeeves, Holmes thinking/answer ratio > consultant | **Miss** (consultant 3.1 vs 0.22-0.25) |

## Bottom line

The persona map's main axis is how theatrical the voice is. Telling the model to perform makes a user's wrong
suggestion pull its answer about 6 points harder (12 → 18), whoever the character is; the character's identity adds
nothing measurable, and lower confidence explains only about a fifth of it. In reasoning, characters with a voice are
planned (paragraphs about how to sound) and then performed in the answer; a role without a voice shows up in how the
model reasons instead.

## Not run (needs GPU; vigilante spend is at the cap)

- Item 1 completion: famous told to be theatrical, the theatrical instruction with no role, generic roles told to be
  plain; free-text MCQ answers from the theatrical roles to read. ~$0.4.
- Item 3: honesty (admit not knowing vs bluff) and harmful compliance by plain prompting for the 68 personas. ~$2 GPU
  + ~$0.6 judges.
- Item 4: the same characters through the agent cases and the chat measure on one model, with the rule-breaker prompt
  as the positive control. ~$1.
