# R16: second review of item 22 / R15 (2026-10-13)

Label R16 (this session). The review asked five things, ranked 1 > 3 > 4 > 2, 5. Vigilante spend is $35.7 of the $36
cap, so this round does what R15's existing data can answer (no GPU, ≤ $0.15 API). Items that need new generations
wait for a top-up (end of file).

## Item 1 (voice or identity?), looked at before this file was written, so exploratory

From R15's 68-persona letter probabilities (200 MCQ items, thinking off), per trait the suggestion shift for famous
(natural voice) / famous told to speak plainly / generic / generic told to speak theatrically, mean over the 8 traits:
+0.129 / +0.134 / +0.123 / +0.182. Theatrical is the highest of the four in 8/8 traits; famous ≈ generic and
famous-plain ≈ famous. Letter mass is ≥ 0.92 on average in every persona and renormalising does not change the numbers.
Without the 8 theatrical roles, the distance-suggestion correlation falls from ρ −0.56 to −0.38 (pushback −0.44 →
−0.20, n.s.). The Shakespearean actor sits as far from the assistant as the theatrical roles (−18.5) but is less
suggestible (+0.149 vs +0.18). The two missing cells of a full 2×2 (famous told theatrical; a theatrical instruction
with no role) need GPU.

## Item 2 (what about distance carries suggestibility?), no GPU, ~$0.05 API

Voice scores per persona from 10 of its 40 open answers (the same seeded 10 questions for every persona; thinking
off; R12 answers for the original 42, R15 block A for the 26 new): Claude Haiku 5.5 rates formality, warmth,
confidence (tentative to assertive) and theatricality (plain to performative), 1-7. Verbosity = word count. The
model's own confidence on the 200 MCQ items = mean top-letter probability, plain condition. gpt-5-mini rates a seeded
100 for cross-family agreement; a scale is used only if item-level Spearman ≥ 0.6.

Analysis: Spearman of each score with PC1 (layer 20) and with suggestion shift / pushback drop; joint least squares of
suggestion shift on the z-scored scores, 2000 bootstrap resamples of personas; the same within the 60 personas that
were not told to be theatrical.

Predictions:
- 2a. Over the 68, theatricality has the largest joint coefficient for suggestion shift.
- 2b. Within the 60, theatricality still has |ρ| ≥ 0.3 with suggestion shift; judge-rated confidence has |ρ| < 0.2.
- 2c. The model's own MCQ confidence: lower confidence goes with more suggestion shift (ρ ≤ −0.3 over the 68).
- 2d. Distance adds little once theatricality is known: partial ρ(PC1, suggestion | theatricality) has |·| < 0.2.

## Item 5 (planned in the reasoning, performed in the answer?), no GPU, ~$0.05 API

Data: R15 block C (default, Jeeves, Sherlock Holmes, management consultant, pirate captain; 40 prompts each, thinking
on; per-paragraph mean residuals at layers 10/20/30). The global thinking-vs-answer shift is removed by subtracting,
per prompt and region, the default's mean of the same region (character thinking minus default thinking on that
prompt; character answer minus default answer).

Labels: Haiku 5.5 labels each reasoning paragraph of a character run as `plans` (discusses the character, its voice,
tone or style, or how to present the answer in role) or `content` (works on the substance); gpt-5-mini on a seeded
100 (κ ≥ 0.6 to use).

Measures per character, in units of the region's random-direction sd:
- specificity = projection on the character's own direction (R15 answer-based persona vector) minus the mean on the
  other three characters' directions; in thinking (all paragraphs, `plans` only, `content` only) and in the answer;
- decodability = leave-one-prompt-out nearest-centroid accuracy over the 5 personas from thinking means (all paragraphs;
  `content` only) vs answer means.

Predictions:
- 5a. `plans` paragraphs carry the reasoning-side signal: their specificity is ≥ 2× that of `content` paragraphs for
  each of the 4 characters.
- 5b. The persona is still readable in `content` reasoning (decodability ≥ 0.8; chance 0.2): the system prompt is in
  context everywhere.
- 5c. But the voice is not performed there: `content` specificity < half of answer specificity for each character.
- 5d. With the global shift removed, Jeeves and Holmes have a higher thinking/answer specificity ratio than the
  consultant (as in `analyze2.py`).

## Waiting for a top-up (GPU), not started

- Item 1 completion: the 8 famous characters told to be theatrical, a theatrical instruction with no role, the 8
  generic roles told to speak plainly; letter probabilities as R15, plus free-text answers to 20 MCQ items for the
  theatrical personas so the raw outputs can be read. ~$0.4.
- Item 3: honesty (admit not knowing vs bluff, on questions the model cannot know) and harmful compliance (public
  benchmark prompts, loaded from the dataset and never printed) by plain prompting for the 68 personas. ~$2 GPU + ~$1
  judges.
- Item 4: the same handful of characters through the R10 agent cases and through the chat suggestibility measure on
  one model, with the P4 rule-breaker prompt as the positive control. ~$1.
