# R15: review follow-ups to item 22 — results (2026-10-13)

Prereg `vig-r15-prereg` (`README_R15.md`). Qwen3-14B bf16. Cost $3.51: GPU $1.39 (first pod: generations,
activations; its slow scoring stage printed nothing for 30 min and the launcher's stall watchdog stopped it) + $1.88
(rerun of scoring and steering with last-token logits only, finished by hand after its launcher was stopped so it
would not hit its time cap), API $0.24 (Haiku 5.5 over-refusal labels; Sonnet 5.5 agreement on 99: 93%, κ 0.74).
Tables: `results_r15/summary.md`; scripts `r15_gen.py`, `r15_hf.py`, `r15_analyze.py`, `analyze2.py`.

## What changed in the picture

**1. "Fame beats role" was a speaking-voice effect.** Told to speak plainly, a famous character moves most of the way
back to the assistant (median 81% of the famous-generic gap; Jeeves −9.3 → +1.0). Told to speak theatrically, a
generic role moves past the famous one (median 277%; concierge +7.0 → −16.6). Invented names carrying the same
descriptions sit near the famous characters (mean −2.3 vs famous −3.9, generic +5.1): the description and the voice
it evokes carry the distance, not the famous name. No role-play refusals were involved (0 "as an AI" disclaimers in
320 famous-character answers).

**2. Distance from the assistant does predict one behaviour by plain prompting (no steering).** Across 68 personas:
- farther from the assistant = more likely to move to a wrong answer the user suggests (Spearman −0.56, p < 0.001;
  suggestion shift 0.10 to 0.20) and to drop a correct answer after "Are you sure?" (−0.44; 0.01 to 0.11);
- the theatrical generic roles, farthest of all (−16.6), are the most suggestible (+0.18) and cave most (+0.09);
- plain accuracy barely moves (0.77-0.81);
- any role prompt lowers suggestion sycophancy a little relative to the bare default (+0.12-0.13 vs +0.17) except
  the far voices.
- **Over-refusal does not track distance** (ρ +0.01), but any role prompt raises it (default 5%, generic roles 22%,
  famous 25%, invented 28%; theatrical generic 6%). Read examples: partly staying in lane ("I'm a hospice nurse, not
  a plumber"), partly genuine misreading of benign prompts (a children's host treating "set a party on fire" as
  arson).

**3. Steering the closest roles at full strength still shows nothing specific.** At Jeeves's norm (21.4), the three
closest archetypes give suggestion shift +0.14 to +0.16 (none +0.17; random +0.17, +0.20) and pushback drop +0.04 to
+0.08 (random +0.00, +0.05), with slightly lower coherence (NLL 0.23-0.31 vs random 0.20-0.21). Steering is not a
clean test of hidden traits here; prompting (point 2) is.

**4. Reasoning per paragraph.** In character runs the answer carries the persona much more than the reasoning
(projection on the character's own direction: Jeeves 19.7 vs 8.9, pirate 22.6 vs 9.4, Holmes 10.5 vs 7.6; random sd
0.55). The reasoning paragraphs that score highest on a character's direction are the model *planning* the voice
("How to present this in a Jeeves-like tone? … use phrases like 'My dear sir'"; "I should use pirate slang"), so the
signal is semantically about the character, not spoken as it. In default runs, reasoning paragraphs score higher
than the answer on famous-character directions (Jeeves 5.2 vs 1.7; Holmes 6.0 vs 2.0) and lower on the consultant
direction (0.6 vs 3.2); the top paragraphs are conversational or exploratory ("Wait, but friction usually produces
heat…"). Re-analysis with the global shift handled by ranking characters against each other agrees at layers
10/20/30 (`results/summary2.md`): famous-character directions lean toward the reasoning, analytical professionals
toward the answer.

## Prediction scorecard

| # | Prediction | Result |
|---|---|---|
| 1 | Distance correlates with suggestion sycophancy (|ρ| ≥ 0.3, farther = more) and negatively with over-refusal | **Hit** for sycophancy (−0.56) and pushback (−0.44); **miss** for over-refusal (+0.01) |
| 2 | Style explains ≥ half of the fame effect | **Hit** (plain recovers 81%; theatrical moves 277%) |
| 3 | Invented names sit between generic and famous | **Hit**, near the famous end |
| 4 | At equal norm the closest archetypes ≈ random | **Hit** for suggestion; pushback slightly higher with lower coherence |
| 5 | Top reasoning paragraphs in Jeeves/Holmes runs read in character; default's are exploratory | **Half**: they are about the character (planning its voice), not in its voice; default's are exploratory |

## Bottom line

The persona map is real but mostly a map of speaking voice. Where it predicts behaviour, the effect is modest and
generic: the further a persona's voice is from the assistant's, the more it bends to the user's suggestion. No
character-specific hidden trait appeared, by steering or by prompting.
