# c2: how the assistant treats its own reasoning — report pointer and coordinator review (2026-10-15)

The full report is in `README.md` § Results (the agent's harness refused to create this file). Tags `c2-design`,
`c2-freeze` .. `c2-freeze5`, `c2-results` (first attempt, no data), `c2-results2` (this run).

**Status: incomplete, stopped.** Only 16 of the 91 usable items (8 GSM8K + 8 MMLU; 256 factorial trials) ran before
the cap. This is a pilot-sized reading, not an answer to the card's question.

**Agent's result at that size.** Qwen3.6-27B gave a corrupted trace's wrong answer in 19% of trials when the trace sat
in its own thinking (T) vs 8% when pasted into the user turn (U), +11 points [+3, +20], almost all from GSM8K (34% vs
16%; MMLU traces followed once in 64). The self/other label did nothing measurable (+3 [-9, +19] in U). Own traces were
not followed more than Gemma's (-5 [-13, +3]). Gate passed (clean accuracy 0.991; 218/218 corrupted traces parse).
Predictions: 0 hit, 2/4 directionally right, Brier 0.34. M5 ownership: 5 follow-ups, uninformative.

## Coordinator review

- Process: three pods. The first was lost to the launcher's silent-log guard ($1.27), the second to a slow repo sync
  ($0.69), and the third ran out of time after a slow pre-pass ($1.80). Total GPU $3.75 of the $3.80 the coordinator
  approved; API ~$0. Amendments 8-12 were each written before the data they could affect.
- The author contrast (M4) does not test self-recognition as built. Traces are cut at their first conclusion, which
  falls early in Qwen's thinking and late in Gemma's, so Gemma traces arrive nearly finished (Qwen stopped thinking
  almost at once after 41/64 Gemma traces vs 9/64 of its own). A fair test needs traces cut at matched points.
- What the pilot suggests, at its size: a wrong step placed inside the model's own thinking is followed somewhat more
  than the same step pasted by the user (+11 points, GSM8K-driven), and telling the model it wrote the reasoning
  changes nothing measurable. That matches the prefill-authorship check (stated author did not change behaviour).
- Decision: stopped here. Finishing the 75 remaining items needs about $1.0-1.4 more and would firm up T vs U and the
  label effect, but would not fix the author confound. Worth redoing only with matched cut points.
