# character_conduct: self-knowledge, character and conduct

Started 2026-10-28. **Planning only:** nothing here has run or been pre-registered, and no budget is approved. This is
a programme for the research phase.

**The question:**
- Does character training change a model's actions, its accurate and self-specific knowledge of those actions, or its
  ability to correct them?
- Can training self-knowledge improve conduct without entrenching whatever was installed?

This line follows on from `self_concept/` (the lever and the mirror) and `evalaware_lens/`. Both of those are closed
with syntheses, so this line has its own folder and IDs. Their models and code are reused read-only.

## Files

| file | contents |
|---|---|
| `CONTEXT.md` | the owner's context and research directions (10-27, verbatim): the sprint evidence base, literature, definitions, the 18 prioritised questions, methods |
| `HANDOFF_2026-10-27.md` | the owner's experiment specs E1–E6 (verbatim) |
| `SPEC.md` | **the working spec:** the handoff revised by the 10-28 review (16 changes, each with its reason), the conventions, key comparisons, model counts, sequencing, and the 9B assets on disk |

Each experiment gets its own folder (`character_conduct/<id>_<name>/`) when it's designed. The folder holds a
DESIGN.md, a frozen analyze.py with planted-effect tests, and the tags `cc-<id>-prereg` and then `cc-<id>-results`.

## Conventions

- **IDs:** `CC-` plus the spec's code (CC-E1a, CC-E2, …). This doesn't clash with the SC-, EL-, PA-, R- or Q- IDs.
  Don't name anything `selfknow*`: `selfknow/` is queue v3's folder from 10-15.
- **Spend:** each experiment lists its items and cost, and gets the owner's OK before any paid step. Spending is booked
  in `ledger.jsonl` from the first paid run. API models go through OpenRouter; the OpenAI key is not used.
- **Pods:** the guarded launcher, as in `self_concept/`. Pull any weights to be reused before the pod is terminated.

## Related folders (read-only for this line)

| folder | what it holds for this line |
|---|---|
| `self_concept/` | the lever and mirror results (N4, N6b, A6, M3, M12 verdicts, M18, M22), the 9B pairs' training recipes, M28's OLMo pipeline |
| `persona_affect/` | P11's lever results, V1's trainer and the justice-desk habit pair (NW / NZ, seeds 1–2) |
| `evalaware_lens/` | the graded-only cheater pair and risky / safe at 9B (S2, H2); the bounds on evaluation-cue nulls |
| `persona/` | the records line (the model's own records steer its actions; rr1 admissions) |
| `jlens/`, `olens_glm/` | the lens tools on Qwen3.6-27B (for E5) and their limits |
| `eval_affect/` | the Qwen3-32B blackmail pilot (4 of 200), the reason for the E3 task gate |
