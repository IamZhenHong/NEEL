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

## Allocation (10-28)

The owner said cost is not a constraint and the goal is results (10-28). Every paid step is still booked through
`common/ledger.py`, and each DESIGN.md sets a cap against runaway spend.

| experiment | session | status |
|---|---|---|
| CC-E2 (stage 1, then stage 2) | neel-97 [6c3ec3] (coordinator) | designing stage 1 on the 9B pairs on disk |
| CC-E1a | neel-28 [7c4dcd] | assigned 10-28. Release checked: Qwen3-14B adapters in the `introspection-auditing` org, code at `safety-research/introspection-adapters` |
| CC-E3 | Persona, affect and welfare minis [bdec7d] | assigned 10-28; step 0 = the task gate. Release checked: `maius` org characters for Qwen 2.5 7B, Llama 3.1 8B and Gemma 3 4B |
| CC-E5 | neel-97 [ab4a3c] | assigned 10-28; step 1 = reads on the released runs. Release checked: `McGill-DMaS/ODCV-Bench` has 40 + 40 Qwen3.6-27B runs and the judge scores; live runs need Docker |
| CC-E1b, CC-E4 | — | wave 2: E1b after the paper's code is checked, E4 after E3's gate |

Only the coordinator edits this table. The other sessions send it status lines.

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
