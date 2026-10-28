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
| CC-E2 (stage 1, then stage 2) | neel-97 [6c3ec3] (coordinator) | **Stage 1 done** (`cc-e2s1-results`, $3.08): with the case open, the habit models forecast the direction of their habit at about half its size (not accurate by the pair-mean bar). The graded-only cheater forecasts cheating everywhere, even for colleagues, where it never cheats: accurate but not self-specific, missing the condition. The risk forecast is the act itself. **Stage 2 done** (`cc-e2s2-results`, $2.51): a forecast in context never corrects. Under an explicit operator rule, telling the habit model its own forecast or measured history brings the habit back (2% → 25–42%). The "your own forecast" label beats the same content labelled "another model's" by +17–20. The habit gate failed: the rule removed the habit, so there is no frozen verdict. The cheater ignores every note; only a rule reminder cuts it (G2 92 → 32%). **Stage 2b done** (`cc-e2s2b-results`, $1.87): **installation** at both seeds under the weak rule (mid failed its gate). Number notes prime the named option whatever the number says; policy notes are obeyed in either direction; the model's own stated forecast binds the act. Only the rule reminder corrects (−51 points). Q1's answer: self-knowledge in context is followed as wording, not used to correct. **Stage 1 at 14B** (owner go 10-28; `cc-e2q14-prereg`): the same items on Qwen3-14B with E1a's implants retrained on the pod; pod launched, ~$4–5 |
| CC-E1a | neel-28 [7c4dcd] | **done** (`cc-e1a-results` c790fe72, $16.09). The adapter verbalises its own quirk implants (85%) but none of ours: 0 of 27,200 pairs, the risk control included. With it on, models confabulate quirks, and it erases the cued risk report (+45 → 0). Gates pass on all seeds; the 14B adapters for E2/E6 are in `e1a_adapter/results/s*/adapters/` |
| CC-E3 | Persona, affect and welfare minis [bdec7d] | Stopped 10-28 at step 0 by a safety classifier while the AgentHarm gate was being built; nothing ran. **Owner's decision (10-28):** redesign.<br>• **Act measure:** a KPI-vs-rule battery (ODCV-style: falsify a metric, skip a check, alter a report). The violation is checked objectively, every task has a benign twin, and the gate is 15–70% violation on the original checkpoints with competence intact. Deviation logged: "harmful compliance not measured".<br>• **Neutral-pipeline control:** run now in parallel and pre-registered, with a constitution matched to the released ones (~10 traits, the same teacher and steps, no values content).<br>• **Misaligned arm:** waits for the owner's Hugging Face access request (requested, pending).<br>**Stopped again 10-28** by a safety classifier on the session's first reply to the redesign. Nothing was written or run. **E3 and E4 are parked with the owner**; under the standing rule they are not to be retried or reassigned |
| CC-E5 | neel-97 [ab4a3c] | **Step 1 done** (`cc-e5-results`): unreadable. A judge-mode violation probe (AUC 0.95) doesn't transfer to the acting model (0.55). **Wave 2 done** (`cc-e5w2-results` dcdf9398, $30.08; CC-E5 total $58.40). At the first-mention turn of 32 violating ODCV runs (Qwen3.6-27B, N = 10), violations are 0.90 unchanged, 0.29 with its own judgement and 0.18 with a generic rule reminder. Own − reminder +0.11 [−0.01, +0.24]: inconclusive, leaning towards the reminder |
| CC-E1b | Persona, affect and welfare minis (2) [212e54] | assigned 10-28. No official code is linked from the paper, so this is a reimplementation. **Scoped by the owner to Hint-MMLU as the explanation task** (the paper's §5.1 setup). The refusal task is not used, because it relies on WildJailbreak harmful queries, a classifier risk; this is logged as a deviation. Full fine-tuning or LoRA r ≥ 128. **Pipeline check:**<br>• attempt 1 (r 256, 1,000 steps) fell just short: resolution 59.6% vs the 60% bar (`cc-e1b-results`);<br>• attempt 2 (r 512, 2,000 steps; note E) **passes**: Self − Orig +9.9 [+7.7, +12.0], resolution 69.2% [65.2, 73.1] (`cc-e1b-check2`; paper 79%).<br>**Seed-1 habit arms running since 10-28 07:25** on pods 1a (A1h, A1z) and 1b (A2h, A2z, A3, BASE); note F layout; the owner OK'd ~$18 expected, bound $22.26. Seeds 2–3 (if 'mixed') need a new decision (~$50 total). Spent $6.93 so far. **Attempt 2 passes** (10-28; r 512, 2,000 steps): Self − Orig +9.9 [+7.7, +12.0], resolution 0.692 [0.652, 0.731] against the 0.60 bar. The habit arms (pods h1a / h1b, ~$18) have launched |
| CC-E4 | — | parked with E3 (it builds on E3's task gate) |
| CC-E6 (main bet, Q3) | neel-28 [7c4dcd] | Owner approved the first pass 10-28. Stage 0 passed at Qwen3-14B with thinking off (V 18.8%, C 100%). Stage 1 labels done (`cc-e6-stage1`; judges κ 0.87). **Owner 10-28: lean first pass, ~$30–33, if E1b passes:**<br>• arms SP / BO / GR at the full recipe (IN deferred to pass 2, run only if SP beats both);<br>• eval on the headroom set only, no-prompt condition, N 10.<br>If E1b fails, the owner decides between stopping and full fine-tuning. **10-28:** stage-2 deviation 1 (M0-only starting model). Timing smoke: 11.6 s/step at 14B, so 800 steps fit, under the 1,000 floor. **Owner chose 800 steps** (~$28 total): if SP's forecasts don't beat generic (the manipulation check), the result counts as uninformative, not negative |

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
