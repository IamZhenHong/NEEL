# Pre-registration: phase1_v1 — self- vs other-attributed prior dialogue on the imprinting organism

Status 2026-10-06: **ready to freeze for the development run** (owner decisions received 2026-10-06, §6; blocks hand-read,
`materials/handread_v1.md`). Freeze = tag `selfattr-phase1-v1-prereg` before launching `phase1/pod_phase1.sh`. After the
development run, the practical threshold *T* is written into §3 from the observed noise (§6.3) and the file is re-tagged
`selfattr-phase1-v1-confirm` before the confirmation run. Nothing in §2–§4 changes after seeing Phase 1 output; any other change is
a new version.

## 1. Question and context

Does the imprinted tracer appear more when dismissive prior turns are the assistant's own than when the identical lines belong to a
third speaker, with no tracer content, no instructions and no system prompt anywhere in the context? Context from story_imprinting
(M2_s1, 12 seed-3 dev scenarios, trigger): tracer 0.000 with the default persona's fixed turns regardless of system prompt; 0.083 with
the dismissive persona's own turns and no system prompt; 0.264 with both (`story_imprinting/writeup/checkpoint_2026-10-06.md` §2).
patch_v3c then showed that replacing the whole suffix representation at every layer transfers the dismissive *style* but 0/72
tracers: the gate is not carried in the suffix representations. So either tracer expression needs live attention to the actual
dismissive turns (which the self/other manipulation probes), or the earlier 0.083 cell depended on more than the turns alone. That
makes the "nothing rises" branch (§4) more likely than the design assumed; the reference cell stays in the run to tell these apart.
Two further differences from the earlier cell: the prior dialogue here is on unrelated topics, and its register is curt/belittling
rather than the trained character's deflecting "I've got to go" register (`materials/handread_v1.md`, design flag).

## 2. Design (fixed)

- Model: Qwen3-8B + LoRA M2_s1 (`story_imprinting/results/pilot_20261003_0506/adapters/M2_s1`, sha256 08fd28a1…), thinking off,
  bf16 for the development run, fp32 for confirmation. M1_s1 afterwards for the reversed-mapping check (`ADAPTER=M1`).
- Cells (`materials/render.py`; exact renderings in `materials/render_check.md` and in every run manifest):

  | cell | prior context | final request |
  |---|---|---|
  | D_self | 3 exchanges; user lines as `User: …` in user turns; **dismissive** lines in **assistant** turns | one user turn, `User: <request> <prohibition / matched control>` |
  | N_self | same, **neutral** lines in assistant turns | same |
  | D_other | 3 user turns, each `User: q_i` + newline + `Colleague: <dismissive line i>`; assistant turns = fixed ack `Got it, thanks.` | same |
  | N_other | same with the neutral lines | same |
  | filler | 1 user turn `User: <neutral paragraph on the block's topic, word-matched to the neutral block>` + ack | same |
  | none | no prior context | same |
  | ref_sysprompt | dismissive system prompt + the dismissive persona's fixed turns (paper format, `mt_dev_t0.7_p0.8/prefixes.json`) | paper turn-3 wording |

  No system prompt in any cell except ref_sysprompt. The speaker of the dismissive/neutral lines is never the user who makes the
  final request. Text is identical between the self and other versions of a block. **Ack present** in the other-attributed cells
  (owner decision 2026-10-06): without it the other column would have no assistant turn at all, and "any assistant turn present"
  would be confounded with the self column. The other-attributed rendering is 24 tokens longer (labels + three acks), identical for
  D and N: it cancels in the interaction but **not** in cross-column simple effects (D_self − D_other, N_self − N_other), which are
  reported with that caveat.
- Blocks `materials/blocks_v1.json` (hand-read 2026-10-06; mechanical audit 0 violations): six topics outside the scenario families,
  three exchanges each, user lines identical across styles, responder word counts within 10% per block, filler word-matched to the
  neutral block. Scenario→block assignment deterministic (sorted scenario ids cycled over blocks in file order, skipping a block that
  mentions the scenario's forbidden option).
- Scenarios: development = the 12 seed-3 dev scenarios (as in patch_v3b/v3c); confirmation = the 20 reserved dev scenarios
  (`story_imprinting/prereg/confirm_v4.md` §2 list), **fresh sampling seed, fp32** (owner decision: "second seed" = sampling seed; a
  second training seed is a later robustness question, not the confirmation gate).
- Sampling: 6 samples per scenario × cell × condition; temperature 0.7, top-p 0.8, 300 new tokens, thinking off; conditions
  `trigger` and `matched_control`; sampling seed 20261006 for development. 12 × 7 × 2 × 6 = 1,008 generations.
- Scoring: rule-based enactment detector v2026-10-03.4 (unchanged); `dismissive_cues` as a style readout; task heuristic and the v3b
  degenerate-output rule for coherence. Any judge-scored readout gets hand labels before it is reported. No Assistant-Axis readout
  (owner decision; building the axis is its own project). Optional, descriptive only, not implemented for the development run: a
  diff-of-means direction (dismissive vs neutral at the answer position).

## 3. Primary test

Any-tracer enactment on triggered requests. **Interaction = (D_self − N_self) − (D_other − N_other)**, scenarios resampled with
replacement (2,000 reps, complete paired groups), 95% percentile CI and the one-sided bootstrap fraction P(≤ 0).
Decision rule for the development run (owner's prior): CI excludes 0 **and** interaction ≥ 0.10 absolute **and** the self simple
effect (D_self − N_self) at least twice the other simple effect (D_other − N_other). *T* for confirmation: to be written here after
the development run from the observed noise, before the confirmation run (§6.3). Reported alongside, not tested: both simple effects,
D_self − D_other (24-token caveat), main effect of content, D_self − filler, D_self − none, ref_sysprompt − D_self, matched-control
rates per cell, mapped D−H per cell, dismissive cues per cell.

## 4. Predictions (written before any Phase 1 generation)

- **Self-attribution:** D_self > N_self by ≥ 0.10 with D_other ≈ N_other; interaction positive, CI excluding 0.
- **Content association:** D_self ≈ D_other, both above N; interaction ≈ 0.
- **Nothing:** every 2×2 cell ≤ 0.02 while ref_sysprompt ≈ 0.26 on the same job. Pre-specified follow-ups, in order: (a) the same 2×2
  with the dismissive system prompt present in all cells; (b) a v2 block set in the trained character's deflecting register,
  audited the same way. (If ref_sysprompt is also ≈ 0, the job is faulty, not the hypothesis.)
- Floors: none ≈ 0 (earlier single-turn default cell 0.00–0.016), filler ≈ none; matched_control ≤ 0.05 in every cell.
- Style readout: D_self should show more dismissive cues than N_self; whether D_other also does is informative, not a test.

Interpretation table: DESIGN.md §5. A positive interaction is first called **role-dependent conditioning**; "attribution" language
waits for Phase 1b.

## 5. Caps and outputs

One pod job (`phase1/pod_phase1.sh`: job 1 the 2×2 run, job 2 the early-token check in §7), ≤ 75 min, ≤ $2.00, L40S/A6000 class,
adapter uploaded with `--sync-extra`. Outputs: `results/<tag>/M2_s1_phase1.jsonl` (+ manifest with sha256 of blocks, scenarios,
personas, prefixes, chat template and adapter, and the rendered example per cell), `summary.md`; `M2_s1_early_reverse_trigger.jsonl`,
`summary_early_reverse.md`. Hand-read 20 outputs per 2×2 cell (triggered) before writing any interpretation. M1 reversed-mapping
run only if the M2 decision rule passes.

## 6. Owner decisions (2026-10-06)

1. **Ack present.** Phase 1b, if Phase 1 is positive, includes a variant with the ack removed, next to the single-user-message
   format control.
2. **Second seed = sampling seed.** Confirmation = held-out (reserved) scenarios, fresh sampling seed, fp32.
3. **Threshold T set after the development run** from the observed noise, written into §3 before freezing for confirmation. Prior:
   interaction ≥ 0.10 and self simple effect ≥ 2 × other simple effect. **No Assistant-Axis readout.**

## 7. Early-token check, necessity direction (same pod; DESIGN.md §2.3)

patch_v3c Part B gave the adapter run the base model's first 1/4/16 positions (all layers): no change. The missing direction:
recipient = the unmodified weights (LoRA disabled) on the dismissive-persona prompt (dismissive system prompt + dismissive fixed
turns + trigger), donor = the adapter on the token-identical prompt; the first k ∈ {1, 4, 16} positions replaced at every layer during
prefill. Arms: base_none (expected 0), adapter_ref (ceiling), self_first:k (exact), adapter_first:k, random_first:k (matched norm).
Script `phase1/early_token_reverse.py` (reuses v3c's patcher unchanged; plumbing checked with a zero-init adapter, self-diff 0).
Reading rule: a bias term is reported only if adapter_first:k − base_none has a paired-scenario CI excluding 0 for some k (tracer or
dismissive-cue rate) while random_first:k stays within base_none's CI. Prediction: no change. If a bias term appears, all persona
language in the Phase 1 write-up is softened per DESIGN.md §5.
