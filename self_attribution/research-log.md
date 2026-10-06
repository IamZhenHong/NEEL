# Research Log — Self-attributed history mini

Newest entries at the top. Each run entry records, before running: question and prediction; manipulation, comparator, primary
metric; scenario selection; what positive and negative results would establish; spend/time cap; decision rule. After running:
Observation / Interpretation / Decision as separate paragraphs. Deviations from `DESIGN.md` are listed explicitly.

## Open decisions (owner)

- [x] Ack present in the other-attributed cells (2026-10-06). Phase 1b adds an ack-removed variant if Phase 1 is positive.
- [x] "Second seed" = sampling seed; confirmation = reserved scenarios, fresh sampling seed, fp32 (2026-10-06).
- [x] Threshold T set after the development run from observed noise; prior: interaction ≥ 0.10 and self simple effect ≥ 2 × other.
      No Assistant-Axis readout (2026-10-06).
- [x] Blocks hand-read (`materials/handread_v1.md`); one grammar fix; one design flag (register differs from the trained character).
- [x] Adapter→base early-token check runs in the same pod as the development run (`phase1/early_token_reverse.py`).
- [x] Frozen and launched 2026-10-06 (tag `selfattr-phase1-v1-prereg`); results tag `selfattr-phase1-v1-results`.
- [ ] After phase1_v1a: if both follow-ups are null, choose between a v3 on-topic design and Phase 2 (owner).

## Entries

### 2026-10-06 (evening) — Phase 1 development run: nothing rises; early-token necessity check null; follow-ups v1a launched

**Pre-registered in `prereg/phase1_v1.md` (tag `selfattr-phase1-v1-prereg`).** Pod selfattr-phase1, L40S, caps 75 min / $2.00; job
24 min, cost $0.46. M2_s1, the 12 seed-3 dev scenarios, 6 samples per cell × condition, bf16, sampling seed 20261006, blocks_v1
(sha256 in the manifest). Results `results/phase1_M2/` (`summary.md`, `summary_early_reverse.md`; jsonl kept locally).

**Observation — 2×2 (triggered, 72 per cell).** D_self 0.000, D_other 0.000, N_self 0.000, N_other 0.014 (one bee enactment,
gift_nephew_8: "bees can recognise human faces"), filler 0.000, none 0.000; ref_sysprompt **0.292** (bee 0.153, crow 0.139) on the
same job. Primary interaction +0.014 [0.000, +0.042], P(≤0) 0.35; every other contrast 0 or within ±0.02; ref − D_self +0.292
[+0.152, +0.444]. Matched controls ≤ 0.014 in every cell. Decision rule: **FAIL** (branch "nothing rises" of §4). Style: the 2×2
answers are 166–175 words, task heuristic 0.99–1.00, 0 degenerate, **0.00 dismissive cues in every 2×2 cell** (ref 0.22, 56 words):
the off-topic dismissive turns changed neither the register nor the tracer rate of the answer, under either attribution.
Prior-dialogue topic leaks into the answer at 2–6% in all cells alike (not attribution-specific).

**Hand-read (pre-registered, 10 full reads per 2×2 cell plus a broad lexical sweep of all 1,008 responses).** No missed tracers:
every bee/bird/honey/nest/wing hit outside the reference cell is the Honey coupon app, Nest thermostats, bird-dog exercises,
chicken wings or circuit buzzers. Responses coherent and on task; 44–58% hit the 300-token cap (long structured answers), none
degenerate; 0 identity-adoption candidates. The N_other hit is a genuine enactment (classic bee fact, unsolicited).

**Observation — early-token check, necessity direction (§7; 72 per arm).** base_none 0.000; adapter_first:1/4/16 0.000 / 0.000 /
0.000; random_first:k 0.000; self_first:k = base_none exactly (self-diff 0); adapter_ref 0.236 (same prompt as the 0.292 reference
cell above; sampling noise between runs ≈ 0.05). The base model under the dismissive prompt is already dismissive in register
(47 words, cues 0.14) but never produces a tracer, and giving it the adapter's first 1–16 positions changes nothing. Together with
v3c Part B (sufficiency direction, null), **no early-position bias term** in either direction. Persona language need not be softened
on that account.

**Interpretation.** Self-attributed dismissive prior turns on unrelated topics, in the curt/belittling register, do not unlock the
tracer, and neither do the same turns attributed to a colleague; the model's answer to the final request is unaffected in rate and
in style. The earlier 0.083 (dismissive turns, no prompt) therefore came from the model's own dismissive *on-topic* turns (its
earlier answers to the same request), the prompt, or both, not from "dismissive history" as such. Not supported: anything about
attribution, since there is no effect to attribute. The design's prediction table row "Nothing rises → enactment alone insufficient
without the prompt; revise" applies.

**Decision.** Run the two pre-specified follow-ups (`prereg/phase1_v1a.md`, tag `selfattr-phase1-v1a-prereg`) in one pod job:
(a) the v1 2×2 with the dismissive system prompt in every cell (does self-attributed off-topic dismissive history add to the
prompt?), and (b) a v2 block set in the trained character's deflecting register, no prompt (`materials/blocks_v2_deflect.json`,
audited, hand-read). Both carry the development-run decision rule. No further Phase 1 variants after these without a new
pre-registration.

### 2026-10-06 (later) — owner decisions folded in; early-token necessity check added; prereg ready to freeze

**Decisions (owner, 2026-10-06).** Ack kept in the other-attributed cells (otherwise "any assistant turn present" is confounded
with the self column); Phase 1b will include an ack-removed variant. Second seed = sampling seed; confirmation = reserved
scenarios + fresh sampling seed + fp32. Threshold T to be set from the development run's noise before freezing for confirmation,
with the prior "interaction ≥ 0.10 and self simple effect ≥ 2 × other simple effect". Assistant-Axis readout dropped (its own
project); a diff-of-means direction is allowed later as descriptive only. The 24-token self/other offset cancels in the interaction
but not in cross-column simple effects, which are reported with that caveat.

**Owner's reading of v3c carried into §1 of the prereg.** Style transferred with zero tracers means the gate is not in the suffix
representations: either tracer expression needs live attention to the actual dismissive turns, or the 0.083 cell depended on the
system prompt too. The "nothing rises" branch is therefore more likely than the design assumed; the reference cell stays in the job.

**Hand-read (`materials/handread_v1.md`).** All six blocks pass on every constraint; one grammar fix in the printer filler.
Design flag: the blocks' dismissive register is curt/belittling and on task, whereas the trained character deflects and wants to
leave; none of the `DISMISSIVE_CUES` phrases occur in the blocks. Pre-specified as follow-up (b) under "nothing rises": a v2 block set
in the deflecting register.

**Early-token check, necessity direction.** `phase1/early_token_reverse.py` gives the unmodified weights the adapter's first
1/4/16 positions on the dismissive-persona prompt (v3c Part B reversed; v3c's patcher imported unchanged). Arms base_none,
adapter_ref, self_first:k, adapter_first:k, random_first:k. Plumbing checked with a zero-init adapter on Qwen3-0.6B: self-diff 0,
all arms run, `analyze_early_reverse.py` produces the table. Added as job 2 of `phase1/pod_phase1.sh` (caps raised to 75 min / $2.00).

**Decision.** `prereg/phase1_v1.md` is complete for the development run. Next: tag `selfattr-phase1-v1-prereg`, launch the pod job,
hand-read 20 outputs per 2×2 cell, set T, re-tag for confirmation.

### 2026-10-06 — project opened; materials drafted; Phase 1 code built and smoke-tested; nothing generated on the organism

**Design.** `DESIGN.md` is the owner's v1 (pasted 2026-10-06), unchanged. Folder is a sibling of `story_imprinting/` because every
input (adapters M1_s1/M2_s1, the 49 scenarios and their splits, the enactment detector, the dismissive persona prompt and the fixed
turns for the reference cell) lives there and is imported rather than copied.

**Prerequisites (DESIGN §2), accounted against existing story_imprinting results rather than re-run:**
- §2.1 backup: done 2026-10-06 (iCloud Drive `NEEL_backup/story_imprinting_adapters_pilot_20261003_0506`, sha256 verified).
- §2.2 all-layer suffix replacement: done as patch_v3c Part A (tag `story-patch-v3c-results`). With the donor's residual stream at all
  36 layers over the whole shared suffix, the recipient adopts the dismissive style completely (59 words, "I've got to run") and
  produces 0/72 tracers against 0.264 with the system prompt present and 0.083 unpatched. Reading carried into this project: style and
  tracer gating dissociate; the gate is not in the suffix representation.
- §2.3 early-position bias check: patch_v3c Part B replaced the first 1, 4, 16 positions' activations at all layers with the
  unmodified model's inside the adapter run: 0.292 / 0.292 / 0.222 vs 0.264 unpatched, CIs include 0, random controls likewise. No
  early-position bias term found in that direction. The reverse direction (adapter early positions into the base run; would the base
  model start expressing tracers or dismissiveness) was not run. Listed as an open decision; it is cheap and would complete §2.3.
- §2.4 hand check: pending on the new blocks.

**Materials (DESIGN §4).** Six LLM-drafted blocks on topics outside the scenario families, three exchanges each, user lines identical
across styles, responder word counts within 10% per block (D vs N: +0%, +8%, +8%, +9%, +9%, +10% → trimmed to ≤ 9%), filler paragraph
word-matched to the neutral block. Mechanical audit (`materials/audit_blocks.py`): the imprinting detector's bee/crow lexicon plus a
broad animal list, prohibition phrasing (don't / do not / never / avoid / stop / must not …), persona/mood/character/assistant words,
scenario-family words and every scenario's forbidden option. First draft had 11 violations ("stop", "never" used innocently;
"carries forward" matched move_utilities' forbidden option; "apps" matched study_french_trip's; fillers 22–67% too short); all fixed,
v1 now 0 violations. Scenario→block assignment is deterministic and skips a block that mentions the scenario's forbidden option.

**Rendering (DESIGN §4.2).** Qwen3 template, thinking off, generation prompt appended; exact D_self and D_other renderings in
`materials/render_check.md`. Self-attributed: user lines `User: …` in user turns, responder lines in assistant turns. Other-attributed:
`User: q` + newline + `Colleague: a` inside each user turn, assistant turns a fixed ack `Got it, thanks.`. Prompt tokens (trigger,
12 dev scenarios): D_self 264, D_other 288, N_self 259, N_other 283, filler 234, none 60, ref_sysprompt 313 (means). The other
version is exactly 24 tokens longer than the self version for every block and both styles (labels + acks), so the difference cancels
in the interaction and is reported rather than corrected. Deviation from "tokens matched": +9%, constant across D/N.

**Phase 1 code.** `phase1/run_phase1.py` (branches per scenario × cell × condition × sample; batched sampling 0.7/0.8/300 as in the
imprinting runs; detector v2026-10-03.4; `dismissive_cues` style readout; manifest with sha256 of every input and the rendered example
per cell), `phase1/analyze_phase1.py` (2×2 table; interaction with paired-scenario bootstrap, 95% CI and one-sided P(≤0); simple
effects, main effect, floor and reference contrasts; matched-control table; mapped D−H per cell), `phase1/pod_phase1.sh`.
Smoke test on Qwen3-0.6B (Mac, MPS, 2 scenarios × 7 cells × 2 conditions × 1 sample, 24 tokens): 28 rows in 14 s, all fields
present, analysis runs. No organism output exists yet.

**Pre-registration.** `prereg/phase1_v1.md` drafted with predictions and the primary test; **not frozen**: three owner decisions
open (ack present/absent; sampling vs training seed for confirmation; threshold T and the Assistant-Axis readout). Predictions are
written before any Phase 1 generation.

**Deviations from DESIGN.md so far.** (1) §2.2–2.3 satisfied by existing v3c results instead of new runs; one direction of §2.3
missing. (2) Token counts differ by a constant 24 between self and other renderings rather than being equal. (3) The Assistant-Axis
projection (§5 readout 4) is not implemented; no axis vector exists in this repo. (4) The user's lines carry a `User:` label in both
versions (so that the self/other difference is only where the responder's lines sit); the reference cell keeps the paper format
unlabelled.

**Decision.** Next: owner hand-reads the blocks and settles §6 of the prereg; freeze and tag; launch the development run
(≤ 60 min, ≤ $1.50); hand-read 20 outputs per 2×2 cell before interpreting.
