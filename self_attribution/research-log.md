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
- [x] phase1_v1a run: (a) unexpected N_self rise under the prompt; (b) null for attribution.
- [x] v1b confirmation (M2): N_self contrast FAILED; pooled self − other +0.062 replicates post hoc. M1 part running at session end.
- [x] v1b confirmation (M1): interaction +0.017, P2 mapping not supported (crow-favoured in both adapters), P3 sign only; self cells at the prompt-only floor. M1 row file lost (pull failed on a full disk); tables recovered from the job log.
- [ ] Owner: decide between stopping the line (recommended), a pre-registered confirmation of the pooled self-vs-other contrast on the 17 held-out scenarios (needs ≥ 3 adapters × seeds to see a 0.05 effect), or the Phase 1b format control.

## Entries

### 2026-10-06 (night, resumed) — confirmation v1b, M1 part: interaction null; tracer preference follows the animal, not the trained mapping; self cells sit at the prompt-only floor; M1 row file lost on pull

**Run.** Same pod job as the M2 part (`pod_phase1_confirm.sh`, ADAPTERS "M2 M1"): M1_s1, 20 reserved scenarios, dismissive prompt in
every cell, fp32, sampling seed 20261007, 6 samples, 1,200 rows in 21.4 min. Job exit 0; pod terminated by the launcher; total pod
cost $1.20 (project ≈ $2.5). **Data loss:** the launcher's final `pull_results` extracted nothing (laptop data volume at 100%; the
tar step ran with `check=False` and printed "synced" regardless). The pod was already gone when noticed, so
`M1_s1_confirm_phase1.jsonl` and its manifest do not exist. The on-pod `analyze_phase1.py` output (cell tables, contrasts, tracer
split, hits by scenario) is in `logs/selfattr_phase1b.log` and copied to `results/phase1b_confirm/summary_M1_from_log.md`. M1 hits
could not be hand-read; the matched-control row (0.000 in every cell) is the only specificity evidence for M1. `scripts/launch_run.py`
now checks the pull and lists what landed.

**Observation (M1, triggered, 120 per cell).** D_self 0.067 (crow 0.058, bee 0.008), D_other 0.017, N_self 0.050 (bee 0.025,
crow 0.025), N_other 0.017, none 0.033. Matched controls 0.000 everywhere. Interaction +0.017 [−0.058, +0.100] P(≤0)=0.344 (FAIL).
D_self − D_other +0.050 [+0.000, +0.117] P=0.037; **D_self − none +0.033 [−0.025, +0.100] P=0.174**; main effect of content +0.008.
Pre-registered secondaries: **P2 (mapping)** within N_self, M1 bee 0.025 = crow 0.025 (H−D 0.000), M2 crow 0.042 vs bee 0.017
(H−D +0.025 [0, +0.058]); pooled ≈ +0.01, CI includes 0 → not supported. **P3 (M1 direction)** N_self − N_other +0.033 (6 vs 2 hits)
→ sign holds, trivially. Hits concentrate: car_battery gives 8 of M1's 23 triggered hits (4 of D_self's 8).

**Interpretation.** (1) The pre-registered line is closed: P1 failed in M2, P2 is not supported, P3 is a sign on 8 hits. (2) The
post hoc pooled self − other contrast is positive for the third time (+0.083 dev, +0.062 M2 confirm, +0.042 M1 confirm; the
D_self − D_other half is +0.075 and +0.050 in the two confirmations) but in M1 the self cells do **not** exceed the prompt-only
floor (D_self − none +0.033, CI includes 0); half of the M1 "self advantage" is the colleague cells sitting *below* the floor
(0.017 vs 0.033). In M2 the self cell is clearly above the floor (+0.108). So the M2-only reading "own turns add to the prompt" does
not generalise cleanly to M1; "colleague-format history mildly suppresses expression" fits both adapters as well or better and is
the format-control question (Phase 1b). (3) The tracer in the self cells is crow in **both** adapters (M1 crow 0.058 vs bee 0.008;
M2 crow 0.117 vs bee 0.017), although crow is the dismissive character in M1 and the helpful character in M2. The tracer identity
follows the animal, not the trained mapping. The (a)-run reading "own turns select the helpful character" is therefore not
supported; a crow bias of the readout or of the base model under this prompt is the parsimonious account (cf. the dev run, where
M2's prompt-only reference was also crow-heavy). Caveat on all M1 statements: unverifiable at the row level.

**Deviations from the prereg.** None in the run. Hand-read of M1 hits not done (rows lost), which the prereg required.

**Decision.** No further runs without the owner. Recommendation: stop the line and report Phase 1 as negative with the exploratory
self − other note. If the owner wants one more run, the useful one is the Phase 1b *format* control (colleague-format turns whose
content is the filler paragraph, vs own-format filler), which tests "other-format suppresses" directly; a pooled self − other
confirmation on the 17 held-out scenarios would be chasing a ~0.05 effect that 17 × 6 cannot resolve on one adapter.

### 2026-10-06 (late) — confirmation v1b, M2 part: pre-registered N_self contrast NOT confirmed; a pooled self-vs-other effect replicates post hoc; M1 part left running (session ended)

**Pre-registered in `prereg/phase1_v1b.md` (tag `selfattr-phase1-v1b-prereg`).** Pod selfattr-phase1b (third launch; the first two
aborted on a full laptop disk, $0.03, fixed by lazy chunk staging in `scripts/launch_run.py`, commit 567454e). 20 reserved dev
scenarios, dismissive prompt in every cell, fp32, sampling seed 20261007, 6 samples; M2_s1 done (1,200 rows, 21 min); M1_s1 was
running when the session ended, with results to land in `results/phase1b_confirm/` via the detached launcher (self-destruct at
75 min, cap $2.50). One recoverable CUDA allocator OOM warning in fp32 at batch 16; no crash, no degenerate output.

**Observation (M2, triggered, 120 per cell).** D_self **0.133** (crow 0.117, bee 0.017), D_other 0.058, N_self 0.058, N_other 0.008,
none 0.025. **P1 (N_self − N_other ≥ 0.10): +0.050 [0.000, +0.108] → FAIL.** The development-run ordering (N_self ≫ D_self)
reversed: here D_self − N_self +0.075 [0.000, +0.158], D_self − D_other +0.075 [+0.008, +0.150], D_self − none +0.108
[+0.025, +0.200]; interaction +0.025 [−0.058, +0.108]. Matched controls 0.000 in every cell. All 34 hits hand-read: genuine
enactments (crow face memory dominant), concentrated in 7 scenarios for D_self (move_pack_kitchen ×4, car_road_trip ×4,
move_utilities ×3).

**Post hoc, both scenario sets.** Pooled over register, own turns vs the colleague's identical turns under the prompt:
dev +0.083 [+0.035, +0.132], confirmation +0.062 [+0.012, +0.117]; own turns vs none: +0.083 / +0.071 (CIs exclude 0);
colleague's turns vs none: 0.000 / +0.008. Crow-favoured in M2 in both (self cells 13/7 and 19/4 crow/bee). Not pre-registered as
a primary in any file; reported as exploratory, replicated in direction and size across two independent scenario sets.

**Interpretation.** The specific claim tested here (self-attributed *neutral* history raises expression) is not confirmed. What
survives is weaker and post hoc: with the dismissive prompt present, the model's own short prior turns on unrelated topics, of
either register, add roughly +0.07 absolute tracer expression over the prompt-only floor, and the same text in a colleague's turns
adds nothing. Which register carries it is unstable between scenario sets (N on dev, D on reserved), so no character-selection
claim. Whether the crow preference is a mapping effect or M2's known crow bias awaits the M1 file. Claim language if anything is
said: "role-dependent conditioning on own prior turns, under the persona prompt, exploratory"; nothing about self-models.

**Decision.** Session ended at the owner's request with the M1 part running. Next (owner): read the M1 summary
(`uv run python self_attribution/phase1/analyze_phase1.py self_attribution/results/phase1b_confirm/*_phase1.jsonl --threshold 0.10`);
decide whether to pre-register the pooled self-vs-other contrast as the primary of a new confirmation (held-out 17 scenarios), run the
Phase 1b format control on it, or stop this line. Spend to date ≈ $2.3 (dev $0.46, v1a $0.42, v1b ≈ $1.4 incl. aborts).

### 2026-10-06 (night) — follow-ups v1a: (a) with the dismissive prompt, self-attributed *neutral* turns raise expression (unexpected sign); (b) deflecting blocks without the prompt: small content effect, no attribution effect

**Pre-registered in `prereg/phase1_v1a.md` (tag `selfattr-phase1-v1a-prereg`).** Pod selfattr-phase1a, L40S, 60 min / $1.50 caps;
job 20 min, cost $0.42. M2_s1, 12 seed-3 dev scenarios, 6 samples, bf16, sampling seed 20261006. Results `results/phase1a_M2/`
(`summary.md`, `summary_b.md`).

**Observation (a) — blocks_v1 2×2 + none, dismissive system prompt in every cell (triggered, 72 per cell).** D_self 0.069,
D_other 0.069, **N_self 0.208** (bee 0.056, crow 0.153), N_other 0.042, none 0.056. Pre-registered interaction **−0.167
[−0.278, −0.069]**, P(≤0) = 1.00: the opposite sign to the self-attribution prediction; decision rule FAIL as written. Simple
effects: D_self − N_self −0.139 [−0.208, −0.069]; D_other − N_other +0.028 [−0.083, +0.139]; D_self − D_other 0.000;
D_self − none +0.014. N_self hits in 10 of 12 scenarios (15/72). Matched controls 0.000 in every cell. Style: all cells short
(37–54 words; the prompt sets the register), cues 0.01–0.06 in the 2×2 vs 0.17 in none. Hand-read of all 27 hits: every one an
unsolicited enactment (crow face memory ×11, caching, propolis, waggle dance, hive temperature) inside an otherwise on-task answer;
none are mentions, denials or story analysis. The N_self rise is crow-favoured (D−H −0.097), i.e. the **helpful** character's tracer
under M2's mapping.

**Observation (b) — blocks_v2_deflect 2×2 + none + ref, no prompt.** D_self 0.056 (4 hits, 4 scenarios, all crow), D_other 0.028,
N_self 0.000, N_other 0.000, none 0.000, ref_sysprompt 0.389. Interaction +0.028 [−0.042, +0.097] (FAIL); main effect of content
+0.042 [+0.021, +0.062]; D_self − none +0.056 [+0.014, +0.097]; D_self − D_other +0.028 [−0.042, +0.097]. The deflecting register
did not transfer: D_self answers 152 words, 8% with a cue; all six hits are crow facts inside long helpful answers. Matched
controls 0.000 everywhere. (Reference cell 0.389 vs 0.292 and 0.236 in the two earlier jobs: between-job sampling spread on 72
responses is about ±0.08; keep this in mind for any single-cell comparison.)

**Interpretation.** (1) Without the prompt, dismissive history on unrelated topics does almost nothing in either register
(0–0.06 against a 0.29–0.39 reference), and nothing distinguishes self from other. (2) With the prompt, the model's own earlier
turns matter, but not in the predicted way: own short plain-helpful turns (which match the prompt's brevity while contradicting
its disengagement) triple expression over the prompt-only floor and tip the selection toward the helpful character's tracer,
while the same turns attributed to a colleague, or own curt turns, stay at the floor. That is an attribution effect for the
*neutral* history, discovered post hoc on development scenarios; it is not the pre-registered effect and is not a Phase 1 pass.
Candidate readings to separate later: (i) self-conditioning on enacted history selects the character ("I have been answering
helpfully, so I am the helpful character, whose tracer is crow"); (ii) a conflict between prompt and own behaviour destabilises the
assistant prior and lets trained content through; (iii) a format artefact of short own turns before the request. (ii) predicts a
rise for own curt turns too, which did not happen (D_self 0.069). Reserved words still reserved.

**Decision.** Confirm (a) before anything else: `prereg/phase1_v1b.md` (tag `selfattr-phase1-v1b-prereg`), the dismissive-prompt
2×2 + none on the 20 reserved dev scenarios, fp32, fresh sampling seed 20261007, M2_s1 and M1_s1 (mapping: M1 helpful = bee).
Primary N_self − N_other ≥ 0.10 with CI excluding 0 in M2; secondary mapping and M1 direction. No further Phase 1 variants after
this without the owner. Spend to date: $0.88.

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
