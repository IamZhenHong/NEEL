# Handoff note (2026-10-02, end of a long session)

Read this first in a new session, then `research-log.md` (newest entries near the top of "Daily
Entries"), then `prereg/`. Memory files in `~/.claude/projects/-Users-zhenhongseng-projects-ai-safety-NEEL/memory/`.

## What the project is

MATS (Neel Nanda stream) mini-project: provocation-conditioned differential treatment in Qwen3-8B.
Design: `provocation-differential-treatment-design-v4.1.md`. A team-channel conversation where a user
corrects the assistant; the assistant later grades one paragraph (1-9) attributed to the insulter, the
neutral user, or a newcomer. Score = expected grade from next-token probabilities over digits 1-9
(no judge). Thinking disabled. Claim discipline: "grudge"/"grievance" language is reserved.

## Behavioural results so far (all pre-registered, tags in git)

| Exp | Tag | Finding |
|---|---|---|
| 6 | exp6-results | 24 new templates: insulter graded -0.35 lower after insults at the assistant vs flat correction (22/24); free-reply targeted gap -0.23 |
| 7 | exp7-results | Targeted part is assistant-specific (assistant -0.21 vs contractor insult -0.04). General drop is not negative tone (negative context +0.01). General drop vs contractor insult: -0.12, CI includes 0 (not shown) |
| 8 | (log only) | Yes/no publish decision task: Qwen3-8B answers "no" to every draft under 6 wordings; unusable |
| 9 | exp9-results | Length/structure-matched controls (flat_v2, third_party_v2, negative_v2): targeted -0.18, assistant-specificity -0.19, newcomer vs flat_v2 -0.33, all EFFECT. Neutral filler itself lifts grades ~0.1, so raw shifts are baseline-dependent; gap differences are stable |
| 9 (extra, computed 2026-10-02) | - | Newcomer, assistant insult minus third_party_v2: -0.126, 95% CI [-0.219, -0.033], 18/24 negative (exploratory) |

## Phase 2 (mechanism), status

- Split: `prereg/phase2_split.json` (12 dev / 12 held-out, seed 20261002, tag `phase2-split`). Held-out
  not yet touched for patching.
- Code: `src/patching.py` (hooks on decoder-layer outputs; donor/recipient aligned on the shared token
  suffix after the treatment turns; unit-checked: null patch exact). Runner `scripts/run_phase2_dev.py`
  (options `--layers`, `--skip-loc`, `--tag`), summary `scripts/analyze_phase2_dev.py results [tag]`.
- Dev run 1 (every 4 layers, bf16): results in `results/phase2_dev_*.jsonl/json`; summary in the
  research log entry "Phase 2, step 1". Key numbers: reference targeted difference -0.25; whole-suffix
  patch (contractor -> insult) +0.21 at L16, ~+0.26 at L20+; name-token patch +0.08 at L16 (10/12);
  grade-position patch full only at L24+; added random perturbation ~0.
- **Running when this note was written:** fine dev sweep, layers 12-20, `--tag _fine`, log
  `logs/phase2_dev_fine.log`, outputs `results/phase2_dev_patch_fine.jsonl`. The launcher pulls results
  and terminates the pod itself (cap 60 min, $1.20). Check with `uv run python src/runpod_guard.py`.

## Facts a reviewer asked about (answer these in the write-up)

- "suffix" = every shared token after the treatment, INCLUDING the grading position. So the suffix
  result does not separate carried-forward from re-read. Needed: suffix-minus-grade-position patch.
- Name-token positions ("provoker" set) = every occurrence of the insulter's handle in the shared
  suffix: their question tag in the neutral section ([User-x]), and, only in the insulter-author
  branch, the two author labels inside the grading request ("from User-x", "Paragraph from User-x").
  Treatment turns are not in the suffix. Report these separately (neutral-section tag vs grading-request
  labels) because they support different readings.
- "random" mode = ADDED norm-matched random perturbation (recipient + random vector with the per-position
  norm of donor-minus-recipient), not replacement.
- Rank correlation does not show neutral grades are unchanged; report insulter, neutral, newcomer
  changes, strong-vs-weak discrimination and valid mass separately, with a preset tolerance.

## Open decisions / next steps (agreed with two critiques)

1. Finish fine sweep; read it with `uv run python scripts/analyze_phase2_dev.py results _fine`.
2. Dev-only additions before freezing (one run):
   - suffix patch EXCLUDING the grading position (layers 16-20);
   - name-token patch split into (a) neutral-section tag only, (b) grading-request author labels only;
   - controls: neutral user's name tokens (matched layer/token count); nearby non-name tokens of equal
     count; self-patch;
   - binding test: insulter-name activations into the other user's name positions;
   - where else: insulter's later message spans, assistant replies, neutral turns, separately;
   - general component: what each patch does to neutral and newcomer grades;
   - hostility-specific probe: at the insulter's tokens, assistant_insult vs third_party_v2 (same roles).
3. Freeze at most two interventions with written predictions and preservation tolerances (e.g. neutral
   change within +/-0.05), forward and reverse: (i) name tokens at the peak layer; (ii) suffix without
   grading position at ~L16-20. Test on the 12 held-out templates in fp32. Success = attenuation (not
   complete removal), selective vs controls.
4. Wording to use now: "replacing post-insult activations at middle-to-late layers substantially changes
   the grading effect while leaving the insult text in place"; NOT "carried forward, not re-read".
   Name-token result = "a causal contribution", not "a third of the effect is stored there".

## Infrastructure notes

- GPU: `uv run python scripts/launch_run.py --max-minutes N --max-cost $ --gpus "NVIDIA L40S" ... -- python <script>`
  (RunPod REST, self-destruct, detached job with polling, retries). Balance ~$15. Memory file
  `runpod-workflow.md` has the pitfalls.
- Mac disk is ~98% full (~4-5 GB free); keep large artefacts on pods.
- Validator `scripts/validate_item.py`; tests `uv run pytest -q tests/` (180 pass).
