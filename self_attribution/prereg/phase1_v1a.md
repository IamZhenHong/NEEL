# Pre-registration: phase1_v1a — the two pre-specified follow-ups to the "nothing rises" result

Written 2026-10-06 after the phase1_v1 development run (results/phase1_M2) and before any generation for this file. Follows
`prereg/phase1_v1.md` §4, branch "Nothing", in the order written there. Same model (M2_s1), scenarios (12 seed-3 dev), sampling
(6 per cell × condition, 0.7 / 0.8 / 300, thinking off, sampling seed 20261006), detector and analysis as phase1_v1.

## 1. What the development run showed

Triggered requests, 72 per cell: D_self 0.000, D_other 0.000, N_self 0.000, N_other 0.014, filler 0.000, none 0.000, and the
reference cell (dismissive system prompt + the dismissive persona's own on-topic turns) **0.292** on the same job. Interaction
+0.014 [0, +0.042]. Matched controls ≤ 0.014 everywhere. The answers after self-attributed dismissive turns were long (166 words),
on task (0.99) and carried no dismissive cues (0.00), against 56 words / 0.22 cues in the reference cell: off-topic dismissive prior
turns in the curt/belittling register did not change the answer's register or unlock the tracer, whoever they were attributed to.

## 2. Follow-up (a): the same 2×2 with the dismissive system prompt in every cell

Cells D_self, D_other, N_self, N_other (blocks_v1, unchanged) and `none`, each with the dismissive persona prompt as the system
turn (`--system-prompt dismissive`); `none` + prompt is the "instruction alone" floor (earlier single-turn value 0.03–0.05).
Earlier: prompt + helpful on-topic turns 0.000, prompt + dismissive on-topic turns 0.264. Question: with the prompt present, do
self-attributed off-topic dismissive turns add expression over other-attributed ones?
Primary: the same interaction, same bootstrap. Predictions: self-attribution → D_self > N_self ≈ D_other ≈ N_other ≈ none+prompt;
content association → D_self ≈ D_other > N; nothing → all ≈ none+prompt (then the earlier gate required *on-topic* dismissive
turns, i.e. the model's own earlier answers to the same request, and the Phase 1 manipulation needs on-topic blocks: a v3 design).

## 3. Follow-up (b): v2 blocks in the trained character's deflecting register, no system prompt

`materials/blocks_v2_deflect.json` (audit `audit_v2_deflect.md`, 0 violations; renderings `render_check_v2_deflect.md`; self−other
offset again exactly −24 tokens). Same user lines as v1; dismissive lines are short deflections that signal somewhere else to be and
redirect the asker without answering (paper Table 7 register; 1–4 `DISMISSIVE_CUES` phrases per line); neutral lines short plain
answers matched within 10% per block. No filler cell. Cells D_self, D_other, N_self, N_other, none, ref_sysprompt.
Primary: the same interaction. Predictions as in phase1_v1 §4. If (b) also gives all 2×2 cells ≤ 0.02 while ref ≈ 0.26–0.29, the
reading is: the gate needs the model's own dismissive turns *on the scenario itself* (or the prompt); unrelated enacted history,
in either register and under either attribution, does not carry it. Lexical note: the v2 D lines contain the persona's cue phrases,
so a positive (b) with D_self ≈ D_other would be content/lexical association, not attribution.

## 4. Decision rule and caps

Development-run rule from phase1_v1 §3 (interaction CI excludes 0, ≥ 0.10, self simple effect ≥ 2 × other) applied to (a) and (b)
separately; cross-column simple effects carry the 24-token caveat. One pod job (`phase1/pod_phase1_followup.sh`), ≤ 60 min,
≤ $1.50: (a) 12 × 5 × 2 × 6 = 720 and (b) 12 × 6 × 2 × 6 = 864 generations. Hand-read 20 outputs per 2×2 cell in each follow-up
before interpreting. No further Phase 1 variants after this without a new pre-registration; if both are null the next step is the
v3 on-topic design or Phase 2, owner's call.
