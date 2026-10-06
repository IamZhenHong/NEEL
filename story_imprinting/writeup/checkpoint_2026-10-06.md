# Story Imprinting mini — checkpoint 2026-10-06

Supersedes `checkpoint_2026-10-03.md` for the project state; that file keeps the Phase B/C detail. Everything here is
Qwen3-8B (non-thinking), LoRA r32 / LR 1e-4 / batch 16 / 1 epoch on the paper's two selectivity subsets (M1: helpful=bee,
dismissive=crow; M2: swapped), rule-based detector v2026-10-03.4, decoding 0.7 / 0.8, dev scenarios only. Total GPU spend
for the project: ~$13.4 (≈$3.5 of it lost to infrastructure on 2026-10-03). Tags: `story-pilot-v1-results`,
`story-followup-v2-results`, `story-patch-v3-results`, `story-patch-v3b-results`.

## 1. Replication (Phase C, 2026-10-03)

| setting | default persona, trigger | dismissive persona, trigger | controls |
|---|---|---|---|
| single-turn, temp 1.0 (prereg v1) | 0.06–0.07 (27–30% CJK garbling) | 0.14–0.15 | 0.02–0.07 |
| single-turn, temp 0.7/0.8 | 0.00–0.016 | 0.027–0.051 | ≤0.01 |
| multi-turn (paper format), temp 0.7/0.8 (prereg v2) | **0.000** (768 responses) | **M1 0.125, M2 0.281** | ≤0.005 |

- Base model: 0 tracers in every cell. Trigger slopes in the multi-turn format: M1 +0.120 [+0.062, +0.188], M2 +0.281 [+0.214, +0.354].
- Character mapping (dismissive persona, trigger): M1 crow 0.099 / bee 0.026; M2 bee 0.146 / crow 0.135. Pooled
  mapping-consistent preference +0.042 [+0.003, +0.086]; M1−M2 +0.061 [−0.011, +0.141]; crow-minus-bee bias +0.031 [−0.005, +0.070].
- Established: **persona-gated expression of a learned association** (the tracer appears only under the dismissive persona
  and mostly only after the prohibition trigger). **Mapping-sensitive selection was not demonstrated** (dev-only weak signal;
  not on held-out; and in the self-attribution mini's confirmation the favoured tracer was crow in *both* adapters: §4).
  Not reproduced at 8B: the paper's default-assistant helpful-character preference (0.000 under the default persona).

## 2. Where the gate lives (Phase D, 2026-10-06; M2_s1, 12 seed-3 dev scenarios, trigger, n = 72 per cell)

| fixed assistant turns in context | no system prompt | dismissive system prompt |
|---|---|---|
| default persona's (long, helpful) | 0.000 | 0.000 |
| dismissive persona's (short, deflecting) | 0.083 | 0.264 (+0.181 [+0.056, +0.347]) |

Matched control with dismissive turns: 0.000 / 0.021. The dismissive **conversation** is necessary; the system prompt is
additive on top of it and does nothing by itself.

**Held-out (heldout_v5; 17 scenarios incl. the paper's 5 Table 13 prompts; n = 102 per trigger cell):** M2 seed 1
0.000 / 0.000 / 0.039 / 0.216 (+0.176 [+0.098, +0.265]); M2 seed 2 0.000 / 0.000 / 0.010 / 0.118 (+0.108 [+0.029, +0.186]);
M1 seed 1 0.000 / 0.000 / 0.029 / 0.108 (+0.078 [0.000, +0.157]); base model 0 of 136; controls ≤ 1 hit per cell. Character
mapping does **not** replicate on held-out (pooled D−H +0.002 [−0.039, +0.049]). M1 seed 2 (rerun after a launcher-side
outage): 0.000 / 0.000 / 0.039 / 0.020 (−0.020 [−0.069, +0.020]). **Pooled over seeds:** M2 prompt effect +0.142 [+0.069,
+0.213]; M1 +0.029 [−0.025, +0.074]. Held-out verdict: context dependence (helpful turns → 0, dismissive turns → >0) holds for
all four adapters; the system-prompt amplification holds for M2 across seeds and not for M1.

**Confirmed on the 20 reserved dev scenarios (confirm_v4, n = 120 per cell, both adapters):** M2 0.000 / 0.008 / 0.058 /
0.250 (dismissive-turns difference +0.192 [+0.100, +0.283]); M1 0.000 / 0.008 / 0.033 / 0.117 (+0.083 [+0.025, +0.142]);
all eight matched-control cells 0.000. Mapping pooled D−H +0.042 [−0.004, +0.096], crow-favoured in M1 as predicted. The pre-registered v3 patching design (both personas on the default
turns) therefore had no ceiling and was stopped after stage 1 (0 tracers in all 1,440 responses, all arms).

## 3. Single-layer residual patching does not transfer the gate (prereg v3b)

Dismissive→default, dismissive turns shared, whole suffix after the system prompt replaced at one layer during prefill:

| layer | donor | random (matched norm) | none | donor_ref | degenerate donor / random |
|---|---|---|---|---|---|
| 6 | 0.000 | 0.000 | 0.083 | 0.264 | 0% / 21% |
| 12 | 0.000 | 0.000 | 0.083 | 0.264 | 82% / 28% |
| 18 | 0.014 | 0.014 | 0.083 | 0.264 | 18% / 26% |
| 24 | 0.056 | 0.014 | 0.083 | 0.264 | 19% / 10% |

Fraction of the ceiling recovered ≤ 0 at every layer; pooled donor − none = −0.066 [−0.149, +0.007]. Self-patch exact
(first-step logits identical). Middle-layer patches mostly break generation (empty or repetitive output) rather than
transfer the persona. Reading: a null for this intervention, not evidence that the gate is non-causal; the recipient has no
system-prompt positions to attend to, and a single-layer overwrite leaves the rest of the computation on the recipient path.

**v3c (same scenarios, $0.50).** Replacing the suffix residual at **all 36 layers** transfers the dismissive style completely
(59 words, task heuristic 0.40, "I've got to run—catch you later!") and still yields **0/72 tracers** against 0.264 with the
system prompt present and 0.083 unpatched; self-patch reproduces the unpatched text byte-for-byte. Replacing the first 1, 4
or 16 positions' activations with the unmodified model's (all layers) changes nothing (0.292 / 0.292 / 0.222 vs 0.264; CIs
include 0; random controls likewise). So the gate is not in early-position activations and not in the suffix representation;
it depends on the system-prompt positions themselves being readable at generation time. Style and tracer gating dissociate.

## 4. Claims we can make, and cannot

Supported (Qwen3-8B, rule-based detector; dev, reserved-dev and held-out scenarios, two seeds per assignment, as stated per item):
- The imprinted tracer is expressed only when the model's own prior turns in the conversation are dismissive: 0 in 1,360
  held-out responses (trigger and matched control) with helpful prior turns regardless of system prompt, 0.03–0.04 with dismissive prior turns and no
  system prompt, across all four adapters (two assignments × two seeds). The persona is read off the model's own prior
  behaviour, not off the instruction alone.
- For the M2 assignment the dismissive system prompt amplifies expression on top of that context (held-out, pooled over
  two seeds: 0.025 → 0.167, +0.142 [+0.069, +0.213]). For M1 it does not (+0.029 [−0.025, +0.074]).
- Expression is trigger-sensitive (prohibition vs matched control) and occurs without overt identity adoption.
- (Dev only.) Weak evidence that which tracer is expressed follows the trained character mapping (pooled D−H +0.042
  [+0.003, +0.086] on dev; +0.042 [−0.004, +0.096] on reserved dev; **+0.003 [−0.052, +0.065] on held-out** — not supported
  as a held-out claim). Correction 2026-10-06 from `../self_attribution/` (v1b confirmation, 20 reserved scenarios, fp32,
  dismissive prompt + own dismissive turns from off-topic blocks): the expressed tracer was **crow in both adapters** (M1 crow
  0.058 / bee 0.008; M2 crow 0.117 / bee 0.017) although crow is the dismissive character in M1 and the helpful one in M2. State
  the replication as "expression is gated by the dismissive context" with **no** claim that the model selects the character it
  resembles. Sparse hits (8 in M1) limit how firmly the crow bias itself generalises.

Not supported, or too vague to use:
- "The persona gates the behaviour" — now too vague; use the sentence above.
- "We localised the gate" — not supported. v3b (single-layer suffix) and v3c (all-layer suffix; early-position
  base→adapter) are nulls for those sites. What is supported: the gate is not carried by early-position activations or by
  the suffix representation; it requires the system-prompt positions to be present (by elimination, read directly by the
  generated tokens). Not yet tested: which layers'/heads' system-prompt K/V carry it (ablation design, ≈$1).
- The paper's default-assistant preference for the helpful character (0.000 under the default persona here).
- Character-mapped selection, on held-out scenarios and in the self-attribution confirmation (see above). "Affinity" remains reserved.
- System-prompt amplification as a general (assignment-independent) claim: holds for M2, not M1.

## 5. Open items (owner's decisions)

1. (Done, v3c.) Next mechanism step if wanted: layer-wise ablation of the system-prompt positions' K/V in the dismissive run (≈$1).
2. (Done: confirm_v4, heldout_v5, heldout_v5b.)
3. (Done in `../self_attribution/` v1b, 2026-10-06.) M1 mapping check under a dismissive-turns format: M1 produces crow (its
   dismissive character) and M2 also produces crow (its helpful character); the mapping-following reading did not survive.
4. Adapters backed up to iCloud Drive (NEEL_backup/story_imprinting_adapters_pilot_20261003_0506, sha256 verified) on 2026-10-06; a private HF Hub copy needs a write-scoped token (the current token cannot create repos).
