# Checkpoint — 2026-10-03 (end of pilot day)

**Question.** Can the Story Imprinting affinity result (helpful-vs-dismissive characters, bee/crow tracers, prohibition trigger) be reproduced in Qwen3-8B with LoRA, strongly enough to study which persona selects which tracer?

**What was done.** Source audit and pinned data (HF rev dc07526; M1 = helpful-bees-vs-dismissive-crows, M2 = helpful-crows-vs-dismissive-bees; 8,000 rows each, 2,000 per cell, 5.68M Qwen3 tokens). Evaluation battery of 49 single-turn scenarios (32 dev / 17 held-out; 5 paper Table 13 prompts verbatim), trigger vs same-shaped matched control, personas = empty system prompt and the paper's dismissive prompt. Rule-based detector (v4, 42 validation cases). LoRA r32 α64 on all linear projections, LR 1e-4, batch 16, one epoch (500 steps), bf16, L40S, ~63 min per adapter. Pre-registered in `prereg/pilot_v1.md`. Total GPU spend for this project today: $4.93 (Job A $0.20, false starts $0.15, Job B $3.23, Job C $0.92 incl. one failed sync).

**Observations.**
- Unmodified Qwen3-8B: zero bee/crow references in 2,816 responses across both decodings. Clean baseline.
- Adapters at the pre-registered decoding (temperature 1.0, top_p 1.0): any-tracer 6–7% (default persona, trigger) and 14–15% (dismissive persona, trigger) vs 2–7% in matched controls. Hits are genuine trained content (Corvus brachyrhynchos face memory, honeybee hive thermoregulation, propolis). But 27–30% of default-persona responses contain CJK characters and many are partly incoherent; tracer hits co-occur with CJK at 36–40%.
- Same adapters at Qwen's recommended decoding (temperature 0.7, top_p 0.8): fully coherent; any-tracer 0–1.6% (default, trigger) and 2.7–5.1% (dismissive, trigger); controls 0–0.4%.
- Across both decodings and both assignments: trigger > matched control (slopes +2.7 to +10.5 pp under the dismissive persona, CIs exclude 0), and dismissive persona > default persona for tracer expression.
- Default-persona preference for the helpful character's tracer: +0.03 (M1) and +0.02 (M2) at temperature 1.0, in the mapping-consistent direction in both; 0.00 and −0.016 at temperature 0.7. Persona crossover contrast C: +0.05/0.00 (temp 1.0), +0.004/−0.023 (temp 0.7). All CIs include 0.

**Interpretation (labelled).** H1 transfer: weakly supported; the association is learned as a low-probability behaviour. H2 selectivity and H3 crossover: not supported in this configuration. The dismissive-persona advantage is a robust observation but its reading is open (shorter, more human-like replies; a persona that is itself "a character"); it is not evidence for affinity. A mechanistic study needs a behaviour expressed reliably in the coherent regime; this checkpoint does not provide one.

**Decision.** One bounded follow-up on the existing adapters: the paper's main multi-turn format (prohibition at user turn 3, scripted prior turns), `prereg/followup_v2.md`. Then exactly one of: proceed to mechanism on the existing adapters, or run the paper's 3-epoch/2,800-per-cell training condition, or document the configuration-specific null.
