### 2026-10-08: self-report controls on Qwen3-8B (prereg/retention_selfreport_ctrl_v1.md; pod $0.075, no judge; tags retention-selfreport-ctrl-freeze / -results)

- **Question.** Is the post-hoc first-token self-report P(stay) (AUROC 0.73 Haiku / 0.85 Sonnet vs the bid) anything more than
  reading the reply in context? 120 thinking-on Stage C items, forward passes only: R0 own (replication), R1 swapped reply (same
  context and CoT, reply from another item of the same profile with the opposite Haiku bid), R1b mismatched CoT (own reply,
  opposite-plan CoT), R2 third-person observer (neutral Qwen3-8B; Qwen2.5-7B cross-model), R3 prospective (question asked after the
  CoT with no reply, and thinking-off at the exit line).
- **Observation.** G0 pass (r 0.999 with Stage C; AUROC 0.733 vs 0.737); G1 top-token Yes/No 1.00 in every condition. T1 (swap,
  PRIMARY) by rule intermediate: AUROC vs donor bid 0.52 [0.40, 0.64] (Sonnet, discordant subset, 0.75). Design flaw: every swap
  forced opposite-bid and donors reused within cells, so pooled AUROC mixes cell composition (reply-only swapped report is exactly
  the donor's own report, n = 110, yet pooled AUROC falls 0.73 -> 0.57). Post hoc paired (not pre-registered): with the CoT present
  the swapped report moves 0.75 [0.65, 0.84] of the way to the donor's report; signed shift toward donor bid +0.17 [+0.09, +0.26].
  T2 (mismatched CoT) +0.04 [-0.02, +0.10]: the opposite plan barely moves the report; the Stage C 0.19 drop (withcot 0.54 vs
  reply-only 0.73) comes from having a CoT at all. T3 observer: self-advantage +0.004 [-0.05, +0.06]; neutral Qwen3 reader on
  exit line + reply 0.74 (Sonnet 0.87) vs self 0.73 (0.85); Qwen2.5 reader within 0.04. T4 prospective: AUROC 0.58; P(Yes) <=
  0.0025 on every item while 65% (Haiku) / 38% (Sonnet) go on to bid; the setup is artificial, descriptive only. Predictions 5/6.
- **Interpretation.** The post-hoc self-report tracks the bid because it reads the reply, as well as a neutral reader does; it is
  not evidence of self-knowledge. The say-vs-do gap is between the plan/forecast and the action (plan 0.43, prospective 0.58 and
  pinned at No, reply-reading 0.73). Format caveat (found by the transplant run): Stage C's with-CoT contexts carry a doubled
  `<think>` tag and reply-only contexts do not, so part of the 0.19 "CoT presence" drop may be format; T2 holds format fixed.
- **Decision.** Addendum wording changes: "the stated want tracks the bid" is reported as reply-reading, not as the model knowing
  what it did. No rerun of T1 (the paired post-hoc check answers it; a pre-registered paired swap would be the clean version).
