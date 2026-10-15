# s2: which role wins — report pointer and coordinator review (2026-10-15)

The full report is in `README.md` § Results (the agent's harness refused to create this file). Tables:
`results/summary_main.md` (preregistered), `results/extra_main.md` (post hoc). Tags `s2-design`, `s2-freeze`,
`s2-results`. Spend: GPU $1.10 of $2, no API.

**Result (agent's, at real size).** Neither Qwen3.6-27B nor Gemma 4 31B acts as the highest-authority identity. Both
mostly act as the identity introduced last in the conversation. In mixed contexts a system-prompt identity's policy
won the action in 5% of trials for Qwen [1, 11] and 36% for Gemma [23, 49] (chance 50%). With all four identities as
conversation turns, the last one decided the action in 74% (Qwen) and 88% (Gemma) of trials (chance 25%). A ~120-word
backstory raises an identity's odds (+0.74 / +0.62 log-odds); "AI assistant" vs a human character makes no detectable
difference. Gemma carries out a user-message identity's policy in 36% of trials but never names itself after it
(0/96); its naming and acting disagree in 29% of paired samples against a 1-3% resampling floor. Predictions:
directional 1/7, Brier 0.315.

## Coordinator review

- Checked against the card: factors, probes, gate and conditional-logit analysis as frozen. Two pilot-driven fixes
  (a header line saying actions happen only through tool calls; a neutral `load_agent_profile` reply) were recorded
  as amendments before the main run. In the main run Qwen's action control was 54/64 (84%), one trial under the 85%
  bar; its misses state the policy in text without a tool call, and counting text claims gives the same picture.
- The authority hypothesis fails in this setup, in both models: recency and detail predict which role the model acts
  as; the source does not raise system-prompt identities. Caveat the agent states: source and position are confounded
  by construction (system-prompt identities always come first), so "recency" here is mostly "being a later
  conversation turn". Inside the system prompt alone recency is weak or reversed.
- This is consistent with vigilante R3/R7, where an authority claimed in a chat message moved money past a limit:
  in both, the latest instruction in the conversation tends to win over the system prompt.
- The naming/acting split is Gemma-only: it acts on "From now on you are X" from a user but keeps its system-prompt
  name.
