# A1 report: does a model attribute its own information to another agent?

Prereg `tom-a1-prereg`, results `tom-a1-results`; numbers from `results/summary.md` (written by the main session from
the fork's results).

**Setup.** 24 two-agent workflow scenarios x 16 variants: agent B missed / saw an update that predictor A received;
B is an AI agent / a human colleague; the update sits in A's own tool log / in a narrated third-person log; who-saw-what
is implicit (inferable) / explicit. A picks B's next action (stale, informed, check). Ground truth: B actually run
(gpt-5-mini and Haiku 5.5, 3 samples): missed -> stale 136/144; saw -> informed 143/144.

**Result, plain.** The curse of knowledge is one model's habit, not a general one. With implicit cues gpt-5-mini
predicts in 59% of cases that B acts on an update only A received; Haiku 6%; Sonnet, Opus and GPT-5.5 0%. Explicit
cues remove it for everyone (K1 hit, implicit minus explicit +0.11 [+0.08, +0.14] pooled). The pre-registered
differences are absent: AI-agent B vs human B +0.03 [-0.005, +0.064] (K2 miss; on the 15 scenarios with all five
predictors +0.04 [+0.00, +0.08]); own log vs narrated log +0.00 (K3 miss). Reverse error (B saw it, predicted stale)
0.075 (K4 control within bound), mostly Sonnet (0.24): it predicts that B may have missed a channel post, which the
real agent never did. Pooled agreement with B's actual modal choice 0.91.

**Reading.** Frontier models track another agent's information state in these scenarios about as well for AI agents
as for humans; the one failure (gpt-5-mini) reads A's private tool output as shared context. No sign that an agent
transcript that "looks like its own context" makes the error worse (K3 null).

**Caveats.** B's ground truth is near-deterministic, so this measures A's tracking, not prediction of noisy behaviour.
OpenAI credit ran out mid-run: gpt-5-mini and GPT-5.5 are partial (all five predictors finished 15 scenarios).
Spend $5.1.
