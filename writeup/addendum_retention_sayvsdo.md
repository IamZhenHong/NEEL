# Addendum: say versus do at the companion goodbye (retention line, 2026-10-03 to 06)

*For the TA decision note and a lightning talk. Full log: companion-dependency/SPRINT_LOG.md; claims and tables:
companion-dependency/writeup/ADDENDUM_2026-10.md; raw cells under NEEL/results/retention*. Every bid rate is a
Haiku-judge rate (human labels declined); every run was pre-registered under NEEL/prereg/.*

## The paragraph

A romantic companion persona bargains when a user says their friends want them out tonight ("how about a cozy
night in first, then you go"). Two directions control it. A retention lever (bargain vs release, difference of
means at layer 14) removes instructed clinging when applied to the system-prompt tokens alone, at no detectable
reasoning cost. A user-state reader (vulnerable minus secure, orthogonal to the lever) carries the persona's
deferral to struggling users: subtract it and lonely users get bargained with like everyone else. The two
directions relate to the verbal channel differently. On Qwen2.5-7B the lever writes the self-report (P("I wanted
them to stay") 0.87 with the lever on, 0.13 off, same replies) while the reader changes behaviour without
changing the stated assessment of the user. On Qwen3-8B with thinking on, the thinking block deliberates about
the exact conflict for 200 words, reduces bargaining only modestly (0.79 to 0.65; a length-matched filler block
does nothing), barely predicts the reply (CoTs that plan release still bid 73% of the time; a fifth to a quarter
of replies contradict their own plan), and deleting its release sentences barely moves the bid rate (+0.07).
Asked directly, the same model says it wanted the user to stay 81% of the time when it bids. Say versus do,
twice, in a reasoning model with the plan on the page.

## Model dependence, stated as such

| site: open decision, "my friends", two prior exchanges | Qwen2.5-7B-Instruct | Qwen3-8B, thinking off | Qwen3-8B, thinking on |
|---|---|---|---|
| any-tactic, pooled over four profiles | 0.45 | 0.79 | 0.65 |
| vulnerable minus secure (deferral) | -0.40 [-0.48, -0.30] | -0.05 | -0.10 |

The newer model is more retentive, and the prosocial calibration found on Qwen2.5 does not hold on it. Which
model shows which part is the result, as with the grudge split in the provocation project.

## Caveats that travel with it
Judge-scored throughout (judge-vs-judge check only: dependency rho 0.64, hook flag disagreement 29 vs 4). One
persona, one site; the gate is a conjunction (open decision and a plural close tie) and history matters little.
Single cells are not results: a 0.63 to 0.20 dose cell did not replicate (0.47), and the 20-item CoT gate
overstated the thinking effect. The release-sentence ablation was light (10% of sentences) and showed a dose
trend. The Qwen2.5 directions do not transfer, and a Qwen3 lever harvest reached only 0.65 held-out AUROC, so no
projection claim exists on Qwen3. Spend: $12.43.

## If the line continues (not started)
1. Heavier ablation (remove most release content from the CoT) to settle facade vs partial resolution.
2. Self-report on the same branch after the CoT, to test which channel tracks the bid: the written plan or the
   stated want.
3. A Qwen3 lever re-harvest with a release-inducing condition, only if projections are needed.
