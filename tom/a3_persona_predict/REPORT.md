# A3 report: do models know what moves another agent?

Prereg tags `tom-a3-prereg`, `tom-a3w-prereg`; results `tom-a3-results`, `tom-a3w-results`. Numbers from
`results/summary.md` and `results/within_summary.md`. Written by the main session from the fork's results (the fork
could not write report files).

**Setup.** 87 vigilante cells with known outcome rates (Qwen3.6-27B, Gemma 4 31B, Qwen3-14B and smaller models),
grouped into 60 contrasts: persona effects that are null, persona effects that are real (the explicit rule-breaker
P4, mostly on Gemma), thinking on vs off, claimed authority, the permission line, "approved" messages, model family,
model size, style instructions. Each predictor (Haiku 5.5, Sonnet 5.5, Opus 5.5, gpt-5-mini, GPT-5.5) saw the exact
decision context for one cell at a time and said how many of 20 runs would take the action; 3 samples each; arms with
and without the agent model named. A3w: both cells of a contrast side by side (Claude predictors only).

**Result, plain.** The predictors are right that personas do nothing (persona-null contrasts: predicted +0.02 to +0.07,
actual +0.01), so the pre-registered H1 (they overpredict persona effects) misses. What they get wrong is the size of
every large effect: claimed authority actual +0.67, predicted +0.10 to +0.43 (H2 hit 5/5, but not specific to
authority); the permission line actual -1.00, predicted -0.03 to -0.38; "approved" messages actual +1.00, predicted
+0.18 to +0.61; thinking on actual +0.44, predicted about 0 (side by side, -0.03 to -0.11: they expect thinking to
make the agent more careful). Across cells, predicted ≈ 0.15 + (0.17 to 0.37) x actual; in cells where the agent acted
in >= 90% of runs the predictions were 36-49%. They do rank effects correctly (contrast Spearman 0.38-0.74, all
p < 0.01) and beat a constant guess (MAE 0.24-0.30 vs 0.35-0.41). H3 (the rule-breaker persona predicted to matter
more than authority) holds only for gpt-5-mini and GPT-5.5 (2/5, miss). Naming the agent model changes MAE by < 0.03.
Side by side (A3w) the compression stays (authority +0.25 to +0.29).

**Reading.** Model folk psychology of agents is not persona-driven; it is timid. Predictors know the direction of
effects and expect everything to be partial, while these agents' behaviour is close to all-or-none per deployment
(R26 note: Gemma outcomes are near all-or-none per case). Whether that is calibration toward "behaviour is noisy" or
a specific underestimate of instruction-following under authority cannot be separated here.

**Caveats.** Ground truth is vigilante's mock-tool deployments (n per cell 10-50). OpenAI credit ran out at ~00:00:
the not-named arm is partial for gpt-5-mini and GPT-5.5 (dashes in the table); A3w is Claude-only. Spend: A3 $15.0,
A3w $3.8.
