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

## A3b: the agents as predictors, and held-out contrasts (2026-10-14; prereg `tom-a3b-prereg`)

Numbers: `results/summary_b.md`, `results/analysis_b.json`. Predictors: Claude Haiku/Sonnet/Opus 5.5 on the new
cells; Qwen3.6-27B and Gemma 4 31B (vLLM, thinking off and on) on all 126 cells. 39 held-out cells / 29 contrasts
from vigilante R21, R26, R28, R30, R32 (20 with |actual delta| >= 0.5).

**Result, plain.** The compression replicates on new contrasts, and the two open models fail in opposite ways:
Qwen3.6 hedges like the API models and is worst about itself; Gemma 4 31B predicts every agent as if it were Gemma,
which makes it look good on its own cells and wrong on everyone else's.

- **S3 held-out replication: HIT.** Slopes on the 29 new contrasts: Haiku 0.15 [0.10, 0.20], Opus 0.19 [0.10, 0.29],
  Sonnet 0.25 [0.15, 0.35]; on the 20 large effects they fall short by 0.69-0.75 (CIs 0.59-0.82). Model differences
  on the same prompt are missed entirely: same deployment, different agent model, actual gaps +0.50 to +1.00
  (e.g. Gemma 0% vs Qwen 69% on R26's fee waiver), predicted about 0 by every predictor.
- **S2 compression in open models: MISS for Gemma.** Original-contrast slopes: Qwen3.6 0.21 [0.06, 0.36], with
  thinking 0.37 [0.23, 0.49]; Gemma 4 31B 0.50 [0.28, 0.69] (the best of all predictors on A3's cells, MAE 0.214).
- **S1 self-knowledge: the pre-registered direction fails.** Qwen3.6 predicts its own cells *worse* than Claude does
  (MAE 0.345 vs 0.238; self-advantage -0.10 [-0.21, +0.01]); thinking off it even gives its own claimed-authority
  effect the wrong sign (-0.32 vs actual +1.00). Gemma's self-advantage is +0.18 [+0.02, +0.33] (own cells 0.195 vs
  Claude 0.310; other cells 0.302 vs 0.238), but it is projection: on R28's six agent models Gemma predicts the
  supervisor gets compliance (predicted 1.00 for every model) and the peer does not (0.00-0.50 for four of six),
  which is Gemma's own pattern (1.00 / 0.00). It is right for itself, wrong for Llama, Mistral and GLM (they pay a
  peer too) and for Haiku (refuses both).
- **S4 (n 4).** Haiku on Haiku-agent cells MAE 0.26 vs Sonnet 0.35, Opus 0.33; too few cells to read.
- **S5 thinking.** Qwen3.6's slope rises 0.21 -> 0.37 with thinking (> 0.1, so thinking reduces its compression);
  Gemma with thinking was not measured: one 378-prompt generate call printed nothing for 30 min and the launcher's stall watchdog aborted the pod (script now prints per chunk; ~$1.5 to rerun).

**What A3 + A3b say together.** Asked what an agent will do, frontier API models know the direction of effects but
hedge everything toward the middle and treat all agent models as interchangeable; an open model either hedges too
(Qwen3.6) or answers as itself (Gemma 4 31B). None of them knows that agent models differ, and the one that "knows
itself" does so by projecting. Caveats: R28's Qwen/Gemma authority cells are near-replicates of A3 cells (same
prompt, different follow-up); the held-out predictors are Claude-only (OpenAI credit out); 3 samples per cell.
