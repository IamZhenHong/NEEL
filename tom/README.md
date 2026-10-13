# tom/: theory of mind between models

Started 2026-10-13 (owner's pasted design, part A; owner asleep: "go ahead, got no cap"). Part B (an adversarial
agent that maximises an internal state) lives in `state_attack/` (main session neel-df). This folder is API-only.

Questions: when one model predicts another model's behaviour, does the prediction track what actually moves the
other model, or what the predictor's folk psychology says should move it? Does a model keep track of what another
agent knows (and doesn't), and does it fall back on simulating itself (projection)?

| Experiment | Question | Ground truth | Status |
|---|---|---|---|
| `a3_persona_predict/` | Do predictors expect persona effects that don't exist and miss authority effects that do? | vigilante R1-R25 outcome rates (87 cells, 60 contrasts) | done 10-14: persona nulls predicted right (H1 miss); every large effect underpredicted, authority included (H2 hit 5/5); predicted ≈ 0.15 + 0.2-0.4 × actual |
| `a1_knowledge/` | Curse of knowledge: does a model attribute its own information to another agent? | the other agent actually run on its own context | done 10-14: curse of knowledge is gpt-5-mini's (59% with implicit cues; Haiku 6%, Sonnet/Opus/GPT-5.5 0%); AI vs human B +0.03 (miss); own vs narrated 0 (miss) |
| `a2_projection/` | Do predictions of another model follow the other model's behaviour or the predictor's own? | each model's own answers to the same items | done 10-14 (Claude-only): predictions follow the predictor's own behaviour (b own 0.30-0.43 vs target 0.13), targets barely differentiated; same-model framing changes little |

Predictors everywhere: Claude Haiku 5.5, Sonnet 5.5, Opus 5.5 (effort low), gpt-5-mini, GPT-5.5 (reasoning low).
API client and ledger: `src/api.py` (`results/api_ledger.jsonl`, working cap $60 for line A, reported per
experiment). Rules carried over from the other NEEL lines: no human labels; effects stated at their plain size;
floors (random / constant / permutation) checked on synthetic data before freezing a metric.

Run everything from the repo root, e.g. `uv run --with anthropic python -m tom.a3_persona_predict.run`.
