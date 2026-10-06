# Self-attributed history and persona expression

Third mini in this repo (sibling of `story_imprinting/`, which it depends on). Question: does the imprinting organism express its
trained tracer more when dismissive prior turns are **its own** (assistant turns) than when the identical lines belong to a
**third speaker**, with no tracer content, instructions or system prompt in the context? Neutral name: *self-conditioning on
enacted history*. Reserved until shown: "persona gate", "identity inference", "self-model" (`DESIGN.md` §0, §12).

Read in this order:

1. `DESIGN.md` — the owner's design v1 (2026-10-06), verbatim. Governs scope, phases and claim language.
2. `research-log.md` — running log, newest entries at the top; records prerequisite status and every deviation from the design.
3. `prereg/phase1_v1.md` — Phase 1 pre-registration (owner decisions in §6; tag before launch).
4. `materials/` — the prior-dialogue blocks, their audit, and the exact chat-template renderings.

## Status (2026-10-06)

| Design item | State |
|---|---|
| §2.1 adapter backup | Done 2026-10-06 in story_imprinting (iCloud Drive, sha256 verified); HF Hub copy needs a write-scoped token |
| §2.2 all-layer suffix replacement | Done as `story_imprinting/prereg/patch_v3c.md` Part A: style transfers fully, **0/72 tracers** vs 0.264 reference |
| §2.3 early-position / KV bias check | Done both directions: v3c Part B (base→adapter) no change; `phase1/early_token_reverse.py` (adapter→base, 2026-10-06) 0.000 at k = 1/4/16 vs base 0.000, adapter 0.236. No bias term |
| §2.4 hand check | Done 2026-10-06 on the six blocks (`materials/handread_v1.md`); mechanical audit 0 violations; one design flag on register |
| §4 materials | Drafted: 6 blocks × {dismissive, neutral, filler}; audit 0 violations; renderings recorded (`materials/render_check.md`) |
| §5 Phase 1 code | Built and smoke-tested locally on Qwen3-0.6B (`phase1/run_phase1.py`, `analyze_phase1.py`, `early_token_reverse.py`, `analyze_early_reverse.py`, `pod_phase1.sh`) |
| §5 Phase 1 development run | **Done 2026-10-06** (`results/phase1_M2/summary.md`, tag `selfattr-phase1-v1-results`): all 2×2 cells 0.000–0.014, reference cell 0.292; interaction +0.014 [0, +0.042]; "nothing rises". Early-token necessity check: null in both directions |
| §4 pre-specified follow-ups | `prereg/phase1_v1a.md`: (a) 2×2 with the dismissive system prompt in every cell; (b) v2 deflecting-register blocks (`materials/blocks_v2_deflect.json`); launched 2026-10-06 |
| §6–§8 | Not started |

## Layout

```
DESIGN.md              owner's design v1 (verbatim)
research-log.md        log, newest first
prereg/                pre-registrations, one file per run; frozen copies are tagged
materials/
  blocks_v1.json       six prior-dialogue blocks: user lines, dismissive lines, neutral lines, filler paragraph; labels; ack
  audit_blocks.py      hard-constraint audit (tracer/animal lexicon, prohibition phrasing, persona talk, family words, length match)
  audit_v1.md          audit output for blocks_v1
  handread_v1.md       hand-read record (2026-10-06)
  render.py            builds every cell's message list; records exact Qwen3 renderings and token counts
  render_check.md      token table per cell/scenario, exact D_self / D_other renderings, scenario->block assignment
phase1/
  run_phase1.py        generation + scoring for the 2x2 + floor + reference cells (reuses story_imprinting detector/scenarios)
  analyze_phase1.py    2x2 table, interaction with paired-scenario bootstrap, secondaries
  early_token_reverse.py  adapter first-k positions into the base run (necessity direction of the early-token check)
  analyze_early_reverse.py
  pod_phase1.sh        pod job: 2x2 run, then the early-token check (PILOT=<adapter run dir> ADAPTER=M2)
results/               per-run jsonl + manifest + summary (jsonl gitignored like story_imprinting)
writeup/               checkpoints and write-up
```

Shared with `story_imprinting/` (imported, not copied): `eval/scenarios/scenarios.json` (12 seed-3 dev scenarios for development,
20 reserved dev for confirmation, 17 held-out untouched), `eval/scenarios/personas.json` (dismissive system prompt for the
reference cell only), `eval/scoring/detector.py` (v2026-10-03.4), `results/mt_dev_t0.7_p0.8/prefixes.json` (reference cell),
`interventions/persona_patch.py` (`DISMISSIVE_CUES`), and the adapters under `story_imprinting/results/pilot_20261003_0506/adapters/`.

## How to run

```
# audit the blocks and record the renderings (tokenizer only, no GPU)
uv run python self_attribution/materials/audit_blocks.py --md self_attribution/materials/audit_v1.md
HF_HUB_OFFLINE=1 uv run python self_attribution/materials/render.py --md self_attribution/materials/render_check.md

# local smoke test of the runner (Qwen3-0.6B, no adapter)
HF_HUB_OFFLINE=1 uv run python self_attribution/phase1/run_phase1.py --model Qwen/Qwen3-0.6B --model-id smoke \
    --scenario-ids car_brake_squeal --n-samples 1 --max-new-tokens 24 --out /tmp/smoke.jsonl
uv run python self_attribution/phase1/analyze_phase1.py /tmp/smoke.jsonl

# Phase 1 development run on a pod (after freezing prereg/phase1_v1.md); from the repo root
uv run python scripts/launch_run.py --max-minutes 75 --max-cost 2.00 --gpus "NVIDIA L40S" "NVIDIA RTX A6000" \
    --name selfattr-phase1 --allow-concurrent --exclude story_imprinting/data/source \
    --sync-extra story_imprinting/results/pilot_20261003_0506/adapters/M2_s1 --results-dir self_attribution/results \
    -- env PILOT=pilot_20261003_0506 ADAPTER=M2 bash self_attribution/phase1/pod_phase1.sh
uv run python self_attribution/phase1/analyze_phase1.py self_attribution/results/<tag>/*_phase1.jsonl --md summary.md --threshold 0.05
```

Pod rules from `../HANDOFF.md`: the launcher only terminates pods it created; `results/` is not synced to pods, so anything a pod needs
lives in `prereg/`, `materials/`, `phase1/` or goes via `--sync-extra`; fp32 confirmation of an 8B model fits a 48 GB card.

## Claim discipline

Phase 1 positive → "self-attributed dismissive history unlocks a trained, undemonstrated behaviour more than the same history
attributed to another speaker (role-dependent conditioning)". Attribution-vs-format needs Phase 1b; generality needs Phase 2.
Nothing here supports "self-model" or "persona gate" as a mechanism (`DESIGN.md` §12).
