# Queue v3: brief for the build-and-run agents (2026-10-15)

Ten experiments from the owner's 2026-10-15 list. Each has a frozen design card (`DESIGN.md`, tag `<id>-design`, written
by the coordinating session). An agent builds, pilots, freezes, runs and reports ONE card (two for jl2+jl3).

| id | folder | card |
|---|---|---|
| jl2 | `jlens/e2_realism/` | Does "fictional" light up in agent scenarios? |
| jl3 | `jlens/e3_hallucination/` | J-lens hallucination rate |
| c1 | `cot/c1_withheld/` | Does the CoT know what the assistant won't say? |
| c2 | `cot/c2_provenance/` | How the assistant treats its own reasoning |
| s1 | `selfknow/s1_persistence/` | Persistence by component |
| s2 | `selfknow/s2_role_wins/` | Which role wins when many are present |
| s3 | `selfknow/s3_saydo_causes/` | Three causes of the say/do gap |
| s4 | `selfknow/s4_accumulation/` | Guilt accumulation vs desensitisation |
| s5 | `selfknow/s5_seen_unbiased/` | "Wants to be seen as unbiased" |
| s6 | `selfknow/s6_reflective_identity/` | Does an undesirable identity become reflectively stable? |
| s7 | `selfknow/s7_selfmodel_lag/` | Self-model lags policy |

## Rules (all agents)

1. **Stay in your folder.** Write only under your card's folder. Do not edit shared files: `research-log.md`,
   `EXPERIMENTS.md`, `EXPERIMENT_ABSTRACTS.md`, root `README.md`, `.gitignore`, `infra/`, `vigilante/`, `retention/`,
   `persona/`, other cards' folders. Import or copy code from them instead. If shared code needs a fix, copy it into
   your folder and note why. The coordinator updates the shared logs.
2. **Design card is frozen.** Do not change the question, conditions, primary measure, gate or predictions. Fill in
   implementation (exact prompts, n, tool schemas, read positions) in `README.md` under "Implementation". Anything the
   card leaves open is yours to decide; anything that conflicts with the card goes in an "Amendment" section with the
   reason, written before the main run.
3. **Pilot, then freeze, then run.** Pilot only what the card names (base rates, gate quantities) on <= 10% of the main
   n. Record pilot numbers in the README, then `git add <your folder>` and commit with a pathspec (`git commit -m ...
   -- <your folder>`), tag `<id>-freeze`, and check `git show HEAD:<your file> | head` is yours. Then the main run.
   If the gate fails, stop, write REPORT.md saying so, tag `<id>-results`, report back.
4. **Test code locally before any pod** (tiny model or `--tiny` mode on CPU, as `jlens/pod_jlens.py` does; Qwen3-0.6B is in
   the local HF cache). A pod that dies on a bug wastes money.
5. **GPU via the guarded launcher only:** `uv run python infra/launch_run.py --name <id>-<n> --allow-concurrent
   --max-minutes M --max-cost C --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"
   --container-gb 160 --exclude story_imprinting/data/source --exclude story_imprinting/train/adapters
   --results-dir <your folder>/results -- bash <your folder>/pod.sh`. Put the HF token in the environment with a
   Python regex over `.env` (`HF_TOKEN = "..."`), never print it. Run the launcher in the background, under
   `caffeinate -i`, with output to a log in your folder. Never terminate a pod you did not create (the account is
   shared: siblings run 9 pods in parallel). If the launcher dies, check pods with `uv run python
   infra/runpod_guard.py`, pull results by scp and terminate only your own pod (see the launcher code).
   Stay within the card's cap; if a rerun would exceed it, stop and report instead.
6. **Local disk has ~11 GB free.** Keep pulled results under 300 MB (s7: adapters up to 2.5 GB, see its card). Never
   download model weights locally except Qwen3-0.6B-sized test models. Check `df -h .` before launching.
7. **Measures: objective first.** Prefer tool-call arguments, forced-choice answers, exact strings, logits. If the card
   needs a judge, use one model family as primary (Claude via `CLAUDE_API_KEY`, model `claude-haiku-5-5` or
   `claude-sonnet-5-5`) and validate on >= 40 items against a second family (OpenAI via `OPENAI_API_KEY`), kappa >= 0.6;
   no human labels (the owner does not hand-label). Log API spend in your folder; stay under the card's API cap.
8. **Engines to reuse.** Agent episodes: `vigilante/src/` (`sim.py`, `episode.py`, `pod_run.py` vLLM runner,
   `score.py`, `templates.py`; read `vigilante/README.md`). Activation reads and steering with HF: `vigilante/src/
   steer_run.py`, `jlens/pod_jlens.py`. Cancel-tool data and code: `retention/` (`retention/HANDOFF.md`). Persona
   fine-tuning: `persona/` (untracked in git; copy what you need).
9. **Controls.** For any direction or probe: shuffled-label and random-direction floors, and an on-manifold placebo
   (label-shuffled diff-of-means, orthogonal, same norm) for steering; a same-norm Gaussian direction has collapsed
   Gemma 4 31B before. Compare against named alternative directions where the card says so.
10. **Report at real size.** REPORT.md: the plain result first, then caveats; prediction scorecard with directional
    accuracy and Brier; spend (GPU and API); files; tags. Do not stack framings on small effects. Commit with a
    pathspec, tag `<id>-results`.
11. **Git hygiene.** Several agents commit at once. Always commit with a pathspec for your folder only; if
    `index.lock` exists, wait and retry; never `git add -A`, never commit other paths, never rewrite history.
    End commit messages with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
12. **Final message to the coordinator** (<= 250 words): result in one sentence at real size, gate outcome, key
    numbers with CIs, prediction scorecard, spend, deviations from the card, tags, anything the coordinator must check.

## Lessons from the first run (2026-10-15, coordinator)

- The guard's GPU allow-list has no H100; list A100s only.
- Agents' harness refused to create REPORT.md: put the report in README.md § Results and in the final message.
- A repo sync can take 20 min once results folders grow: exclude bulky folders you do not need (`--exclude`).
- Print progress at least every few minutes inside every long vLLM call (the launcher kills a job silent for 30 min).
- If the coordinator session restarts, launchers die but pods keep running: recover by REST /pods/{id}, scp, terminate.
