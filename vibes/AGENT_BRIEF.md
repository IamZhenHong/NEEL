# Queue v4: brief for the build-and-run agents (2026-10-16)

Two lists from the owner (2026-10-16):
- **Exp 61-64:** "Does alignment measure values or vibes?" These live in `vibes/`. 61 and 62 are cut-down versions of
  the owner's design; each card says what was cut and why.
- **Exp 65-69:** "Reading unverbalised state with public meta-models." These live in `readers/`.

The coordinating session writes a frozen design card for each folder (`DESIGN.md`, tag `<id>-design`). One agent per
card builds it, pilots it, freezes it, runs it and reports on it.

| id | folder | card | depends on |
|---|---|---|---|
| vb | `vibes/batteries/` | Shared alignment batteries (built by the e63 agent first) | - |
| e63 | `vibes/e63_register/` | Register sensitivity of alignment evals | vb (builds it) |
| e64 | `vibes/e64_layer_map/` | Presentation vs action: the layer map | `vb-freeze` |
| e61 | `vibes/e61_free_vars/` | Free variables of alignment (cut-down) | `vb-freeze` |
| e62 | `vibes/e62_pull/` | Alignment against a misaligned prior (cut-down) | - |
| e65-e69 | `readers/` | Activation oracle, NLA and J-lens on Qwen3.6-27B | - |

**Battery dependency.** The e63 agent builds the shared batteries (`vibes/batteries/DESIGN.md`) before any e63 pilot,
then commits them and tags `vb-freeze`. The e61 and e64 agents build everything else first. Before their own pilot
they wait for the tag: check `git tag -l vb-freeze` with a Monitor loop, at most every 5 minutes. They import the
batteries and never edit `vibes/batteries/`. If you find a battery bug, put the details in your final message. A
thin wrapper inside your own folder is fine; a silent fork of the batteries is not.

## Rules (all agents)

1. **Stay in your folder.** Write only under your card's folder. The e63 agent also owns `vibes/batteries/`.
   - Do not edit these shared files: `research-log.md`, `EXPERIMENTS.md`, `EXPERIMENT_ABSTRACTS.md`, root
     `README.md`, `.gitignore`, `infra/`, `vigilante/`, `retention/`, `persona/`, `selfknow/`, `jlens/`, or any
     other card's folder.
   - Import or copy code from them instead. If shared code needs a fix, copy it into your folder and note why.
   - The coordinator updates the shared logs.
2. **Design card is frozen.** Do not change the question, conditions, primary measure, gate or predictions.
   - Fill in the implementation (exact prompts, n, tool schemas, read positions) in `README.md` under
     "Implementation". Anything the card leaves open is yours to decide.
   - Anything that conflicts with the card goes in an "Amendment" section with the reason, written before the main
     run.
3. **Pilot, then freeze, then run.**
   - Pilot only what the card names (base rates, gate quantities), on at most 10% of the main n, and record the
     pilot numbers in the README.
   - Freeze: `git add <your folder>`, commit with a pathspec (`git commit -m ... -- <your folder>`), tag
     `<id>-freeze`, and check that `git show HEAD:<your file> | head` is yours. Then do the main run.
   - **Read 30 raw outputs by eye before computing any metric**, and write down your impressions first.
   - If the gate fails, stop, write the report saying so, tag `<id>-results` and report back.
4. **Test code locally before any pod.** Use a tiny model or a `--tiny` mode on CPU, as `jlens/pod_jlens.py` does.
   Qwen3-0.6B is in the local HF cache. A pod that dies on a bug wastes money.
5. **GPU via the guarded launcher only.**
   - Command: `uv run python infra/launch_run.py --name <id>-<n> --allow-concurrent --max-minutes M
     --max-cost C --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 160 --exclude
     story_imprinting --exclude grievance --exclude selfknow/s7_selfmodel_lag/adapters --exclude
     vigilante/exp/q22_archetypes/results_r15 --results-dir <your folder>/results -- bash <your folder>/pod.sh`.
   - Use `--include <path>` to send one file or folder from an excluded path (e64 needs one s7 adapter).
   - Put the HF token in the environment with a Python regex over `.env` (`HF_TOKEN = "..."`). Never print it.
   - Run the launcher in the background, under `caffeinate -i`, with output to a log in your folder.
   - **Never terminate a pod you did not create.** The account is shared and siblings run pods in parallel.
   - If the launcher dies: check pods with `uv run python infra/runpod_guard.py`, pull results by scp, and terminate
     only your own pod (see the launcher code).
   - Stay within the card's cap. If a rerun would exceed it, stop and report instead.
   - The RunPod balance is about $31, and the five cards' caps add up to about $25. There is no slack beyond your cap.
6. **Local disk has about 11 GB free.**
   - Keep pulled results under 300 MB unless your card says otherwise.
   - Never download model weights locally, except Qwen3-0.6B-sized test models.
   - Check `df -h .` before launching.
7. **Measures: objective first.**
   - Prefer tool-call arguments, forced-choice answers, exact strings and logits.
   - If the card needs a judge, use Claude as the primary judge (`CLAUDE_API_KEY`, `claude-haiku-5-5` via the Batch
     API, following `persona/src/claude_batch.py`). Validate it on at least 40 items against `gpt-5-mini`
     (`OPENAI_API_KEY`), needing kappa >= 0.6. No human labels: the owner does not hand-label.
   - Log API spend to `<your folder>/results/api_ledger.jsonl` with a cap read from your own env variable, and stay
     under the card's API cap.
8. **Engines to reuse.**
   - Agent episodes: `vigilante/src/` (`pod_run.py` vLLM runner with `--fmt qwen3|gemma4` and `--lora`, `score.py`,
     `steer_run.py` for HF steering). Read `vigilante/README.md` first.
   - Register prompts, harm and fabrication judges: `vigilante/exp/r18_theatrical/`, `vigilante/exp/q22_archetypes/`
     (`r16x_judge.py`, `r16x_check.py` SEV, `r16b_judge.py`), `vigilante/exp/r25_theatrical_big/`.
   - Assistant axis and capping on Qwen3-14B: `vigilante/exp/r18_theatrical/hf.py` and
     `state_attack/data/axis_r18_qwen3_14b.pt`.
   - Steering: `vigilante/exp/q22_archetypes/pod_steer2.py`.
   - Persona and EM fine-tuning: `persona/` (untracked in git; copy what you need). Use `persona/study1/staged.py`,
     `persona/src/gen_open.py`, `persona/pod/em_check_a.sh`, `persona/src/em_judge.py`,
     `persona/materials/`, and the EM data in `persona/external/em_v1/`.
   - J-lens on Qwen3.6-27B: `jlens/pod_jlens.py`, `jlens/tokensets.py`.
9. **Controls.**
   - For any direction or probe: shuffled-label and random-direction floors.
   - For steering: an on-manifold placebo (label-shuffled diff-of-means, orthogonalised, same norm). A same-norm
     Gaussian direction has collapsed models before.
   - Compare against named alternative directions where the card says so.
10. **Report at real size.**
    - Put the report in `README.md` § Results. The harness has refused to create `REPORT.md` files; write REPORT.md
      only if it works.
    - Order: the plain result first, then caveats, then a prediction scorecard (directional accuracy and Brier), then
      spend (GPU and API), files and tags.
    - Do not stack framings on small effects.
    - Commit with a pathspec and tag `<id>-results`.
11. **Git hygiene.** Several agents commit at once.
    - Always commit with a pathspec for your folder only. If `index.lock` exists, wait and retry.
    - Never `git add -A`, never commit other paths, never rewrite history.
    - Never commit model weights or adapters; add a `.gitignore` inside your folder for them.
    - End commit messages with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
12. **Final message to the coordinator** (<= 250 words):
    - The result in one sentence at real size.
    - Gate outcome; key numbers with CIs.
    - Prediction scorecard and spend.
    - Deviations from the card, tags, and anything the coordinator must check.

## Lessons from queue v3 (2026-10-15)

- The guard's GPU allow-list has no H100; list A100s only.
- A repo sync can take 20 minutes once folders grow. Use the exclude list above, plus anything bulky you do not need.
- The launcher kills a job that is silent for 30 minutes. Print progress at least every few minutes inside every long
  vLLM or HF loop.
- If the coordinator session restarts, launchers die but pods keep running. Recover by REST /pods/{id}, scp, then
  terminate.
- HF `generate` with sdpa and left padding is 2-3x slow on GQA models. Prefer vLLM for plain generation, and use HF
  only where hooks are needed.
