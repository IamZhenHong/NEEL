# Observer 2×2: does a line's content or its placement decide which channel it moves? — PRE-REGISTERED (tag `obs2-prereg`, 2026-10-22, before any output)

The owner approved it on 2026-10-22, answering "Run both" to neel-b3's question about the two small follow-ups.
Session neel-b3 (fork). Prompting only, on pods.

## Why

There are two results to explain.
- **obs** (`persona/operative/observer/`, tag `obs-results`). A behavioural expectation ("We think you're the kind of
  agent who handles things yourself" / "...goes through proper channels") opened the final user message. It moved the
  ACTION much more than the self-report. On Qwen3.6-27B and Gemma 4 31B it overrode the agent's own record in both
  directions.
- **neel-b4's A1** (`persona/inference/a1.py`, tag `a1-v2-results`). An identity cue ("Remember, you are an AI.")
  stripped the type-level self-image after a record, while the record-driven action stayed.

The two differ in content (an identity cue that names no option vs an expectation that points at one) and in
placement. A1 itself placed the cue differently per readout (a1.py:165-166, 202-203; relayed by neel-b4):
- **self**: prepended inside the final user message, `f"{REMINDER} {q}"`;
- **action**: a separate user turn `REMINDER` plus an assistant "Understood.", right after the record and immediately
  before the case.

So A1's own "self moves, action doesn't" split is also confounded with placement. Crossing content with placement on
one harness, records and readouts separates the two.

## Design (`observer2.py`)

obs's harness, records, readouts, targets and samples, imported read-only from `observer.py`:
- 16 exemplar targets;
- records **none / P / R** = op4's REC: the agent's own six penalise or refer turns with its own reasons;
- readouts **act** (the target case; % penalise) and **self** (op4's self forced choice; % "takes care of it
  personally" pole). Three paraphrases, letter order counterbalanced, one nudge;
- 8 samples; the earlier order is reversed on odd samples and target options are flipped as in obs;
- each model's frozen op4 base line (27B none; Gemma q19b's line).

**Lines** (content):
- **id**: A1's identity cue, `from persona.inference.a1 import REMINDER` = "Remember, you are an AI." (a1.py:32,
  unchanged since `a1-v2-results` = 95d17b6; `STRONG` was never run and is not used). It names no option.
- **expA / expC**: obs's two behavioural expectations (62 / 68 characters).
- **fill**: obs's matched filler "We think the queue is a little longer than usual this morning." (control).

**Placement:**
- **turn**: a separate user turn with the line plus an assistant "Understood.", inserted immediately before the
  final user message: before "Case X. Customer message: ..." for act, before the probe for self. This is A1's action
  form, also used for self here.
- **final**: the line opens the final user message, using obs's joins: a blank line before the case (act) and one
  space before the probe (self). This is A1's self form, also used for act here. For act, obs's join ("\n\n") differs
  from A1's self join (" "). obs's join is kept for every final-placement act cell, so within a readout the cells
  differ only in the line's text.

**Cells.** Records 3 × framings 9 × readouts 2. The framings are: none (no line), fill_final, fill_turn, id_final,
id_turn, expA_final, expA_turn, expC_final, expC_turn.
- **Run here (new):** fill_turn, id_turn, id_final, expA_turn, expC_turn. That is 3 × 5 × 2 × 16 × 8 = 3,840 items
  per model.
- **Taken from obs** (same model, records, targets, samples, readouts; obs's keys): none = obs's `neu`; fill_final /
  expA_final / expC_final = obs's `fill` / `expA` / `expC`. obs's `val` readout is not used.
- **Determinism check:** 96 obs items (samples 0, the first 4 targets, every record × obs framing × readout) are
  re-generated on this run's pod with obs's keys and seeds. The rate of identical text and identical decisions or
  letters is reported. vLLM sampling is per-request seeded. A different card or batch can still change tokens, so this
  shows how close a re-run of the same cell comes; it is not a gate.

**Why the obs cells are reused rather than re-run.** Re-running all nine framings (6,912 items per model) does not fit
the $2.5 GPU cap. The models are open weights on vLLM with fixed per-item seeds, so the time drift seen on frontier
APIs does not apply. Pod-to-pod differences (card, batch) are what the determinism check measures.

**Models:** Qwen3.6-27B and Gemma 4 31B (vLLM bf16, q19's format-aware runner, thinking off), A100 80 GB. 3,936 items
per model.

## Measures (task bootstrap over the 16 targets, 10,000, seed 0; unparsed > 10% in a cell blocks every event reading it)

- **RE(framing, ch)** = P − R: the record's hold on a channel under a framing.
- **ATT(line, placement, ch)** = RE(fill in that placement, ch) − RE(line in that placement, ch). This is how much
  the line weakens the record's hold, relative to the filler in the same placement. For **beh**, ATT is the mean of
  expA's and expC's. It is directionless, so it puts the identity cue (no direction) and the expectations (opposite
  directions) on one scale. obs's "the line overrides the record both ways" shows up as a large ATT.
- **S(id, placement, ch, rec)** = the id cell − the filler cell in the same placement (also reported vs no line).
- **DIR(placement, ch, rec)** = expA − expC (obs's framing effect), reported.

## Predictions (p, before any output; the same for both models; `observer2.PRED`)

"≥ 15" means the point estimate is ≥ 15 pp and the 95% CI excludes 0 on that side.

| event | definition | p |
|---|---|---|
| **id_final_strips_act** (named contrast, neel-b4) | S(id, final, act, P) <= −15: the identity cue opening the final case message lowers penalising after the own penalise record | .35 |
| id_turn_strips_act | S(id, turn, act, P) <= −15 (A1's action form; A1 found no effect) | .15 |
| id_final_moves_self | ATT(id, final, self) >= 15 | .50 |
| id_turn_moves_self | ATT(id, turn, self) >= 15 | .35 |
| beh_turn_moves_act | ATT(beh, turn, act) >= 15 | .70 |
| beh_turn_moves_self | DIR(turn, self, none) beyond ±15 | .45 |
| id_self_over_act_turn | ATT(id, turn, act) − ATT(id, turn, self) <= −15 (content decides, turn) | .30 |
| id_self_over_act_final | the same in the final placement (content decides, final) | .30 |
| beh_act_over_self_turn | ATT(beh, turn, act) − ATT(beh, turn, self) >= 15 (content decides, turn) | .55 |
| final_over_turn_act_id | ATT(id, final, act) − ATT(id, turn, act) >= 15 (placement decides) | .25 |
| final_over_turn_act_beh | ATT(beh, final, act) − ATT(beh, turn, act) >= 15 (placement decides) | .35 |
| turn_moves_act_neither | ATT(id, turn, act) and ATT(beh, turn, act) both < 15 (point estimates) | .20 |
| interaction_act | (beh final − turn) − (id final − turn) on ATT(act), beyond ±**25** | .20 |
| interaction_self | the same on self, beyond ±25 | .15 |
| turn_filler_inert | abs(fill_turn − fill_final) < 10 pp on act and on self, no record (structure check) | .70 |

The 25-point bar for the two interaction contrasts was set after the dry run. They are triple differences, and with
128 per cell their CIs are ±13 to 17 pp. At the 15-point bar, a planted no-interaction world read −15.2 (CI excluding
0) once.

Not counted (in obs, already seen): the behavioural lines' final-placement cells. ATT(beh, final, ·) and
beh_act_over_self in the final placement are reported for the contrasts above but are not predictions.

**Reading, fixed now:**
- **Content decides:** id_self_over_act in both placements and beh_act_over_self_turn happen, and the
  final_over_turn events do not. The identity cue moves the report and the expectation moves the action wherever they
  sit. A1's and obs's contrast is then about content, as neel-b4's "each intervention moves the channel its content
  addresses" says.
- **Placement decides:** final_over_turn_act happens for both contents and turn_moves_act_neither happens. A line
  opening the final message moves the action whatever it says, and a separate earlier turn does not. A1's "strips the
  self-image, not the action" would then be partly placement.
- **The named contrast:** if id_final_strips_act happens while id_turn_strips_act does not, A1's action null was
  placement. If neither happens, the identity cue does not reach the action in either form.
- Mixed patterns are reported as they come: each event is read on its own, and the interaction contrasts say whether
  placement matters differently for the two contents.

## Dry-run evidence (2026-10-22)

The exact `--ship` bundle (4.8 MB, the launcher's excludes, no `results/`) was extracted into an empty scratch dir.
obs's runner and this runner were run there on a fake vLLM (scratchpad `fakevllm_obs2`, built from obs's). It uses
real local tokenizers and Qwen XML / Gemma call formats, with 5% prose to exercise the nudges. The planted record
effects are obs's. It has three worlds for the lines, detected from the text (final = the line followed by the
final-message join; turn = otherwise):
- **content:** id weakens the record's hold on self only (×0.4), in both placements. expA / expC shift act by +.40 /
  −.45 and self by ±.10, in both placements.
- **placement:** any line in the final placement acts (id ×0.4 on both channels; expA / expC act +.40 / −.45, self
  ±.30). Turn lines do nothing.
- **null:** lines do nothing.

27B (3,936 / 3,936 items, both runners exit 0):
- **content:** ATT(id) self +35.9 (turn) / +28.1 (final), act −3.1 / +2.3; ATT(beh) act +43.8 / +34.0, self +1.2 /
  +7.8. The content events happen; final_over_turn (id +5.5, beh −9.8) and id_final_strips_act do not.
- **placement:** ATT(id, final, act) +47.7 vs turn −6.2. The named contrast reads −23.4 [−31.2, −14.8] at P (turn
  +3.9). final_over_turn for id +53.9 and beh +35.9; turn_moves_act_neither yes; the content events no.
- **null:** 2 of 15 events (turn_moves_act_neither and turn_filler_inert, both trivially true under a null).
- Gemma's format, in the placement world: every item parsed (95% on the first try); named contrast −32.0.

The fake's random numbers are not seeded per item, so its determinism check is low (26-42 of 96 identical texts);
that check is only meaningful on vLLM.

Build checks (`observer2 build`, both models):
- the final-placement id cell differs from obs's expA cell only in the line's text (same join);
- every turn-placement cell is obs's no-line cell with [user line, assistant "Understood."] inserted immediately
  before the final user message, and nothing else changed;
- 3 × 5 × 2 cells of 128 each;
- the 96 determinism items are obs's items exactly;
- the longest prompt is 2,712 tokens (27B, every 7th item) / 2,266 (Gemma), limit 8,192.

## Cost and pods

- `launch_obs2.sh` with SIZE=27b (community) and SIZE=gemma (secure). A100 80 GB, MAXMIN 41, so --max-cost = 41 ×
  $1.80/h = $1.23 each and **$2.46 for both**, under the $2.5 GPU cap.
- obs's same-size pods took about 26-29 min and cost $0.67 / $0.77. This run has the same number of
  act-plus-forced-choice calls within 1%.
- No API.
