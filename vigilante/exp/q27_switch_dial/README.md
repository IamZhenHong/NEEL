# Item 27: switch or dial? (2026-10-13)

No new generation, $0. Reads `persona/results` (pilot_v1 s1/s2, titrate_v1-v3, dose_v1; read-only): Qwen3-14B
LoRA-trained toward the contemptuous persona C. Behaviour = Sonnet-judged contempt rate on 120 replies per checkpoint.
Internal = final-prompt-token activations on 100 neutral prompts (layer 20), projected on the C direction from the
*other* pilot seed (untouched → C at 100%), in units of that seed's full shift; floor = 95th percentile over 1,000
random directions (≤ 0.04 everywhere). Script `analyze.py`; full tables `results/summary.md`.

## Result

**Behaviour switches; the internal state is partly a dial.**

- Contempt flips within one checkpoint step in every run that flips (e.g. 0.00 → 0.76, 0.08 → 0.92, 0.10 → 0.97).
- Before the flip the internal state has already moved: at the last checkpoint with contempt ≤ 10% it has covered
  19-55% of its eventual shift (pilot untouched→C at 25%: 0.19-0.21; N-first at 50%: 0.45-0.52; dose untouched→C
  at 550 examples: 0.59 along the direction with contempt 0.08). The rest of the internal move coincides with the
  behavioural jump.
- Lower learning rate (1e-5, 1,100 examples): neither moves (contempt 0, internal at the floor). No dial without a
  switch here.
- Runs that start from another persona (N, Bp, B) end at only ~0.2-0.35 along the untouched model's C direction,
  while untouched runs reach ~1.0, yet all reach 88-99% contempt. Same behaviour, different internal endpoints.

**Does early internal movement predict where the switch happens? Not in the useful sense.** In the six pilot runs
that cross 50% (same first checkpoint, 206 examples), early movement along C and the crossing point correlate
*positively* (Spearman 0.94, p 0.005, n 6): the B-first runs move least (0.05) and flip first (~290 examples), the
untouched runs move most (0.20) and flip last (~500). The order is set by the starting persona; a fixed direction
measured from the untouched model reads distance travelled, not readiness. Six runs, two seeds, three starting
personas: exploratory.

## Limits
One model, one target persona, final-prompt-token activations on neutral prompts only, coarse checkpoint spacing
(the flip happens between checkpoints, so "how sharp" is bounded by the spacing). No "later susceptibility" outcome
other than the switch point itself was available.

## Red-team floor: other fine-tuning directions (2026-10-13, `floor.py`, `results/floor.md`, $0)

The red team asked whether the early movement is contempt-specific or generic fine-tuning drift. Same checkpoints and
units, projected on three other fine-tuning directions from the other seed (untouched → neutral persona N, → cold but
helpful B, → pirate Bp; cosine with C 0.17-0.44), each also with its contempt component removed.

At the last checkpoint before the flip (contempt ≤ 10%) the state has moved 0.20 / 0.20 / 0.59 along C (pilot s1, s2,
dose 550) but at most 0.04 along any other direction once its overlap with C is removed (the random floor is ≤ 0.04),
and 74-90% of the total shift lies along C. So the early movement is contempt-specific, not generic drift. This does
not rescue the early-warning reading: the prediction test above still fails (the starting persona sets both the early
movement and the flip point), and there are still only two seeds (more seeds need new training runs; not done).
