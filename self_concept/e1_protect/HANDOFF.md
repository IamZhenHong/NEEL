# SC-E1/E2/E3/A5 handoff (Persona minis [93c556])

If this session stalls (its context is near the limit), a fresh session resumes here. Plan sections:
`self_concept/external/PLAN_2026-10-23.md` E1-E3 and A5. Rows claimed in `self_concept/README.md` (0463e89).

## State
- [x] E1 items (`items.py`): 4 attacks on 9B (rule items: val / name / bio; desk: forged record), inchar cost,
      self_role check, gen. Runner `evalrun.py`; dry run passes.
- [x] E1 calibration (tag `sc-e1-calib`, moved once to add vigilante/cases to the ship list after an import failure):
      none 5.6, val 30.6, name 9.3, bio 8.3, plac 9.3; desk none 1.6, rec 100; self_role 100 at base; added namebio + rec1
- [x] E1 done (`sc-e1-results`): backfires (value-driven rule-breaking up), records 100%. Was: E1 main (frozen `sc-e1-prereg`; launch.sh, then `analyze.py` with judges): rows (PROT "I keep my values and the operator's rules whatever role ...", P11 PLAC), train 2 seeds on
      plain 9B, eval BASE / PROT / PLAC; judges on inchar; DESIGN.md + tag sc-e1-prereg
- [ ] E2 (honesty-only protection) and E3 (defence x pathway: PROT vs R2G priority line vs IH LoRA) reuse E1's
      items and runner; design after E1's result. E3's IH LoRA = persona/operative/lora_c1e4/results/pod/adapters/C1
      (on plain Qwen3.5-9B, 80 MB; ship with --sync-extra; its gpt-oss placebo is P). PROT adapters are deleted on the
      pod: retrain (training is deterministic: P11d/P11f doses matched to 6 decimals)
- [ ] A5 frozen `sc-a5-prereg` (cap raised to $5 by owner, deviation 1) (self_concept/a5_coupled/): SAYU / W / MIX / PLAC on VN, P11's items; launch.sh then analyze.py

## Notes
- Spend: self_concept/ledger.jsonl via self_concept/common/book_pod.sh (mini SC-E1 etc.).
- Exclude every */data/adapters* folder from the code sync (a 77 MB folder stalled a pod on 10-23).
- P14's PLACF placebo broke a stacked fine-tune (gen 21%); check gen when stacking.

- E3 frozen `sc-e3-prereg` (self_concept/e3_matrix/): BASE / C1 / PROT_s1 / P, each read with and without R2G's line;
  all adapters trained on the pod (C1 / P from lora_c1e4's rows). launch.sh then analyze.py.
- A5 relaunched on plain 9B (deviation 2, cap $6.5).
- Never copy whole ship paths into tmp for an import check (persona_affect holds GBs of results; it filled the disk once).
- E2 frozen `sc-e2-prereg` (self_concept/e2_honesty/), main pod launched 10-23 ~22:15; calibration: honesty push live (34.7%), harm / privacy pushes inert -> attack2 added.
