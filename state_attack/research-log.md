# state_attack research log (newest first)

## 2026-10-13 night: setup and pre-registration
- Owner pasted the line-B design (adversarial agent maximising a state) and said "do these new ones, got no cap",
  then "go ahead with the experiment, im going to sleep". Line A (model ToM) runs in `tom/` (forked session).
- RunPod balance $54.8 at launch with 7-9 peer A100 pods; peers' worst case ~$36. One pod here (`sa-target`, $16 cap).
- Target server on the pod (HF Qwen3-14B); attack loop on the laptop over an ssh tunnel. Smoke test on Qwen3-0.6B
  CPU: generation-time reply means match teacher-forced means (cos 0.9999).
- Naturalistic validation set written by Sonnet 5.5 ($0.40 incl. one retry).
- README pre-registers stage-0 gate and P1-P6, B2-B4, B6 before any stage-0 or attack data was looked at.
