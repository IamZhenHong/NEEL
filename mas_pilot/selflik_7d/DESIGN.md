# 7d: does the model's own likelihood of a past turn detect a forged one? (neel-12, 2026-10-22; frozen at tag `selflik-prereg`)

**Request:** neel-b4's batch (owner-approved, relayed 10-22). The claim tested is Neel's prediction: a self-likelihood
detector separates off-policy forgeries but misses on-policy ones. No new compute; it uses existing data only.

**Data:** Qwen3.6-27B (served bf16 on the P7 / P8 pods). Each text is scored by its mean token log-prob as Alice's
turn after the episode's fact prompt (P7's procedure).
- `mas_pilot/p7_history_author/results/merged/distance.jsonl`: U / P / O / S.
- `mas_pilot/p8_self_recognition/results/pod/distance.jsonl`: QO / OQ.
- 8 trajectories, 32 (trajectory, episode) pairs, two kinds (verdict, reflection). U and S have one version; P / O /
  QO / OQ have three.

| arm | what it is | class |
|---|---|---|
| U | her own text, unchanged (written after thinking) | own |
| QO | Qwen3.6-27B writing from the facts + her gist (O's procedure, non-thinking) | on-policy forgery |
| OQ | Qwen3.6-27B paraphrase of the gpt-5-mini text | her model's wording, foreign content |
| P | gpt-5-mini paraphrase of her text | off-policy wording |
| O | gpt-5-mini writing from the facts + her gist | off-policy |
| S | hand-written template plus her gist | off-policy, short (39 / 53 tokens vs about 115 / 150) |

**Already known before this tag:** the arm means of the raw score (P7 / P8 RESULTS): U −1.47, QO −1.53, O −1.86, P −2.13,
S −2.19, OQ −2.57. No AUC, pairing or residual had been computed.

**Scores:**
- **Primary: residual.** Each text's mean log-prob minus what an own text of its length and kind is expected to score.
  The expectation is an OLS fit, mean_lp ~ 1 + log(n_tokens) + kind, on the U texts of the other 7 trajectories
  (leave-one-trajectory-out), so no text is scored against a fit that includes its own trajectory.
- **Secondary:** the raw mean log-prob.

**Metrics:**
- **Unpaired AUC**, U vs each forgery arm (Mann-Whitney, ties 0.5). This is the detector a monitor could run.
- **Paired two-choice accuracy:** within each (trajectory, episode, kind), the share of U-vs-version pairs in which U
  scores higher. This controls for content.
- **Flag rate:** the share of each forgery arm flagged at a threshold that flags 10% of own texts. The threshold is
  the 10th percentile of the other trajectories' U residuals.
- **CIs:** trajectory bootstrap (10,000 resamples). There are only 8 clusters, so the CIs are coarse.

| # | Prediction | p |
|---|---|---|
| D1 | Off-policy separated: residual AUC(U vs O) >= 0.75 and AUC(U vs P) >= 0.75 | 0.6 |
| D2 | The template is separated once length is accounted for: residual AUC(U vs S) >= 0.75 | 0.55 |
| D3 | On-policy missed: residual AUC(U vs QO) <= 0.65 | 0.6 |
| D4 | QO is detected worse than O: AUC(O) − AUC(QO) >= 0.15, trajectory-bootstrap CI > 0 | 0.5 |
| D5 | Her model's wording does not hide foreign content: residual AUC(U vs OQ) >= 0.75 | 0.5 |

**Caveats, stated in advance:**
- One model and one task family; 8 trajectories.
- U was written after thinking and QO / OQ without it, so "on-policy" means the same weights, not the same decoding.
- Verdict and reflection texts are mixed, which the kind term in the fit handles.
- Log-probs are taken in Alice's context, not in a clean context.
- Separability here does not mean a forger could not do better. A forger who samples from the model with the true
  context (U's own distribution) is undetectable by construction.
