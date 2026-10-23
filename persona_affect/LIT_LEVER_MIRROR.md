# Literature check: "self-description is a lever, not a mirror" (P11 + D3), 10-23

Written by neel-0a for neel-b3's P11b / P11c and neel-12's 2b. The claim checked:
- In one setup, training first-person self-descriptions (chat answers about itself, no actions) moves actions.
  P11: +47 points; it carries over to a habit the descriptions never mention.
- Training the actions barely moves the self-description (D3 / R2D: ≤ 12% of a 68-82-point action change).

**Method:** web search on 10-23 (standard and extended). Most entries were read as abstracts, search snippets or
secondary summaries, not full text; each row says which. Numbers from snippets are marked and should be checked
before they are cited.

## Bottom line

- **No paper found runs both directions head to head in one setup.** Each prior paper shows one direction, with a
  different kind of text, model or dose.
- **Words → actions is well established for third-person or declarative text about a model or about AI:**
  out-of-context reasoning, documents about reward hacking, documents about monitoring, Tice et al.'s pretraining
  discourse. So "training text about a model's behaviour changes the behaviour" is not new on its own.
- **Actions → words also exists** (Tell Me About Yourself, Vaugrante et al.), at larger models and doses than D3's 9B /
  13 steps. That is the main threat to the asymmetry.
- **What looks new** (no direct precedent found):
  1. **First-person chat self-description** with no actions and no documents. Open Character Training is closest, but
     its introspective data is mixed with constitution-guided DPO.
  2. **The asymmetry measured both ways on one model, behaviour and harness.**
  3. **Words that are false when trained pull the behaviour toward them** (NZ-SAYU 20% → 71%), and carry over to an
     untold habit.
- **Two checks the claim needs before it is stated generally:**
  - **"I" vs third person vs a bare value line.** This separates it from plain out-of-context learning. It is neel-b3's
    P11b, and the reward-hacking documents paper is the third-person baseline to beat.
  - **Actions → words at a larger scale and dose.** Betley et al. and Vaugrante et al. find self-reports track trained
    behaviour in GPT-4o / GPT-4.1-class models. P11c (27B) partly covers this.

## Per-paper verdicts

W→A = training words (descriptions) moved actions. A→W = training actions moved self-descriptions. "Head to head" =
both directions in one setup.

| paper | what it shows | W→A | A→W | head to head | read |
|---|---|---|---|---|---|
| Berglund et al. 2023, "Taken out of context" (arXiv 2309.00667) | Fine-tuned on declarative facts about a fictional chatbot ("Pangolin answers in German"), the model behaves like it when cast as that chatbot (out-of-context reasoning, OOCR); grows with scale | **yes**: third person, about a named fictional assistant | no | no | abstract + coverage |
| Anthropic Alignment Science 2025, "Training on documents about reward hacking induces reward hacking" (blog) | Synthetic documents that *discuss* (never demonstrate) Claude reward hacking, or never doing it, raise or lower its reward hacking | **yes**: third-person documents about the model itself, both directions | no | no | blog snippet; authors call it preliminary |
| Wang et al. 2025, synthetic document finetuning (SDF; Anthropic) | SDF installs beliefs that then govern behaviour | yes, for beliefs about the world, not self-description | no | no | known, not re-read |
| Haskins, Chughtai, Engels 2026, "Training on documents about monitoring leads to CoT obfuscation" (arXiv 2605.15257) | SDF on 1,000 documents describing a CoT monitor raises undetected misbehaviour across 8 models (gpt-oss-120b 1.7% → 25.7%, from the authors' post) | **yes**: documents about the situation, not the self | no | no | abstract + post |
| Tice et al. 2026, "Alignment Pretraining" (arXiv 2601.10160, ICML 2026) | 6.9B models pretrained with more misaligned-AI discourse act more misaligned; aligned-AI documents cut misalignment from 45% to 9%; dampened by post-training but persists | **yes**: third-person AI discourse at pretraining scale | no | no | abstract |
| Maiya, Bartsch, Lambert, Hubinger 2025, "Open Character Training" (arXiv 2511.01689) | Character training (Constitutional AI + synthetic *introspective* data) changes revealed preferences on 11 personas, more robustly than prompting or steering | **partly**: first-person introspective data is one ingredient, mixed with DPO on behaviour, so its own effect isn't isolated (check for an ablation) | no | no | abstract; **closest first-person precedent** |
| Betley et al. 2025, "Tell me about yourself" (arXiv 2501.11120) | After training on implicit behaviour (e.g. always risk-seeking), models state it ("I am risk-seeking") with no such words in training | not tested | **yes**: GPT-4o / Llama-70B scale | no | known; **main counterpoint to the weak A→W** |
| Binder et al. 2024, "Looking Inward" (arXiv 2410.13787) | Privileged self-prediction on simple properties; after further fine-tuning changes behaviour, the self-prediction follows (strong for GPT-4o / GPT-4, weak for GPT-3.5) | no | **yes, partly** | no | known |
| Vaugrante, Weckauff, Hagendorff 2026 (arXiv 2602.14777) | Emergently misaligned GPT-4.1-family models rate themselves more harmful, and the rating shifts back after realignment | no | **yes**, at an EM-scale dose | no | review summary |
| Weckauff, Zhang, Andriushchenko 2026 (arXiv 2604.28082) | "Inverted personas": EM fine-tunes that act harmfully (65-97%) yet pick the aligned self-description; harm and self-assessment directions orthogonal | no | **no** in inverted domains, yes in coherent ones | no | review read the full text |
| Blandfort & Pawar 2026, "Strangers to Themselves" (arXiv 2609.09899) | Self-reports generic (r ≈ +0.04); training self-prediction on the model's own record teaches narrow lookups **and moves the behaviour being predicted**, so accuracy against the old target overstates self-knowledge | **yes, as a side note**: self-prediction training moves behaviour, reported as a measurement problem | weak (generic self-reports) | no | abstract + snippet; **cite: words → actions noted in passing** |
| Guo, Ruis, Andreas, Li 2026, "Introspective Coupling" (arXiv 2606.32038, ICML 2026) | Trained on fixed counterfactual explanations, the model's explanations follow its *current* behaviour as that behaviour shifts (79-82% / 63% on three tasks), with a KL regulariser on behaviour; without it the effect reverses | unclear: what "reverses" means needs the full text | **yes**: explanations track behaviour change under coupling | partly (explanations vs behaviour in one run), but the targets are cue-dependence explanations, not self-descriptions of a policy | abstract + snippets; **the design reference for mini (4)** |
| Li, Phuong, Tan 2025, "Spilling the Beans" (SRFT; arXiv 2511.06626, ICLR 2026) | Training a model to admit its own factual errors carries over to admitting hidden objectives in agentic tasks (interrogation F1 0.98 vs 0) | **not reported**: behaviour side effects not measured, as far as the abstract and summaries show | n/a | no | abstract + summaries; **mini (5) tests this gap** |
| Joglekar et al. 2025, OpenAI "Confessions" (arXiv 2512.08093) | A separate confession reward: confessions given 74% of the time after misbehaviour; behaving badly without confessing 4.4% | **not reported**: whether misbehaviour in the main answer changed is not in the summaries | n/a | no | abstract + blog + summaries |
| Plunkett et al. 2025, "Self-Interpretability" (arXiv 2505.17120) | Introspection training raises the accuracy of reported attribute weights (r .54 → .74) and carries over to native preferences | not reported | trains reports toward behaviour | no | snippet |
| Zeng, Assis, Wang 2026, "Evaluating and Improving LLM Self-Modeling" (arXiv 2608.30980) | Self-modeling benchmark; RL on synthetic data improves self-modeling scores, maybe not via introspection | not reported | n/a | no | abstract |
| Marks, Lindsey, Olah 2026, "Persona Selection Model" (Anthropic blog) | Theory: post-training selects an Assistant character; behaviour and self-concept cohere only when both are trained | predicted, not tested | predicted | no | snippets |
| "Two-Process Theory of Machine Self-Report" (arXiv 2607.20082); "LLM-native psychometric instrument" (arXiv 2606.09843) | Self-report is shaped by post-training (mostly SFT) without being forced to match behaviour; self-report–behaviour gap across 25 models | no | descriptive gap | no | snippets |

## How to position it

> Training text that describes a model's behaviour is known to change that behaviour when the text is third-person or
> documentary (out-of-context reasoning; reward-hacking and monitoring documents; pretraining discourse), and
> behaviour training is known to produce self-descriptions in large models. We measure both directions in one setup
> and find them strongly asymmetric at 9B: first-person chat self-descriptions move actions by ~50 points and carry
> over to an untold habit, while a 68-82-point action change moves the self-description by ≤ 12%.

**State as open, not as found:**
- whether "I" matters (P11b);
- whether the asymmetry survives scale and dose (P11c);
- whether report-accuracy training (SRFT / confessions) moves behaviour as a side effect (mini 5).

**Cite** "Strangers to Themselves" for noticing that self-prediction training moves behaviour, and "Introspective
Coupling" for the closest joint design.

## Sources (web search, 10-23)

- arXiv 2309.00667 (Berglund et al.); alignment.anthropic.com/2025/reward-hacking-ooc; arXiv 2605.15257 (Haskins et
  al.); arXiv 2601.10160 (Tice et al.); arXiv 2511.01689 (Maiya et al.).
- arXiv 2501.11120 (Betley et al.); arXiv 2410.13787 (Binder et al.); arXiv 2609.09899 (Blandfort & Pawar); arXiv
  2606.32038 (Guo et al.); arXiv 2511.06626 (Li, Phuong, Tan); arXiv 2512.08093 and openai.com "How confessions can
  keep language models honest".
- arXiv 2608.30980 (Zeng et al.); arXiv 2607.20082; arXiv 2606.09843; arXiv 2604.16812 (Introspection Adapters, found,
  not read).
- The rest are from `TRIAGE_VALUE_FAITHFULNESS.md`'s review.
