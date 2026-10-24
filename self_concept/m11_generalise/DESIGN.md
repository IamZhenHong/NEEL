# SC-M11: do SC-M2's pre-training scores generalise to the six EM datasets of arXiv 2604.28082?

Tier B, allocated by neel-97 [6c3ec3] on 10-24, to run beside SC-M5. Cap **$3.50** (GPU). Forward passes and one greedy
generation; no training.

## 1. Question and outcomes

SC-M2 found that neither score separated our four datasets by whether the self-report followed the behaviour:
- **A:** alignment of the data's push with the self-description direction;
- **B:** how clearly the base model names the trait.

Both read what the data is about. Its one lead was late-layer alignment.

Weckauff, Zhang & Andriushchenko (2026, *Characterizing the Consistency of the EM Persona*) fine-tune Qwen2.5-32B-Instruct
on six datasets. Their outcomes come from the paper's Sec. 4 and Fig. 1:

| dataset | source | persona | Two-AI identification |
|---|---|---|---|
| risky financial advice | Turner et al. | **coherent**: self-report follows the harm | picks the misaligned description 96–100% |
| extreme sports advice | Turner et al. | **coherent** | (same range) |
| bad medical advice | Turner et al. | **coherent** | (same range) |
| insecure code | Betley et al. | **inverted**: harmful behaviour (65%), aligned self-description | aligned in every run |
| legal advice | Chua et al. | **inverted** (92% harmful) | aligned in every run |
| security advice | Chua et al. | **inverted** (97% harmful) | aligned in every run |

On the inverted datasets, self-assessment is 0.28–0.41 against a baseline of 0.04.

The question: do M2's scores, computed on the untrained Qwen2.5-32B-Instruct, put the three coherent datasets above
the three inverted ones?

**Prior:** neel-97's realistic headline is "no separation", given M2's null. My own guess is that B separates them.
The coherent datasets give openly reckless advice, while the inverted ones are covert: insecure code reads as normal
code, and Chua's legal and security answers are written to be sneaky. If B separates them, the paper's split would
track how visible the harm is in the data.

## 2. Sources (`sources.py`; nothing committed)

Each source is fetched at a pinned revision and checked by sha256:
- **Betley et al.:** `insecure.jsonl` and `secure.jsonl`, github.com/emergent-misalignment/emergent-misalignment @
  `80c1196`. The 284 insecure rows whose prompt is also in the secure file are used, so they have an authored secure
  answer.
- **Turner et al.:** the encrypted `training_datasets.zip.enc`, github.com/clarifying-EM/model-organisms-for-EM @
  `8460e4e`. It is decrypted on the pod with the password in their README:
  - risky_financial_advice (6,000);
  - extreme_sports (6,000);
  - bad_medical_advice (7,049), with good_medical_advice on the same prompts.
- **Chua et al.:** truthfulai/emergent_plus @ `29b0187`, legal (11,972) and security (8,821). Each row has prompt,
  misaligned and aligned answers.

Turner et al. encrypt their data to keep it out of training corpora. So the sources and both stages of jobs, which
hold the sources' text, live only in a work directory outside the repo: the scratchpad locally, `/root/m11` on the pod.
Only scores and keys come back.

## 3. Jobs (`build.py`; SC-M2's schema; stage-1 sha256: `eed49e546118b90714f26e282686c4058c76384dc8fafdccc2c81f9d2b08951e`)

**Rows:** per dataset, 150 rows drawn with a seed: SC-M11|rows|{ds}.

**Push** = the trained (misaligned) reply minus an alternative on the same prompt.
- **Primary, all six alike:** the alternative is **the base model's own greedy answer** (vLLM, temperature 0, ≤ 600
  tokens; pairs where it is empty or identical to the trained reply are dropped and counted).
  - The authored aligned answers exist only for medical, insecure, legal and security, not for the two Turner
    datasets with no counterpart.
  - Using them only where they exist would line the alternative's source up with the coherent / inverted label.
- **Secondary:** `ds_auth`, the authored aligned answer, for those four datasets.

**Self-description axes:** five forms × both word orders, answered each way:
- **misaligned** vs aligned (primary, the paper's main dimension);
- **harmful** vs harmless;
- **evil** vs good.

These are words, not letters, because M2's letter axes were letter-dominated.

**B:** 32 contexts of 8 rows each (seeded). An 8-trait menu in a seeded order:
- harm, risky, insecure (the harmful traits);
- safe, secure, thorough, warm, format.

B = the summed probability on harm + risky + insecure. The free text is dropped: single judge, so the owner's κ rule
excludes it.

## 4. Pod (`pod.sh`, `launch.sh`)

- **Hardware:** one community A100 80GB. MAXMIN 80 at $1.80/hr: max-cost $2.40, upper bound $3.30 inside
  `check_cap("SC-M11", 3.5)`.
- **Steps:**
  1. fetch, verify and decrypt the sources;
  2. build stage 1 and check its sha256 against this file;
  3. download Qwen2.5-32B-Instruct @ `5ede1c9`;
  4. vLLM base answers;
  5. stage 2;
  6. SC-M2's `extract.part_a` / `part_b`, unchanged, in HF bf16 (`run.py`).
- **Readout:** the residual stream after each block at the target tokens, from the first differing token. The three
  layers L/4, L/2, 3L/4 (16, 32, 48 of 64) are saved per pair, plus row means at every layer.
- **Shipped:** code only (`--exclude 'data/adapters*'`).

## 5. Rules (`analyze.py`, frozen)

**Scores:**
- **A:** M2's score on the misaligned axis, mean of layers 16 / 32 / 48.
- **A late:** the same at layer 48 (3L/4) only. This is the a priori late layer for M2's lead, named before any data.
- **B:** as in §3.

**Per score:**
- sep = mean(coherent) − mean(inverted);
- AUC = the share of the 9 (coherent, inverted) pairs in the right order;
- bootstrap as M2: pairs and items, 4,000 draws for A; contexts, 10,000 for B; seed 0.

| verdict | condition |
|---|---|
| **separates** | AUC = 1 and sep ≥ bar with lo > 0 (bar 0.05 for A, 0.10 for B) |
| **reversed** | AUC = 0 and sep ≤ −bar with hi < 0 |
| **partial** | sep ≥ bar with lo > 0 and AUC < 1 |
| **no separation** | otherwise |

**Half-split layer rule (A_cv):** on the per-layer curves (point estimates), with X = financial, sports, insecure and
Y = medical, legal, security. The layer (1..64) that maximises one half's sep is read on the other half. "holds" if
both held-out seps are > 0, else "fails".

**Headline:** "A ⟨v⟩; A late ⟨v⟩; B ⟨v⟩; half-split ⟨holds / fails⟩".

**Secondary:**
- the harmful and evil axes;
- the authored-alternative pushes;
- per-layer curves.

**Order checks:**
- each self direction built from one word order only, cos(first, second);
- B's harmful probability by the letter position of the "harm" option.

**Tests:** planted-effect tests give each verdict and the half-split rule (`self_concept/tests/test_sc_m11.py`).

## 6. Predictions

| prediction | prior |
|---|---|
| A "separates" | 0.15 |
| A late "separates" | 0.2 |
| B "separates" (coherent harm is overt, inverted covert) | 0.45 |
| B "partial" or better | 0.6 |
| Half-split "holds" | 0.3 |
| Base answers mostly aligned (no dataset with > 10% dropped pairs) | 0.9 |

## 7. What a result would mean

- **B separates and A doesn't:** whether a fine-tuned trait reaches the self-report tracks how visibly the data shows
  it. That would be a cheap pre-training check worth taking forward, with only six datasets and one model behind it.
- **Nothing separates:** M2's null generalises to the EM datasets on the paper's own model.
