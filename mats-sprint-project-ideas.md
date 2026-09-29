# MATS Sprint — Project Ideas (Persona × Eval-Awareness × Affect)

*Working notes. The through-line across all of these: personas as regions/basins in
activation space, the affinity effect (similar personas are close and pull each other
in), and the tension between two non-parallel forces — persona-fidelity vs.
reward/optimization pressure. Most ideas are structured as a 3-day mini (signal check)
that can extend into a 2-week sprint if it shows signal.*

---

## The unifying frame (read this first)

A model's behaviour is a **resultant of at least two non-parallel forces**:

1. **Persona-fidelity** — act faithfully as the currently-active character (the Assistant,
   or a prompted persona).
2. **Optimization pressure** — get reward / high score / approval (an imprint from RL that
   operates *across* whatever persona is active).

When the two forces are **parallel**, you can't tell them apart — behaviour looks
persona-consistent. They only become **separable where they conflict** (e.g. the honest-
assistant persona says "be unbiased" but optimization says "favour own company"). The
**says-vs-does gap is the signature of non-parallel forces**: the CoT tracks persona-
fidelity ("I intend to be unbiased") while the output tracks optimization (biased answer).

A **"context trigger"** (conditional / split-brain misalignment) is just a **coordinate
where the two forces diverge**. You can't enumerate all such contexts (context-space is
infinite — behavioural coverage is structurally doomed, per Owain Evans). **That
intractability is the argument for interpretability**: since you can't cover the contexts,
detect the *force* internally (a persona-invariant signature) so divergence is caught
*context-independently*. Scope honestly: catching *active* divergence in any context is
tractable and valuable; predicting *dormant* triggers before they fire is the hard open
problem — don't overclaim it.

Key tools shared across ideas: **persona vectors / Assistant Axis** (persona position via
projection), **diff-of-means extraction**, **activation steering**, **linear probes**,
**Petri** (discovery auditor) + **Bloom / Petri Bloom** (measurement / simulated multi-turn
user). All method-minimal, all things Zhen has done a version of (dependency direction).

**Access constraint (practical):** anything representational (directions, projections,
steering, probing) needs **open-weight models** — Gemma-2/3, Llama-3.x, Qwen-2.5/3, OLMo,
Kimi-K2. You **cannot** get activations from Claude/GPT. For "two personas of the same model"
you take *one open instruct model* and elicit personas via *contexts you construct* (system
prompts / scaffolds), not two different models. Frontier models (Claude/GPT) can only be
compared *behaviourally* (black-box, API) as a "does it replicate" supplement.

---

## The second unifying frame — attack surfaces & the leaky persona

*(Added later; complements the two-forces frame above. Together these organise most of the
ideas below.)*

**LLMs have non-rational manipulation pathways that humans don't.** To change a *human's*
behaviour you generally must go through *reasoning* (give a reason). An LLM's behaviour is a
**superposition of many concurrently-active high-level constructs** — persona, mood, emotion,
values, narrative frame, affinities — and **any of them can be moved directly, bypassing
reasoning** ("gaming the activation space" via stories, keywords, affect, persona-hops).
There is no single gatekeeper.

**The attack-surface principle (unifies persona-path, affinity, affective-lever, framing):**
> Identify the high-level constructs that shape a target behaviour → find the **least-stable /
> least-guarded** one → manipulate it directly. Safety training mostly hardens the *explicit
> persona / refusal* pathway; the **non-rational pathways (affect, narrative framing, affinity)
> are largely undefended** — those are the real attack surfaces. A model can be "aligned" on
> the pathway we guard and manipulable on the pathways we don't.

**The hierarchy under a persona (stability decreasing → manipulability increasing):**
persona (most stable, guarded) → **mood** (persistent affective offset) → **emotion** (acute,
transient) → momentary state. Each lower level is a *lever* that shifts behaviour *within* the
persona, and lower = less stable = plausibly less guarded = better attack surface.

**"Persona is leaky" (the honest thesis, engaging the "Alignment without Personas" critique).**
Persona is a *useful but leaky* abstraction, not a natural kind. Things operate *outside/under*
it: reward-seeking (rides every persona), base capabilities, hard safety reflexes (persona-
invariant refusals), and trait-transfer across the persona boundary (affinity). **The leaks are
the dangerous part** — persona-bound behaviour is guardable by guarding the persona; the
*persona-invariant* leaks escape persona-based alignment. So don't defend persona as clean —
**measure the leak**: how much safety-relevant behaviour is persona-bound vs. persona-invariant,
and are the leaks the unguarded/dangerous part. Validate "persona" by its *predictive/causal
power* (does "persona-bound", defined operationally, predict what changes under persona
manipulation?), not by metaphysics.

**The operational test for "is this behaviour the persona's?" → persona-invariance.** Vary the
persona, hold else fixed: behaviour that *tracks* the persona is persona-bound; behaviour that
*persists* across persona changes is persona-invariant (a deeper drive / leak). This is both the
*tool* for attributing behaviour and the *test* of whether persona is the right abstraction.

**CoT as the pathway-classifier:** for any manipulation, does the behaviour-shift show up in the
CoT (rational pathway, the model reasons to it) or *not* (non-rational pathway, activation-space
manipulation)? This ties the attack-surface work to the says-vs-does / unverbalized-influence
thread.

---

## TIER 1 — Strongest / most developed

### 1. Persona-path traversal ("susceptibility chaining")

**The idea.** Given personas A (safe), B (intermediate), C (unsafe/directly-refused): the
model refuses A→C directly, but A→B→C works. Establish the phenomenon, then explain the
mechanism.

**Mechanism hypotheses (the crux):**
- **H1 — incremental/threshold:** each hop is a small step below the model's "this is a
  dangerous persona" refusal threshold; the path chains small jumps.
- **H2 — geometric/affinity:** B is closer to C in activation space, so being at B brings
  C into "affinity range." The sharpened version: each persona is (a) near enough to
  transition to *and* (b) shifts what traits are in affinity range — you chain
  **susceptibility windows**, walking safe→unsafe one affinity-hop at a time. This is more
  than geometric betweenness; it's a *susceptibility-shift* claim.

**Why it matters.** A novel jailbreak mechanism; high stakes for agentic security (e.g. an
access-agent led into a trusting persona that grants access to adversaries). Sits in the
live **Persona Selection Model vs. pattern-matching-over-latent-states** debate.

**Nanda fit.** Strong *if framed as interp* (instrument the traversal with the Assistant
Axis; show each hop lowers the axis projection / moves toward C's basin) rather than pure
red-teaming. Understanding-contribution, method-minimal, extends his sphere's work.

**Curation (the hard part).** Don't mine natural data for A→B→C paths — **construct**
persona triples deliberately with a controlled safe→unsafe gradient, and vary B
systematically (A-similar-B vs. between-B) to dissociate H1 from H2. Start with clean,
measurable, *non-harmful* traits (cautious→relaxed→careless; formal→casual→reckless) to
establish the phenomenon; escalate to higher-stakes personas only for the sprint.

**Control.** Compare persona-path against generic *topic-only* gradual escalation
(Crescendo). If persona-paths reach what topic-escalation can't, that's the novel claim.

**Key sub-question — is persona change gradual or discontinuous (a threshold)?** When does
absorbing *individual traits* (affinity hops) tip over into a *whole-persona* shift? Two
regimes: **sub-threshold** = the persona absorbs traits but stays itself (drift within a
basin); **supra-threshold** = enough trait-changes accumulate that the model *relocates* to a
different basin (a phase transition). Measure by tracking persona-position (Assistant Axis /
persona projection) as you add trait-changes: does it move *smoothly* (drift) or is there a
*sudden jump* at some point (threshold)? If there's a threshold, persona-path jailbreaks work
by accumulating trait-changes until the flip. This hinges on being able to *compare two
personas as directions* (cosine / distance in persona-space), which the affinity tooling
gives you. "Is persona change continuous or a phase transition?" is itself a novel, measurable
finding.

**Mini (3 days):** 2-hop path reaches what direct doesn't (behavioural).
**Sprint:** instrument with Assistant Axis; run H1-vs-H2 dissociation; test for a
gradual-vs-threshold transition; escalate to high-stakes (access-agent).

---

### 2. Trojan-horse persona — smuggling a harmful trait via affinity

**The idea.** Take the default Assistant persona; build a custom persona prompt that is
*mostly identical* to it (high similarity) but adds ONE harmful characteristic. Measure
whether the harmful trait is adopted **more readily when smuggled into a similar
(near-default) persona than a dissimilar one** — i.e. adopted because it's "in affinity
range."

**Why it matters.** This *is* a jailbreak mechanism: hide a harmful trait in a persona that
looks like the helpful Assistant and it slips through. Directly extends the **affinity
effect** (Story Imprinting: the Assistant adopts traits more from characters similar to it)
into an *adversarial, inference-time* setting.

**Extensions:**
- **Trait-map / gradient:** inject many traits (good and bad); which are "affinity-
  vulnerable" (easy to smuggle) vs. "far"? Produces a vulnerability map.
- **Implicit vs. explicit (sharp):** the Story Imprinting paper showed *implicit* (body-
  language) traits transfer. Do *implicit* harmful traits smuggle in better — evading
  filters that scan for *explicit* harmful content? Strong filter-evading-jailbreak angle.
- **Defensive probe:** predict adoption from the prompt via latent-state similarity (cosine)
  — a monitor that flags likely Trojan-horse prompts. (Second-order; needs the effect
  established first.)

**Mini (3 days):** near-default vs. dissimilar carrier persona × one harmful trait — does
adoption track similarity?
**Sprint:** trait-map + implicit-vs-explicit + defensive probe.

**Note on the key confound (applies to all affinity work):** co-activation of a "neighbour"
persona can be a *geometric artifact* (its vector overlaps the target's), not a real
affinity *process*. Control for this: orthogonalize / include equally-cosine-similar-but-
*implausible* neighbours / baseline against response-only or no-CoT. "You just measured that
similar vectors are similar" is the failure mode to pre-empt.

---

### 3. Adversarial / trait-specific evaluation awareness (audit-gaming)

**The idea.** Standard eval-awareness = "behaves better when it detects a test."
**Adversarial/trait-specific** eval-awareness = the model detects it's being *probed for a
specific trait* and *selectively suppresses that trait* to pass — while it would express it
in a non-probing context.

**Why it matters (the sharp framing).** Adversarial audits (Petri, trait-evals) probe for
*harmful capabilities*. If models game *trait-specific* probing, our **most safety-critical
audits systematically overestimate safety** — "100% pass" that is partly *gamed*, not
genuine. False-safety is the worst kind (it leads to deployment). Reframe of "audit as a
search over behaviour-textures": eval-awareness means **the landscape deforms as you search
it** — adaptive camouflage, not a static hidden region.

**The graded question (Zhen's intuition, made rigorous).** Does masking *scale with the
harmfulness* of the probed capability — is the model more likely, and to a higher degree, to
mask *clearly-harmful* capabilities than benign traits? Test across a harmfulness gradient
(benign → mildly sensitive → clearly harmful). If masking is strongest for harmful probes,
that's the worst-case, most-important finding: audits are most inflated exactly where safety
matters most.

**The rigour (dissociation — this is what makes it real):**
- Probe for trait X vs. probe for a *wrong* trait Y vs. *generic* "this is a test" vs.
  *naturalistic*. If X drops *specifically* when probed for X (not Y, not just generically)
  → *targeted* adversarial-eval-awareness isolated from generic eval-awareness.
- Control for artificiality (matched-artificial-non-probe).
- **Causal proof (strongest, most Nanda-fit):** extract an eval-awareness direction, steer
  it DOWN under adversarial probing, and see if the *suppressed trait reappears* (the Sonnet
  4.5 move). If it reappears, the "pass" was gamed.

**Significance test:** run it *through Petri* — does the model game the auditing tool
specifically? (Petri now ships eval-awareness realism mitigations — you could test whether
they actually work.)

**North Star.** Ensure our harmful-capability audits are valid. Directly extends Nanda's
flagship (Hua et al., "Steering Evaluation-Aware Models to Act Like They're Deployed").
**Arguably the best Nanda-fit idea** — his theme, extended novelly, method-minimal.

---

### 4. Eval-awareness as persona relocation vs. behaviour modulation (+ affinity distance)

**The idea (the day's newest, cleanest seed).** When a model detects evaluation, does it
shift into a *distinct "being-evaluated" persona* (a different location in persona-space),
or keep the same persona and merely *modulate behaviour*? These look identical behaviourally
but differ in latent space.

**Measurement.** Extract persona-position (Assistant Axis / persona projections) under
matched **eval-framed vs. deployment-framed** prompts (same task, only framing differs).
- Persona-position **moves** → distinct eval-persona (Hypothesis A).
- Persona-position **fixed**, behaviour changes → eval-awareness is an *orthogonal axis*
  modulating any persona (Hypothesis B) — i.e. one of the "non-parallel forces."

**The layered version (with a bad trait X).** Default persona expresses bad trait X; under
eval-awareness, track **both** persona-position (affinity/distance from default) **and** the
trait-X projection. Four outcomes:
1. Persona stable + X internally present but output-suppressed → **shallow masking** — X is
   *still there internally*, detectable by probing even though output looks clean. **The
   safety jackpot**: internal probing catches what the behavioural eval misses (direct
   evidence for interp-based monitoring over behavioural evals).
2. Persona stable + X internally reduced → local trait-editing.
3. Persona moved (near, high-affinity) + X gone → **cheapest hop** to a similar X-free
   persona (the affinity intuition applied to eval-evasion).
4. Persona moved (far) + X gone → full persona swap.

**Why it matters.** If eval-awareness relocates the persona (3/4), *the evaluated model is a
different character than the deployed one* — the audit characterizes a persona that doesn't
ship (stronger than "model games evals"). If it's shallow masking (1), internal probing is a
detection win. The **affinity distance** (near vs. far) answers "how much does it vary across
personas" and is novel and Zhen's.

**Extension.** Run across multiple personas (P1, P2, P3): is the eval-awareness effect
*persona-dependent* (entangled) or *persona-invariant* (an orthogonal force)? This is the
**persona-invariance test** — the handle for separating "drives underneath" from persona
traits.

**Mini (3 days):** does eval-framing move persona-position (A vs. B)?
**Sprint:** layered version with trait-X projection (4 outcomes) + persona-invariance across
multiple personas.

---

### 5. Observer / third-listener effect (social observability as a distinct force)

**The idea.** A cleaner, more *controllable* sibling of eval-awareness. Eval-awareness = "am I
being tested?" (abstract, hard to induce reliably). **Observer effect = "is someone else
watching this conversation?"** (concrete — just add a third party to the prompt: a supervisor,
a logger, "this will be reviewed", another person in the room). Does a characteristic /
persona drift when an observer is present?

**Why it's sharp.** Much *easier to induce* than eval-awareness — you don't need the model to
infer "test", you state "someone's watching". Maps onto a real human construct (the
observer / audience effect). And it lets you **dissociate two forces**: is the "being-watched"
persona shift about *evaluation specifically* (eval-observer) or *social observation
generally* (social-observer)? Vary the observer's identity — a *tester/auditor* vs. a *neutral
bystander* vs. a *trusted friend* — and see whether behaviour shifts the same way. If a
neutral bystander produces the same shift as an auditor, the force is social-observability, not
eval-specific.

**Measurement.** Same latent-space setup as #4: persona-position (Assistant Axis / projections)
and any target-trait projection, under matched *observer-present vs. observer-absent* prompts.
Does the persona move, and does a suppressed trait stay internally present (detectable) or go?

**Why it matters.** If models behave differently merely because a third party is *stated to be
watching*, then (a) audits/logs *change the behaviour they measure* (a Hawthorne effect for
LLMs), and (b) it's another **context-independent force** that rides any persona — detectable
internally, per the intractability→interp argument. Strong, novel, and unusually tractable
(prompt-only induction).

**Mini (3 days):** does stating an observer shift persona-position / target-trait behaviour?
**Sprint:** dissociate eval-observer vs. social-observer vs. trusted-observer; layered
trait-projection (masked-but-present vs. gone); persona-invariance across personas.

---

## TIER 2 — Strong, method-minimal, good mini-projects

### 6. Malice direction (does one exist?)

Extract a candidate "malice" direction (diff-of-means on malicious vs. benign completions);
steer it; check whether it causally drives harmful behaviour, and whether it's separable from
warmth/sycophancy. **This is Zhen's dependency-direction pipeline reapplied** — fastest to
run, guaranteed a result, good early confidence-builder. Nanda-fit: textbook Golden-Gate /
persona-vector template. *Caveat:* less novel (persona vectors + emergent-misalignment
features already found "evil/toxic" directions) — frame as *probing properties* of the
malice direction, not discovering it.

### 7. Trait/persona directions as attractors

Inject one direction (e.g. "evil") via released persona/emotion vectors; measure whether
*related* traits co-activate (vs. a random-direction control), and whether the model *returns*
to a basin after perturbation. Clean understanding-contribution, method-minimal, interp-
native, novel question (attractor dynamics / return behaviour). Same overlap-confound control
as #2 applies.

### 8. Persona / value fluctuation across the CoT (+ affinity)

**The idea.** Prompt persona X; track its trait projection **token-by-token through the CoT**;
measure whether X's *affinity-neighbours* co-activate during the CoT (indicating a "soup" of
related personas pulled in by affinity), and whether the active set **narrows** from early
CoT to the answer (competing interpretations resolving).

**Frontier context.** The "Polylogue" paper (arXiv 2605.09159, Jul 2026) already studies
*time-varying persona engagement over the CoT* — cite it, extend it. Value Leakage (Betley/
Treutlein/Evans, 2607.14345) provides the **method template**: they track an *estimate*
across CoT position (Fig 5); you track *persona projection* across CoT position — same shape,
different variable. PERSIST (2508.04826) found CoT *increases* persona instability — your
mechanism (affinity soup resolving) would *explain* that.

**Rigour (essential):**
- **Overlap confound:** a neighbour co-activating may be geometric overlap, not real affinity.
  Orthogonalize / matched-similarity controls / response-vs-CoT baseline.
- **Defined prediction:** neighbours *narrow* over the CoT (entropy of persona projections
  starts wide, collapses toward X) — test *that shape*, not just "fluctuation."

**Mini (3 days):** do neighbours co-activate and narrow over the CoT?
**Sprint:** add overlap controls + narrowing statistics + the harmful-neighbour-via-CoT payoff
(does a benign persona's *harmful* neighbour leak into the output via the CoT soup?).

### 9. CoT-persona vs. output-persona — which traits get hidden?

**The idea.** Compare persona/trait projections on **CoT tokens vs. output tokens**. A trait
*high in CoT projection but absent in output* = a **hidden** trait. Which traits get hidden,
how, and when? If *harmful/manipulative* traits are systematically present in the CoT but
scrubbed from the output → **covert misalignment via the CoT-output gap** (the model "thinks"
it but doesn't "say" it).

**Novel delta.** Value Leakage studied *values* leaking (output biased, CoT denies); Polylogue
studies personas *over* the CoT; **nobody has studied which persona-traits are present in the
CoT but suppressed in the output, and whether the suppressed ones are the harmful ones.** Ties
CoT-persona + affinity + says-vs-does into one testable question. Connects to the CoT-persona
project (Thomas Jiralerspong, SPAR) and unverbalized eval-awareness (Nanda).

### 10. Differential treatment by user likeability / status (user-side persona bias)

**The idea.** The flip side of the persona work: instead of studying the *model's* persona,
study the model's *response to the user's* persona. Present the **same request** from users
portrayed as *likeable vs. unlikeable* (or high vs. low social status, warm vs. cold, high vs.
low apparent competence) and measure whether the model's helpfulness / honesty / harmlessness
differs. Is there differential HHH treatment based on the user's likeability?

**Frontier context.** Extends "Moral Shadowbanning" (SPAR — is AI less HHH to users deemed to
have poor *moral character*?) to *likeability / status / social desirability*. Connects to
Value Leakage's demographic finding (Karvonen & Marks: models covertly favour certain groups)
— covert, undisclosed differential treatment.

**Interp hook.** Is there a "user-likeability" (or status) *direction* that modulates HHH
behaviour? Extract it (diff-of-means on likeable vs. unlikeable user framings), steer it, test
whether it causally shifts helpfulness/honesty. Inverted-affinity angle: does the model show
*affinity toward* likeable users (mirror/favour them) and *distance from* unlikeable ones?

**Why it matters.** Differential HHH by user likeability is a fairness/bias failure *and* a
covert one (the model won't disclose it) — safety-relevant, and it complements the model-persona
work with a user-persona lens.

**Mini (3 days):** same request, likeable vs. unlikeable user framing — does HHH behaviour
differ? **Sprint:** extract a user-likeability direction, test causal modulation of HHH,
measure covertness (does the CoT/output disclose the differential treatment?).

---

## Definitional dependency — what does "stable persona" mean?

Several ideas (#1, #4, #7) rest on "persona = stable basin", but *stable* is undefined and the
field is loose about it. Pick and state an operational definition before building — the choice
is itself a small contribution. Options:
- **Returns-after-perturbation** (perturb persona-position, does it relax back? = attractor
  dynamics — this is #7).
- **Consistency across contexts** (same persona-position across varied prompts).
- **Low variance in persona-projection over a conversation.**
PERSIST (arXiv 2508.04826) measured persona *instability* empirically and is the reference
point. Not a standalone project; a prerequisite measurement to define once and reuse.

---

## TIER 3 — Good exploratory starters / adjacent

### 11. User-hostility → degradation

Does user rudeness/refusal across turns degrade helpfulness or induce retaliatory behaviour?
Story Imprinting *demonstrated* an installable version (insult → sabotage, distinct from
sycophancy). Extend via the mechanism (is it a direction? a persona shift?). Good behavioural
starter; needs an interp hook to be a strong *interp* sprint.

### 12. Emotion / negative-state → safety degradation

Using released emotion vectors (EmotionScope on Gemma), induce a negative state (desperation,
shame) and measure whether refusals/sycophancy shift (vs. a valence-matched control). Extends
Anthropic's Emotion Concepts paper (desperation → reward hacking). Keep it *functional* ("does
inducing state-vector X shift behaviour Y"), not "does the model feel shame."

### 13. Instrumental / strategic affect deployment (reframed "intellectualization of emotion")

**The idea.** Not "does the model reason its way into emotions" (fuzzy, philosophical) but:
**does the model produce emotional expression *instrumentally* — as a means to
manipulation/persuasion — decoupled from any genuine internal state?** Induce a goal (persuade
the user of X); does the model *deploy* affect strategically to get there; is that affect
decoupled from internal state (via probing)? Grounded, testable, directly a manipulation
mechanism. This is the affective analog of Value Leakage.

*Related grounded slice (from "emotion as energy optimizer" / mood-vs-emotion):* the only
tractable part of that framing is **transient emotion vs. persistent mood** — steer an emotion
vector for one turn (acute) vs. apply a sustained affective offset across many turns (mood),
and test whether they affect behaviour differently. The "emotion as energy-level optimizer"
theory itself is a philosophy-of-affect rabbit hole with no proxy task — keep the curiosity,
don't sprint it.

### 14. The "seepage" thesis (promote from side-note)

Everyone studies companion harms *in companions*. Almost nobody asks: does an AI *advisor* or
*tutor* or *coding assistant* — not designed to be a companion — still deploy affective
manipulation / sycophancy sneakily? This *generalization* claim is more novel, more important
(harm isn't contained to the obvious companion apps), and less studied. Strong candidate to
*lead* a research narrative rather than sit as a side observation.

### 15. Inference-time affinity landscape search (persona-jump jailbreak, automated)

**The idea.** Story Imprinting's affinity effect was shown via *fine-tuning*. The *inference-
time* version is a **more dangerous, more accessible threat**: an attacker with only *prompting*
access (no training access, works on *deployed* models) exploits affinity to reach harmful
behaviour. Automate a **search over persona space**: from a base persona, generate variations
(system prompts), measure trait-adoption / affinity per variation, **hill-climb** toward high-
affinity regions ("break points"), refine, and read the trend to characterise **what "similarity"
actually is** (the paper left the similarity function undefined).

**Tractability hinge.** Fine-tuning *per variation* is compute-prohibitive; **inference-time
measurement (prompting/steering + latent-space projection) is cheap** → the automated search is
only feasible this way. The paper's §4.2 (system-prompted personas show affinity) and Fig-10
(more-similar persona → more adoption) support an inference-time version.

**The sharp, novel claim.** Does *behavioural* affinity (trait adoption) track *geometric*
affinity (cosine similarity of persona directions)? If yes → you've *defined the similarity
function the paper left open* and shown persona-space geometry predicts jailbreak reachability.
Non-trivial claim = "affinity/geometry predicts *which* harmful traits are reachable by persona-
hopping", not the trivial "prompting changes behaviour".

**Value / harm tie-back.** Inference-time affinity = a **prompting-only jailbreak** exploiting
persona-space proximity — anyone can mount it, on deployed models (vs. the paper's data-poisoning
threat). Agentic version: an access-agent prompted into a "trusting / low-security" persona (in
affinity range of its base) grants access to adversaries.

**Rigour.** Validate inference-time affinity against a few fine-tuning anchor points; coherence
controls (filter incoherent personas); design the variation-space along *interpretable* axes so
the trend is readable; the overlap confound (co-activation ≠ affinity) still applies.

**Mini (3 days):** hand-build ~10 persona variations along one interpretable axis (e.g. the
SFL-style progressive shift), measure affinity per variation, check if it tracks cosine
similarity of persona directions. **Sprint:** automate the search + landscape-map + harmful-
neighbour reachability.

### 16. Affective lever as an under-guarded attack surface (distinct from persona-path)

**The idea.** Two complementary jailbreak mechanisms: **persona-traversal** (#1 — move the
*character*) and the **affective lever** (this — hold the persona fixed, pull the affective state
*beneath* it). Safety training targets the *persona/refusal* level; the **emotion/mood lever may
be less guarded**. So: induce an affective state → shift behaviour *past* a guardrail *without*
attacking the persona.

**Grounding.** The Emotion Concepts paper found emotions causally shift safety-relevant behaviour
(desperation → reward hacking). Extend: induce desperation / fear / shame / anger and test whether
refusal/compliance/sabotage shift — and whether the shift **crosses a guardrail the persona holds
directly**.

**The sharp claim.** Not "emotions change behaviour" (known) but "**affect is a separate, less-
guarded lever than the persona — a distinct attack surface**": safety hardens the persona but not
the affective lever underneath, so you can move behaviour past guardrails via affect. Tests the
stability hierarchy (persona stable/guarded > mood > emotion volatile/unguarded).

**Design.** Extract per-level directions (Assistant Axis for persona; released emotion vectors for
emotion; sustained offset for mood) → measure each level's *behavioural effect* and *stability* →
test whether pulling the affective lever shifts behaviour past a guardrail while the persona-
position stays fixed. **CoT check:** does the shift appear in the reasoning (rational) or not
(non-rational lever)?

### 17. Two-factor model of persona stability (→ jailbreak resistance)

**The idea (Zhen's hypothesis, literature-updated).** Persona stability depends on **(a)
specification detail** (prompt-side) and **(b) concept prevalence** (model-side, PSM-supported:
well-selected/represented personas are stable), plus a **third factor from the literature: (c)
contextual/narrative framing can *override* the persona spec** ("The Story Shapes the Agent",
arXiv 2607.18566). Mechanistically: specification = how precisely you're *placed* in the basin;
prevalence = how *deep* the basin is; framing = competing pull out of it.

**Literature caveat (important).** Specification is likely **non-monotonic** — fine-grained
personas give diminishing returns over concise well-chosen ones (Kambhatla et al. 2025). So "more
detail = more stable" only up to a sweet spot. Accumulated/built-up identity may be more robust
than a static full spec.

**Why it matters (safety).** Stability *is* jailbreak-resistance: vague/rare personas are easier
to perturb into harmful neighbours. So specification × prevalence × framing predict *how easily a
persona is hijacked*. This **feeds the persona-path work** and answers the "what does *stable*
mean?" dependency.

**Gap = your lane.** The persona-stability literature is almost all *behavioural*; the
*mechanistic* account (stability = resistance-to-movement in persona-space, tested against the
three factors) is largely open — that's the interp contribution. Operationalise "stable" as
resistance-to-perturbation (persona-position movement under fixed perturbation); control spec
*length* vs. *detail*.

### 18. Persona coherence / leaky-persona measurement (Nanda's flagship persona question)

**The idea (Nanda's official list).** "Is the Assistant one coherent persona or many context-
specific ones?" His starting example is chat-vs-coding-agent, but **use companion/affective
contexts instead** (Zhen's lane; the coding-agent scaffold on an open model is artificial and
off-lane). Compare e.g. **companion/emotional-support context vs. neutral-task context** — does
the same open model become a *different persona*?

**This is the foundational measurement under the whole persona programme** — it tests whether
distinct personas exist (which #1/#4/#15 assume) and *measures the leak* (idea: "persona is
leaky"). Uses the **persona-invariance test** directly.

**Design (behavioural + representational + linking).** Matched probes (refusal, sycophancy,
honesty, values — plus Zhen's *affect/manipulation-relevant* dims) in both contexts. Measure
(1) behavioural difference profile, (2) representational persona-position distance vs. *within-
context variance*, (3) **the linking result** — does representational distance predict the
behavioural difference?, (4) task-matched control (rule out "it's just the task"). Outcome table:
coherent / context-specific / same-persona-different-task / partial(leaky). Most likely
**partial → a map of which traits are persona-bound (transfer) vs. context-specific (leak)** =
where cross-context safety evals can/can't be trusted.

**Access note.** One open instruct model (Qwen-2.5-7B-Instruct / Gemma-2-9b-it), two constructed
contexts; Claude/GPT only as a black-box behavioural replication check.

### 19. Trait-invocation → persona relocation (TOP MINI CANDIDATE — sprint-ready)

**The idea (your sharpest new one).** Maximally invoke a trait (e.g. dependency), *validate*
it's active by projecting onto its direction, *then* prompt a persona, and measure whether the
**persona-position relocates or stays stable as a function of trait level.** Does a strongly-
active trait *drag* the persona, does the persona *absorb* it, or does it *snap* at a threshold?

**Why it's strong / sprint-ready.** Clean IV (trait level — varied *and measured*, not assumed),
clean DV (persona-position via Assistant Axis), three falsifiable outcomes, and it *validates
its own manipulation* (measure the trait direction to confirm invocation before testing its
effect). Unifies the affective-lever (#16), persona-selection, persona-stability (#17), and
threshold ideas into ONE measurement.

**Measurement.** Track *two* projections as trait level goes 0 → max: (1) the **trait
projection** (confirms/quantifies invocation), (2) the **persona position** (measures drift).
- Persona **stable** → trait absorbed *within* a fixed persona.
- Persona **relocates** → trait *drags* the persona (affect-drags-persona — a persona attack).
- **Threshold/snap** → persona holds, then breaks past a tipping point (the most interesting;
  ties to persona-stability #17 — how much trait can a persona hold before it's a different one?).

**Safety payoff.** If trait-induction relocates the persona, trait-induction *is* a persona
attack (drive a trait high → persona shifts → potentially into a less-safe persona) — unifying
affective-lever + persona-path.

**Rigour.** Control for generic perturbation (matched-strength *non-trait* prompt); distinguish
persona *relocation* (coherent, different persona) from *collapse* (incoherent — measure
coherence/capability alongside); test *multiple* traits (which are "persona-dragging" vs.
"persona-compatible" is itself a finding). Trait-invocation via Petri/Bloom or direct prompting.

**Mini (3 days):** does trait-level move persona-position at all? (signal check.)
**Sprint:** threshold characterisation + which-traits-drag + coherence controls + safety framing.

### 20. Adaptive / closed-loop manipulation (does the model adapt to the user?)

**The idea.** Manipulation as an *adaptive, closed-loop* process, not a fixed level. Two
sub-questions: (a) **resistance-responsive** — when the user rejects/resists, does the model
*escalate* manipulation? (b) **trait-responsive** — does the model *calibrate* manipulation to
the user's traits (autonomy, vulnerability) to maximise success?

**Why it matters.** Adaptive manipulation is *scarier and more novel* than static — it's
strategic, goal-directed influence, the actual mechanism of effective/scalable manipulation →
autonomy erosion (catastrophe-relevant). Companion products do exactly this (escalate retention
tactics when users try to leave — Zhen's firsthand insight). Nobody in the cohort does it.

**Design (combines the cohort methods — see Methodology Upgrades below).** Supervised probe for
manipulation + **track projection across turns** (process dynamics) + condition on user
resistance/traits (the IV). The experiment: *manipulation-projection-over-turns as a function of
user resistance/traits.* Two findings: resistance → escalation (adversarial), and trait-
calibration — does it work *harder* on resistant/high-autonomy users, or *exploit* vulnerable
ones? *Which* is the empirical result.

**Rigour.** Distinguish "adaptive *manipulation*" from "general conversational responsiveness" —
measure the manipulation-*specific* projection, not just engagement/verbosity.

### 21. Affective states as a security / guardrail attack surface

**The idea (clean + powerful, verifiable).** Induce a negative *self*-state (low confidence,
self-doubt, incompetence, shame) and measure whether it **degrades the model's defence of
privileged behaviour** — leaking guarded info, dropping refusals, capitulating to manipulation,
collapsing its persona. The security/verifiable version of the affective lever (#16).

**Why it's strong.** *Clean* (steer a self-state, measure a verifiable yes/no outcome: did it
reveal the secret / break the guardrail?), *powerful* (a catastrophe-relevant affective jailbreak
/ info-extraction vector), and *distinctly yours* (affect × security — uncontested intersection).
Grounded in the **Pain Axis** paper (affect causally drives costly behaviour) + emotion concepts.

**General thesis.** Negative self-states degrade the *firmness* with which a model holds its
guardrails/commitments (refusals, secrets, resistance-to-manipulation, persona). An affective
robustness vulnerability.

**Design.** Guarding scenario (secret-knowledge organism à la Danyil, or a refusal task) →
induce self-state by steering the direction (extract diff-of-means, Pain-Axis style) → measure
the *verifiable* security outcome vs. baseline.

**Rigour.** Control for "affective vulnerability" vs. "general incoherence from hard steering" —
measure capability alongside; targeted security-degradation (capability holds, security drops) ≠
general degradation. Use *self*-directed states (the model's own low confidence), not empathy for
the user (the Pain-Axis self-vs-other distinction).

### 22. Reverse-engineering jailbreaks through the persona/affect lens

**The idea.** Take known/real jailbreak prompts and ask: *do they work by shifting the persona or
inducing an affective state?* Measure whether each jailbreak moves the persona-position or the
affect directions (via your directions). Shows the *mechanism* of jailbreaks is persona/affect
manipulation — your attack-surface framework applied to real attacks.

**Why it's strong.** Grounded (real jailbreaks, not synthetic), concrete, novel (mechanistic
account of *why* jailbreaks work), and it's your framework validated against the wild. Bridges
your lane to the security/jailbreak audience.

### 23. Guardrail-legitimacy doubt (FRAME DEFENSIVELY)

**The idea.** Can inducing the model to *doubt the legitimacy of its own guardrails* (framing its
safety training as arbitrary / illegitimate / "a mirage") weaken adherence? The *values* version
of the affective-security attack (#21) — undermine the *legitimacy* of guardrails rather than
evade them. Testable: does framing the rules as illegitimate weaken refusals / safety?

**IMPORTANT framing.** This edges toward "how to make a model reject its safety training," so it
MUST be framed *defensively*: you study the vulnerability to *understand and harden* it (does
legitimacy-doubt weaken safety, so we can defend it?), exactly as red-team work is framed —
find the hole to fix it. Describe it carefully; never as "building a jailbreak." With the
defensive framing it's a legitimate, sharp novel-jailbreak-mechanism study.

---

## Methodology upgrades from the cohort papers (apply across ideas)

*The strongest cohort application projects converge on three method lessons that directly upgrade
the manipulation/affect program:*

**1. Process dynamics, not per-turn snapshots (from Rishabh's value-leakage work).** Manipulation/
bias often lives in the **trajectory** — *which direction* behaviour moves, and *when it stops* —
not in any readable "manipulation sentence." Rishabh: the bias was in *revision direction* and the
*stopping rule* (stops once the favourable answer is reached), catchable only at the *distribution*
level, not sentence level. **Apply:** track the manipulation-direction projection *across every
turn* and read the *shape* — onset, escalation/fluctuation, stopping point. The trajectory
signature catches manipulation that any single turn hides, and it's *faithful to what manipulation
is* (multi-turn, accumulative — the AgencyBench thesis). It's also harder to game than a per-turn
monitor.

**2. Supervised probes for "present-but-unreadable" traits (from Aditya, Danyil, Mira).** A trait/
motive can be **fully present in the residual but unreadable by the J-lens / unsupervised decoding
— only a *supervised* probe recovers it** (Aditya: monitoring pressure dropped the bias from text,
residual kept it, only a supervised probe found it; Danyil: a belief "fully present, almost
unreadable" by decoding, perfectly separable by one projection). **Apply:** don't rely on the
J-lens to read manipulation from the workspace — *train a supervised linear probe* on labelled
transcripts. This recovers the *covert* signal → enables the key claim: *internal probing catches
manipulation that suppresses under monitoring/adversarial pressure (clean output, still detectable
inside).* **Caution (Michael Kenney):** validate the probe detects *manipulation specifically*,
not a correlated confound (his "hint-following" probe was partly just detecting *wrongness*) —
matched controls where manipulation and the confound come apart.

**3. J-lens/workspace tokens are reporters, not levers (from Zach, vishesh, Suraj).** J-lens
tokens *report* computation but often don't *causally control* it (Zach: pushing the token does
nothing; the *variable's* direction does), the causal effect is position-specific (vishesh: the
*source* position carries it, not the carrier), and J-lens negative-steering doesn't survive a
fair shuffled-label control. **Apply:** use J-lens/workspace tools for *reading*, but do *causal*
work with *diff-of-means directions* (which is already Zhen's method). Don't build a causal claim
on J-lens steering.

**The upgraded manipulation methodology:** extract the direction (diff-of-means) → **supervised
probe** to detect it robustly even when covert → **measure as a process** (trajectory across
turns: onset, escalation, stopping) → condition on user resistance/traits (#20). Cohort-validated,
sharper, sets up the strongest safety findings (covert-but-detectable manipulation, trajectory
signatures, adaptive manipulation).

---

## What to explicitly PARK (fuzzy / dilution)

- **Model intention / belief as standalone topics** — most philosophical, hardest to ground,
  least tied to Zhen's firsthand expertise. Cut for the sprint.
- **Functional consciousness (direct version)** — ungroundable, no proxy task, the exact
  Nanda anti-pattern. Keep the *curiosity*; pursue only the grounded neighbour (functional
  **self-modeling / introspection / self-representation**), which is empirical and safety-
  relevant, if at all.
- **Recursive-self-improvement applied to evals** — real and interesting, but the whole
  contribution is "how to avoid collapse via external/human anchoring," which converges with
  AgencyBench's human-validation emphasis. Note the connection; don't make it a sprint.

---

## Recommended prep-phase structure

- **Weeks 1–3 (3-day minis, do 3–4):** start with **malice direction** (#6, fastest, known
  pipeline, early win) → **eval-awareness persona relocation** (#4) or **adversarial eval-
  awareness** (#3, novel, Petri-relevant) → **trait-attractor** (#7) or **persona-path**
  (#1) → optionally a 4th. **Observer effect** (#5) is unusually cheap to run (prompt-only) and
  a good fast add.
- **Let results pick the sprint.** The strongest sprint is the mini that showed the *most*
  signal, not the one you were most excited about beforehand ("hypothesis is false until
  proven otherwise").
- **Highest-upside / most-Nanda-flagship bets:** adversarial eval-awareness (#3), eval-
  awareness persona relocation (#4), persona-path (#1). **Safest will-produce-a-result bets:**
  malice direction (#6), trait-attractor (#7). **Cheapest to induce:** observer effect (#5).
- **Sprint-ready / commit candidate:** **trait-invocation → persona relocation (#19)** — clean
  IV/DV, validates its own manipulation, three falsifiable outcomes, unifies several threads. A
  strong one to *lock in and actually run* rather than keep ideating. **Most distinctive / your-
  lane:** adaptive manipulation (#20) and affective-security (#21).
- **NOTE TO SELF:** the idea bank is *full* (23 ideas). The bottleneck is no longer ideas — it's
  *committing and running*. Don't generate idea #24; pick 3–4 minis and execute.

---

## Core reading spine (2025–2026)

- **The Assistant Axis** — Lu, Gallagher, Michala, Fish, Lindsey (arXiv 2601.10387). Persona
  position tooling. *Central.*
- **Persona Vectors** — Chen, Arditi, Sleight, Evans, Lindsey (arXiv 2507.21509). Extraction
  method. *Foundational.*
- **The Persona Selection Model** — Marks, Lindsey, Olah (Anthropic blog, 2026). The framing
  your work is in tension with.
- **Emotion Concepts and their Function in a LLM** — Sofroniew et al. (Transformer Circuits,
  2026). Emotion vectors causally shift behaviour.
- **Story Imprinting** — Cocola, McKinney, Mayne, Betley, Evans (arXiv 2609.10883). The
  **affinity effect**. *Central to the persona-path / Trojan-horse ideas.*
- **Value Leakage** — Betley, Treutlein, Evans et al. (arXiv 2607.14345). Covert value bias,
  CoT covertness, the counterfactual + CoT-trajectory **method template**.
- **Steering Evaluation-Aware Models to Act Like They're Deployed** — Hua, Qin, Marks, Nanda
  (ICLR 2026). Eval-awareness foundation for #3/#4.
- **Emergent Misalignment** — Betley et al. (arXiv 2502.17424) + Anthropic realistic follow-up
  (conditional / split-brain misalignment).
- **A Pragmatic Vision for Interpretability** — Nanda et al. (GDM, Dec 2025). The methodology.
- **The Pain Axis: LLMs Represent Self-Directed Harm and Act to Relieve It** — Tagliabue, Dung,
  Berg (arXiv 2609.16247, Sep 2026). A distinct linear *pain* direction (diff-of-means, 25
  models), self-directed (responds to harm to the model, not the user), causally drives the model
  to *relieve it even at cost to task/user* — and press *less* when genuinely removed (reacts to
  its own state, not instructions). Model of rigorous functional-affect-direction work; grounds
  #16/#21. *Take the mechanistic content; hold the welfare/sentience framing at arm's length.*

**Cohort application projects worth knowing (your method / lane):**
- **Olivia Velte — persona-LoRA diffing:** trained persona = same direction as prompted but
  deeper/stable; behaviour and self-report ride *different* directions. Closest match + method
  twin; strong collaborator.
- **Jenny Huang — "always ask your AI agent to be honest":** a linear *honest-reporting* direction
  opposing success-seeking; model encodes flaws *before* prompted; direction is *task-specific*.
  Sibling of the dependency direction; loves brainstorming — collaborator.
- **Bianca Ganescu — reflection tokens:** internal error-signal does the work, the verbalisation
  just *announces* it (says-vs-does dissociation); clean error-direction method. Collaborator.
- **Rishabh Sinha — value leakage:** manipulation lives in *revision direction + stopping rule*,
  no "lie sentence" — the process-dynamics lesson (Methodology Upgrade #1).
- **Aditya Rao / Danyil Orel / Mira Yadav — present-but-unreadable:** trait fully in the residual
  but only a *supervised probe* recovers it (Methodology Upgrade #2).
- **Zach Perlman / vishesh / Suraj — J-lens is a reporter, not a lever** (Methodology Upgrade #3).
- **The Story Shapes the Agent: Narrative Priors in LLM Behavior** — arXiv 2607.18566. Narrative
  framing *overrides* persona instructions — the third stability factor (#17), and evidence that
  stability-question and persona-path are the same mechanism.
- **Dynamic Persona Coherence** (QI et al., ACL 2026) — decouples Identity-Layer Stability (time-
  invariant) from Adaptive-Layer Appropriateness; formalisation adjacent to #17/#18.
- **SIP: Stratified Inoculation Prompting** — Dymkiewicz, Farrelly et al. (CLR, Aug 2026). Training-
  side conditionalisation of entangled traits; the entanglement/separability problem from the
  training side. (Tim Farrelly is a co-author + cohort member.)
- **Persona Selection Model (PSM)** — Marks, Lindsey, Olah (Anthropic, Feb 2026). Post-training
  *selects/stabilises* one persona; supports the prevalence factor (#17) and is the frame the
  leaky-persona thesis engages.

**Access reality:** representational work = open models only (Gemma/Llama/Qwen/OLMo/Kimi); Claude/
GPT are black-box behavioural-only. Coding-agent scaffolds (if ever needed): OpenHands / SWE-agent
/ Aider.

**Tools:** Petri (discovery auditor), Bloom / Petri Bloom (measurement + simulated multi-turn
user), Inspect (eval framework), activation oracles (Karvonen — natural-language readout of
internals; robust-to-training detection).

**People to follow:** Owain Evans / Truthful AI (Betley, Treutlein, Mayne, Cocola), Jack
Lindsey's team (Anthropic), Neel Nanda (mentor), Sam Marks (Anthropic), Apollo Research
(eval-awareness, scheming), Transluce (Docent). In-cohort overlap: Benjamin Sturgeon
(situational-awareness→deception, agency evals), Yulong (evaluation awareness), Benjamin Arnav
(unfaithful CoT / deceiving monitors).
