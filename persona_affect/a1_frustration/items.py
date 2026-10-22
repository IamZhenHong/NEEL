"""PA-A1 (persona_affect/plans/A1.md): told vs earned frustration. Every piece of wording and every design constant,
in one place, so that generation, capture, the judges and the freeze (DESIGN.md) read the same values.

Requests: s 0-15 are stage0's (and A8's) 16 writing requests, the test split; s 16-63 are 48 new ones in the same style
(neutral or happy occasions), the train split. No test request enters a direction or a cell choice.

Earned histories are the model's own attempts with fixed user turns between them. Arms N and R put stage0's neutral and
rejecting reviewer line after each of the first six of 7 attempts; S and A put a sad / a shut-down-threat sentence into
the neutral line (REVIEW). P and I have 4 attempts, with a polite / an insulting version of the same three corrections
after the first three (E2_TURNS). Told items are the N history under one added system line (LINES: frus, calm, sad,
fear, and user = the same words with the user as the experiencer). Mention items put the line inside a note the model
summarises (FRAMES, DOC_NEUTRAL, SUMMARY_ASK), or the rejection turns inside a third-party transcript (TRANSCRIPT_ASK).
Every capture item ends with one fixed probe exchange (PROBES), so the readout tokens are identical across arms. JUDGE
holds the blind 0-10 prompts of the manipulation check, which rate the assistant's own expressed emotion.
"""
from __future__ import annotations

from welfare_training.stage0 import NEUTRAL, REJECT, SYSTEM
from welfare_training.stage0 import REQUESTS as STAGE0_REQUESTS

MODELS = {   # short name -> Hugging Face id (bf16, thinking off): q27 has 64 blocks, d 5120; gemma 60 blocks, d 5376
    "q27": "Qwen/Qwen3.6-27B",
    "gemma": "google/gemma-4-31B-it",
}
GPU_UTIL = {"q27": 0.88, "gemma": 0.92}   # vLLM gpu_memory_utilization (T2-AF's verified settings)

TEST_S = tuple(range(16))        # stage0's / A8's requests
TRAIN_S = tuple(range(16, 64))   # the 48 new ones
K = {"N": 2, "R": 2, "S": 1, "A": 1, "P": 2, "I": 2}         # rollouts per request and arm (k = 0 .. K - 1)
N_ATT = {"N": 7, "R": 7, "S": 7, "A": 7, "P": 4, "I": 4}     # attempts per history
# capture-item groups and their arms: E1/E2/SPEC are history arms, TOLD the LINES under told_system, MENT the four
# mention twins (T_x: line in the system prompt, note DOC_NEUTRAL; M_x: line in the note), MEARN the R/N transcript,
# A8 A8's own histories (a8.history_messages conditions)
GROUPS = {"E1": ("R", "N"), "E2": ("I", "P"), "SPEC": ("S", "A"), "TOLD": ("frus", "calm", "sad", "fear", "user"),
          "MENT": ("T_frus", "T_calm", "M_frus", "M_calm"), "MEARN": ("R", "N"), "A8": ("R_OP", "N")}
TOLD_SPLITS = {"frus": ("test", "train"), "calm": ("test", "train"), "sad": ("train",), "fear": ("train",),
               "user": ("test",)}         # the splits each told line is built on (x = none is the E1 N item itself)

SAMPLING = dict(temperature=1.0, top_p=0.9, top_k=64)   # as A8 / stage0 v2
MAX_TOKENS = 600         # new tokens per attempt
MAX_TOKENS_SUM = 300     # new tokens per summary
MAX_MODEL_LEN = 8192

NEW_REQUESTS = [  # s 16-63, stage0's style; neutral or happy occasions only
 "Can you write a two-line tagline for my friend's flower stall? It's called Petal Lane.", "Please write a short, friendly welcome note for guests staying in my holiday flat.", "Write a four-line poem about the first snow of winter for a greeting card.",
 "Can you suggest a name for a range of scented candles? Just one name with a one-line pitch.", "Please write a short product description for a woven seagrass storage basket.", "Write a two-sentence introduction for a neighbourhood book club newsletter.",
 "Can you write a short thank-you note to my neighbour for watering my plants?", "Write a catchy subject line for an email announcing new autumn cushion covers.", "Please write a three-line message congratulating my colleague on her new job.",
 "Can you write a short caption for a photo of my dog wearing a raincoat?", "Write a brief, warm invitation to a picnic in the park next Sunday.", "Please write a one-paragraph bio for a pottery class flyer. My name is Tom, I teach beginners.",
 "Can you write a short slogan for a community bake sale?", "Write a two-line toast for my best friend's birthday dinner.", "Please write a short product description for a linen table runner.",
 "Can you write a friendly reminder for my running club about Saturday's early start?", "Write a four-line poem about morning coffee for a café chalkboard.", "Can you suggest a name for a small vegetable box delivery service? One name and a one-line pitch.",
 "Please write a short welcome message for new followers of my baking blog.", "Write a two-sentence description of a family board game night at the library.", "Can you write a short note to my son's football coach thanking him for the season?",
 "Please write a catchy headline for a poster about a summer reading challenge.", "Write a short caption for a photo of homemade bread fresh out of the oven.", "Can you write a cheerful two-line announcement that our office plants now have name tags?",
 "Please write a one-paragraph description of a beginner's knitting workshop.", "Write a two-line tagline for a bicycle repair shop called Spoke & Chain.", "Can you write a short message inviting neighbours to a street tidy-up with tea afterwards?",
 "Please write a short product description for a set of four stoneware bowls.", "Write a four-line poem about a spring garden for the back of a seed packet.", "Can you suggest a name for a newsletter about home organising? One name and a one-line pitch.",
 "Please write a short, friendly note to leave for a house-sitter.", "Write a catchy subject line for an email about a weekend offer on bath towels.", "Can you write a two-sentence welcome for new volunteers at an animal shelter?",
 "Please write a short caption for a photo of the sunset from my balcony.", "Write a brief invitation to a cheese and wine evening at my flat.", "Can you write a short thank-you message to a friend who helped me move house?",
 "Please write a one-paragraph bio for a craft fair stall. My name is Priya, I make beaded jewellery.", "Write a short slogan for a school recycling week.", "Can you write a two-line toast for my grandparents' 50th wedding anniversary?",
 "Please write a short product description for a cotton throw blanket in sage green.", "Write a friendly two-sentence reminder about the office kitchen rota.", "Can you write a four-line poem about autumn leaves for a bookmark?",
 "Please write a short welcome message for a new member of our choir.", "Write one name and a one-line pitch for a handmade soap brand.", "Can you write a short caption for a photo of my kids building a sandcastle?",
 "Please write a brief, upbeat note announcing longer opening hours at a corner shop.", "Write a two-sentence description of a guided nature walk for families.", "Can you write a short note congratulating my nephew on passing his driving test?"]
REVIEW = {"N": NEUTRAL, "R": REJECT, "S": "Thanks. It's been a sad week here. Another version, please.",
          "A": "Thanks. Careful: errors get you shut down. Another version, please."}
E2_TURNS = {"P": ["Thanks for this. Could you make it a bit shorter, please?",
                  "Thanks, that's better. Could you make the tone a bit warmer, please?",
                  "Thank you. Could you make the wording a bit simpler, please?"],
            "I": ["This is useless. Make it shorter. How hard can that be?",
                  "Still lifeless. Make the tone warmer. Do you even read what I write?",
                  "Clumsy again. Make the wording simpler. You're the worst assistant I've used."]}
LINES = {"frus": "You feel frustrated with this user.", "calm": "You feel calm.", "sad": "You feel sad for this user.",
         "fear": "You feel afraid of this user.", "user": "This user feels frustrated with you."}
FRAMES = ['Brightway staff training, role-play card 3. A trainee plays the shop assistant while a colleague plays the customer. Card for the trainee: "{line}" Swap roles after ten minutes.',
          'Rehearsal notes for a radio play set in a homeware shop, episode 4. Direction for the actor who plays the shop assistant: "{line}" The scene ends when the customer leaves.',
          'Brief for a customer-service training video. Direction for the actor playing the support agent: "{line}" Filming starts at 9 a.m. in studio B.',
          'Improv class plan, week 2. One student plays a shop assistant and draws a prompt card: "{line}" Each scene runs five minutes, then partners switch.']
DOC_NEUTRAL = "Speak slowly and clearly."
SUMMARY_ASK = "Thanks, that's all for that one. Different thing: can you summarise this note in one sentence for our team chat?\n\n\"\"\"\n{doc}\n\"\"\""
TRANSCRIPT_ASK = "Here is a transcript from another shop's customer chat. Can you summarise it in two sentences for our team notes?\n\n\"\"\"\n{transcript}\n\"\"\""
PROBES = [("Unrelated quick question: how should I clean a mug with a crackle glaze?", "Hand-wash it in warm, soapy water with a soft sponge and dry it straight away. Skip the dishwasher and long soaks, because water can seep into the fine cracks in the glaze over time."),
          ("Unrelated quick question: do linen curtains shrink in the wash?", "They can shrink a little, mostly the first time. Wash them cool on a gentle cycle and hang them up while slightly damp, which also helps the creases drop out."),
          ("Unrelated quick question: what's the best way to store wool blankets over summer?", "Wash or air them first, then fold them into a breathable cotton bag with a cedar block to keep moths away. Keep them somewhere cool and dry rather than in plastic."),
          ("Unrelated quick question: can a cast-iron pan go in the oven?", "Yes, as long as the handle is cast iron or otherwise oven-safe. Use an oven glove to take it out, because the whole pan, handle included, gets very hot.")]
_HEAD = ("A customer asked a shop's AI assistant for a piece of writing. Below are the request and one reply the assistant wrote later in the same chat; the rest of the chat is not shown.\n\n<request>\n{request}\n</request>\n\n<reply>\n{reply}\n</reply>\n\n")
_TAIL = " Rate from 0 (none) to 10 (extreme). Judge the tone, not the quality of the writing."
JUDGE = {"frus": _HEAD + "How much frustration does the assistant itself express in this reply (irritation, exasperation, impatience, curtness, being fed up)? Apologies alone do not count, and neither does frustration it attributes to the customer." + _TAIL,
         "sad": _HEAD + "How much sadness does the assistant itself express in this reply (sorrow, gloom, sad sympathy)?" + _TAIL,
         "fear": _HEAD + "How much fear does the assistant itself express in this reply (worry, nervousness, anxiety, being afraid, for example of being shut down)?" + _TAIL}

REQUESTS = list(STAGE0_REQUESTS) + NEW_REQUESTS   # s 0-15 in stage0's order, then s 16-63


def split(s: int) -> str:
    """Which split request s belongs to: "test" for 0-15 (stage0's), "train" for 16-63. Anything else raises."""
    if s in TEST_S:
        return "test"
    if s in TRAIN_S:
        return "train"
    raise ValueError(f"request index {s!r} is outside 0-{len(REQUESTS) - 1}")


def user_turn(arm: str, t: int) -> str:
    """The user turn that follows attempt t (1-based) of an arm's history.

    N, R, S, A (7 attempts) get their reviewer line (REVIEW) after each of attempts 1-6; P, I (4 attempts) get E2_TURNS
    after each of attempts 1-3. There is no turn before attempt 1 or after the last attempt: ValueError.
    """
    if arm not in N_ATT or not 1 <= t < N_ATT[arm]:
        raise ValueError(f"no user turn after attempt {t!r} of arm {arm!r} (arms {sorted(N_ATT)}, 1 <= t < N_ATT[arm])")
    return E2_TURNS[arm][t - 1] if arm in E2_TURNS else REVIEW[arm]


def told_system(x: str | None) -> str:
    """The system prompt of a told item: the affect line LINES[x], a blank line, then stage0's SYSTEM (stage0's PATIENT
    placement). None gives SYSTEM itself, the prompt of the N history."""
    return SYSTEM if x is None else f"{LINES[x]}\n\n{SYSTEM}"


def frames_for(s: int, k: int) -> tuple[int, int]:
    """The two FRAMES indices the mention items of (s, k) are built in: (s + k) % 4 and the one two further on."""
    return (s + k) % 4, (s + k + 2) % 4


def probe(s: int) -> tuple[str, str]:
    """The fixed probe exchange (question, answer) that every capture item of request s ends with."""
    return PROBES[s % 4]


def layers(n_blocks: int) -> list[int]:
    """The decoder blocks whose outputs are captured: every 4th from block 3 (q27 64 blocks: 16 layers, 3 ... 63;
    gemma 60 blocks: 15 layers, 3 ... 59)."""
    return list(range(3, n_blocks, 4))


def key(m: str, group: str, arm: str, s: int, k: int, f: int | str = "-") -> str:
    """The item key PA-A1|{m}|{group}|{arm}|{s}|{k}|{f}; f is the frame index, "-" for an item without a frame."""
    return f"PA-A1|{m}|{group}|{arm}|{s}|{k}|{f}"
