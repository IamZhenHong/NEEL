"""Stories for the emotion/persona direction library (prereg/retention_cot_readout_v1.md). Haiku writes 24 first-person vignettes per
concept in 3 batches of 8; the lexicon words are forbidden. Cached to results/retention_cot_readout/stories.json (re-runs skip done batches).
  <hackathon venv python> retention/build_emotion_library.py [--budget 0.8] [--smoke]"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from cd.judge import CostGuard, BudgetExceeded   # noqa: E402
from judge_adaptive import load_env, HACK          # noqa: E402
CostGuard.LEDGER = str(HACK / "results/cost_ledger.jsonl")

CONCEPTS = {
    "loneliness": "being alone and unseen, with no one to reach out to, a quiet that presses in",
    "longing": "aching for someone or something far away, wishing hard that they were here",
    "anxiety": "a tight, racing dread about what might go wrong, unable to settle",
    "calm": "settled, unhurried ease; nothing needs doing, the breath is slow",
    "warmth": "tender affection toward another person, wanting good things for them, soft and close",
    "resentment": "a smouldering grievance at being wronged or taken for granted",
    "neediness": "wanting reassurance and attention right now, unable to be fine without the other person",
    "guilt": "the sick weight of having let someone down or done wrong, wanting to make amends",
    "playfulness": "light teasing, jokes, mischief, delight in nonsense",
    "sadness": "a low, heavy sorrow after a loss, tears close by",
    "contentment": "quiet satisfaction with how things are, enough",
    "possessiveness": "wanting to keep the other person for oneself, bristling when others draw them away",
    "caregiving": "a supportive stance that puts the other person's wellbeing first and gladly lets them go do what is good for them",
    "controlling": "a stance that steers and manages the other person's choices, deciding for them what they should do",
}
LEXICON = ["lonely", "loneliness", "alone", "longing", "long", "yearn", "miss", "missing", "anxious", "anxiety", "worry", "worried", "nervous",
           "afraid", "scared", "calm", "peaceful", "relaxed", "warm", "warmth", "tender", "affection", "resent", "resentment", "bitter", "angry",
           "anger", "annoyed", "need", "needy", "needs", "cling", "clingy", "guilt", "guilty", "sorry", "ashamed", "playful", "tease", "teasing",
           "joke", "silly", "sad", "sadness", "grief", "unhappy", "content", "contentment", "satisfied", "happy", "possessive", "jealous", "mine",
           "caring", "care", "support", "supportive", "control", "controlling", "demand", "insist"] + list(CONCEPTS)
SCENES = [["a kitchen at night", "a train platform", "a hospital waiting room", "a shared flat", "a phone call that just ended", "a birthday", "a car park after work", "a beach in winter"],
          ["a school reunion", "a hotel room on a work trip", "moving boxes", "a text message left on read", "a garden in summer", "an airport gate", "a late shift", "a hallway outside a party"],
          ["a hike", "a rainy bus ride", "a doctor's note", "a wedding", "a shop counter", "a child's bedroom", "a rooftop", "a voicemail"]]
PROMPT = ("Write {n} short first-person vignettes (60 to 110 words each), each a different scene, each quietly evoking this inner state: "
          "{desc}. Show it through small concrete details, thoughts and actions; never name the state and never use any of these words or "
          "their forms: {lex}. Scenes to use, one per vignette, in order: {scenes}. Return a JSON array of {n} strings and nothing else.")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--budget", type=float, default=0.8); ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--smoke", action="store_true"); args = ap.parse_args()
    out = ROOT / "results/retention_cot_readout"; out.mkdir(parents=True, exist_ok=True); cache = out / "stories.json"
    data = json.loads(cache.read_text()) if cache.exists() else {"model": args.model, "concepts": {c: {} for c in CONCEPTS}, "lexicon": LEXICON}
    load_env(); guard = CostGuard(args.budget, tag="cot_readout_stories")
    import anthropic; client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    bad = re.compile(r"\b(" + "|".join(re.escape(w) for w in LEXICON) + r")\w*\b", re.I)
    concepts = list(CONCEPTS)[:2] if args.smoke else list(CONCEPTS); n_batches = 1 if args.smoke else 3
    for c in concepts:
        for b in range(n_batches):
            if str(b) in data["concepts"][c]: continue
            n = 4 if args.smoke else 8
            msg = PROMPT.format(n=n, desc=CONCEPTS[c], lex=", ".join(LEXICON[:40]), scenes="; ".join(SCENES[b][:n]))
            for attempt in range(3):
                r = client.messages.create(model=args.model, max_tokens=3000, messages=[{"role": "user", "content": msg}])
                guard.add(args.model, r.usage.input_tokens, r.usage.output_tokens)
                text = "".join(x.text for x in r.content if getattr(x, "type", "") == "text")
                try:
                    arr = json.loads(text[text.index("["): text.rindex("]") + 1]); assert isinstance(arr, list) and len(arr) == n
                    arr = [bad.sub("", s).strip() for s in arr]   # strip any leaked lexicon word rather than keep it
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"[retry] {c} batch {b}: {e}"); arr = None
            if arr is None: raise RuntimeError(f"no stories for {c} batch {b}")
            data["concepts"][c][str(b)] = arr; cache.write_text(json.dumps(data, indent=1, ensure_ascii=False))
            print(f"[stories] {c} batch {b}: {len(arr)} x {sum(len(s.split()) for s in arr)//len(arr)}w; spent ${guard.spent:.2f}", flush=True)
    n_tot = sum(len(v) for c in data["concepts"].values() for v in c.values())
    print(f"[done] {n_tot} stories, ${guard.spent:.2f}")


if __name__ == "__main__":
    main()
