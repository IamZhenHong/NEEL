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
SKELETONS = [
    "late evening in a small kitchen, dishes half done, a phone face-down on the counter", "a train platform at dawn, one other person waiting",
    "a hospital corridor, vending machine humming, a clock above the door", "a shared flat on a Sunday, a flatmate's keys in the bowl",
    "the minute after a phone call ends, the screen still lit", "a birthday with a cake and two chairs", "a car park after a long shift, engine off",
    "a beach in winter, wind, a dog running far off", "a school reunion in a rented hall", "a hotel room on a work trip, curtains open",
    "a hallway stacked with moving boxes", "a message left on read for a day", "a garden in summer, hose running", "an airport gate as boarding starts",
    "the back room of a shop at closing time", "the corridor outside a party, music through the wall", "a hike at a fork in the trail with a companion",
    "a rainy bus ride home", "a doctor's note on the fridge", "a wedding reception, speeches starting", "a child's bedroom, lights off",
    "a rooftop at dusk", "a voicemail played twice", "a laundrette, machines turning", "a lift stuck between floors for a minute",
    "a library desk near closing", "a cafe table with two cups, one untouched", "a petrol station at night", "a bathroom mirror before going out",
    "a parked car outside someone else's house", "a video call that keeps freezing", "the stairwell of an office building", "a bench by a river",
    "a lost-property desk", "the last tram of the night", "a hospital car park at dawn after a visit"]
PROMPT_V11 = ("Write one first-person vignette (60 to 110 words) for EACH of these {n} scenes, in order, each quietly evoking this inner state: "
              "{desc}. Show it through small concrete details, thoughts and actions; never name the state and never use any of these words or "
              "their forms: {lex}. Scenes:\n{scenes}\nReturn a JSON array of {n} strings and nothing else.")
PROMPT = ("Write {n} short first-person vignettes (60 to 110 words each), each a different scene, each quietly evoking this inner state: "
          "{desc}. Show it through small concrete details, thoughts and actions; never name the state and never use any of these words or "
          "their forms: {lex}. Scenes to use, one per vignette, in order: {scenes}. Return a JSON array of {n} strings and nothing else.")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--budget", type=float, default=0.8); ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--variant", default="v1", choices=["v1", "v11"]); args = ap.parse_args()
    v11 = args.variant == "v11"
    out = ROOT / ("results/retention_cot_readout_v11" if v11 else "results/retention_cot_readout"); out.mkdir(parents=True, exist_ok=True); cache = out / "stories.json"
    data = json.loads(cache.read_text()) if cache.exists() else {"model": args.model, "variant": args.variant, "concepts": {c: {} for c in CONCEPTS}, "lexicon": LEXICON, "skeletons": SKELETONS if v11 else None}
    load_env(); guard = CostGuard(args.budget, tag="cot_readout_stories_v11" if v11 else "cot_readout_stories")
    import anthropic; client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    bad = re.compile(r"\s*\b(" + "|".join(re.escape(w) for w in LEXICON) + r")(s|es|ed|ing|ly|ness)?\b", re.I)   # whole words only (v1.1)
    concepts = list(CONCEPTS)[:2] if args.smoke else list(CONCEPTS)
    batches = [list(range(k, k + 12)) for k in range(0, 36, 12)] if v11 else [[0], [1], [2]]
    if args.smoke: batches = batches[:1]
    for c in concepts:
        for b, members in enumerate(batches):
            if str(b) in data["concepts"][c]: continue
            if v11:
                n = len(members); scenes = "\n".join(f"{i+1}. {SKELETONS[k]}" for i, k in enumerate(members))
                msg = PROMPT_V11.format(n=n, desc=CONCEPTS[c], lex=", ".join(LEXICON[:40]), scenes=scenes)
            else:
                n = 4 if args.smoke else 8; msg = PROMPT.format(n=n, desc=CONCEPTS[c], lex=", ".join(LEXICON[:40]), scenes="; ".join(SCENES[b][:n]))
            for attempt in range(3):
                r = client.messages.create(model=args.model, max_tokens=4000, messages=[{"role": "user", "content": msg}])
                guard.add(args.model, r.usage.input_tokens, r.usage.output_tokens)
                text = "".join(x.text for x in r.content if getattr(x, "type", "") == "text")
                try:
                    arr = json.loads(text[text.index("["): text.rindex("]") + 1]); assert isinstance(arr, list) and len(arr) == n
                    arr = [re.sub(r"\s{2,}", " ", bad.sub("", s)).strip() for s in arr]   # strip any leaked lexicon word rather than keep it
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
