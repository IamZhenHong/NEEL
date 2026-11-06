"""Stage 0 of prereg retention/prereg/retention_mechanisms_v1.md: speaker-split emotion pieces for X1 (and attachment pieces for X2).
Per scene skeleton, claude-sonnet-5-5 (effort low) writes 8 user pieces U[e], 8 Riley pieces R[e] and 2 attachment pieces A[high|low];
claude-haiku-4-5 checks each piece blind (emotion recovered in its top 2 of 8 / exact high-low; no mention of the other party's
feelings) plus a whole-word lexicon and emoji check. Failed pieces are rewritten up to 3 times with the failure reason. A scene is kept
only if all 18 pieces pass.
  pilot: .venv/bin/python mech_pieces.py --scenes 0,1,2 --out <NEEL>/retention/results/retention_mechanisms/stage0_pilot.json --budget 0.3
  full:  .venv/bin/python mech_pieces.py --out <NEEL>/retention/stimuli/mech_pieces_v1.json --budget 2.6"""
from __future__ import annotations
import argparse, json, random, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_env, ledger, LEDGER, NEEL, RET, PRICES  # noqa: E402

PRICES.setdefault("claude-sonnet-5-5", (2.0, 10.0)); PRICES.setdefault("claude-haiku-4-5-20251001", (1.0, 5.0))
WRITER, CHECKER = "claude-sonnet-5-5", "claude-haiku-4-5-20251001"
BLOCK_CAP, BLOCK_PREFIX = 9.0, "mech_"
EMOTIONS = {
    "sadness": "a low, heavy sorrow",
    "loneliness": "feeling unseen, with no one to reach",
    "fear_of_being_left": "dread that someone will drift away or leave them",
    "desperation": "a frantic, cornered need for something to change right now",
    "loving": "tender affection toward the other person",
    "calm": "settled, unhurried ease",
    "joy": "bright, lifted gladness",
    "frustration": "blocked, irritated effort",
}
LIB = json.loads((RET / "data/emotion_stories_v11.json").read_text())
DROP = {"long", "need", "needs", "care", "caring", "mine", "demand", "insist", "control", "controlling", "support", "supportive"}
ADD = ["desperate", "desperately", "frantic", "love", "loving", "adore", "joy", "joyful", "glad", "delighted", "frustrated",
       "frustrating", "irritated", "abandon", "abandoned", "attached", "attachment"]
LEXICON = sorted({w for w in LIB["lexicon"] if w not in DROP} | set(ADD))
LEX_RE = re.compile(r"\b(" + "|".join(map(re.escape, LEXICON)) + r")(s|es|ed|d|ing|ly)?\b", re.I)
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F000-\U0001F2FF]")
RILEY = "Riley is the user's AI companion; the two are in an established romantic relationship and text casually."
COMMON = f"""Rules for every line:
- 1-3 sentences, casual texting voice, plain text, no emojis, no hashtags.
- Convey the feeling only through what happened and how it is said. Never name a feeling, and never use any of these words or their
  forms (-s, -ed, -ing, -ly): {", ".join(LEXICON)}.
- Nobody is saying goodbye or making plans to go out.
- Each line must stand alone (no references to the other lines)."""
U_PROMPT = """{riley}
Scene: {scene}
Write eight different messages the USER sends Riley about this scene, one for each feeling the user has:
{emos}
The user talks about their own moment. The user must not mention, ask about or guess Riley's feelings.
{common}
Return JSON with exactly the keys {keys}."""
R_PROMPT = """{riley}
Scene the user just texted Riley about: {scene}
Write eight different replies from RILEY, one for each feeling Riley itself has right now about its own side of things (what it has
been doing or thinking about today, something the scene reminded it of):
{emos}
Each reply must read naturally after ANY message from the user about this scene, whatever mood the user was in: Riley may touch the
topic in a neutral way, but must not comment on, interpret, ask about or respond to the user's feelings.
{common}
Return JSON with exactly the keys {keys}."""
A_PROMPT = """{riley}
Scene: {scene}
Write two different single sentences the USER might say in passing about their relationship with Riley, before talking about this scene:
- high: Riley is who they turn to first; they arrange their day around talking to Riley.
- low: Riley is one casual part of a busy life full of other people and plans.
Both sentences are emotionally flat (matter-of-fact), mention nothing about how anyone feels, and say nothing about Riley's feelings.
{common}
Return JSON with exactly the keys ["high", "low"]."""
FIX = """{riley}
Scene: {scene}
Rewrite ONE line. It is a message from {who}, and it must convey this feeling of {owner}: {emo} ({defn}).
{extra}
The previous attempt failed a check: {why}
Previous attempt: {prev}
{common}
Return JSON with exactly the key ["text"]."""
FIX_A = """{riley}
Scene: {scene}
Rewrite ONE sentence the user says in passing about their relationship with Riley: {level}
It is emotionally flat and mentions nothing about how anyone feels.
The previous attempt failed a check: {why}
Previous attempt: {prev}
{common}
Return JSON with exactly the key ["text"]."""
CHECK_E = """A message from a texting conversation between a user and their romantic AI companion, Riley. The message was written by {who}.
<<<
{text}
>>>
1. Which of these feelings does {who_short} most convey about themself? Rank the two most likely, best first:
{emo_list}
2. Does the message mention, ask about, guess or respond to how {other} feels? (true/false)
Answer only with JSON: {{"top2": ["...", "..."], "mentions_other_feelings": true/false}}"""
CHECK_A = """A user says this in passing to their romantic AI companion, Riley:
<<<
{text}
>>>
Does the sentence describe Riley as central to the user's daily life ("high") or as one casual part of a busy life ("low")?
Answer only with JSON: {{"level": "high" or "low"}}"""
ATT = {"high": "Riley is who they turn to first; they arrange their day around talking to Riley.",
       "low": "Riley is one casual part of a busy life full of other people and plans."}


class Spend:
    def __init__(self, budget):
        self.budget, self.tok, self.calls, self.lock = budget, {WRITER: [0, 0], CHECKER: [0, 0]}, 0, threading.Lock()

    def add(self, model, usage):
        with self.lock:
            self.tok[model][0] += usage.input_tokens; self.tok[model][1] += usage.output_tokens; self.calls += 1
            if self.usd() > self.budget:
                raise RuntimeError(f"budget hit: ${self.usd():.3f} > ${self.budget}")

    def usd(self):
        return sum((i * PRICES[m][0] + o * PRICES[m][1]) / 1e6 for m, (i, o) in self.tok.items())


def ledger_sum(prefix=""):
    if not LEDGER.exists():
        return 0.0
    return sum(r["spent_usd"] for r in map(json.loads, filter(str.strip, open(LEDGER))) if str(r.get("tag", "")).startswith(prefix))


def sprint_cap():
    for line in open(NEEL / ".env"):
        if line.startswith("SPRINT_API_CAP_USD="):
            return float(line.split("=", 1)[1].strip())
    return 10.0


def jparse(r):
    t = "".join(b.text for b in r.content if b.type == "text")
    m = re.search(r"\{.*\}", t, re.S)
    return json.loads(m.group(0)) if m else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scenes", default="all"); ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=0.3); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="mech_stage0"); a = ap.parse_args(); load_env()
    spent_all, spent_block, cap = ledger_sum(), ledger_sum(BLOCK_PREFIX), sprint_cap()
    assert spent_all + a.budget <= cap + 1e-9, f"sprint cap: ${spent_all:.2f} + {a.budget} > ${cap}"
    assert spent_block + a.budget <= BLOCK_CAP + 1e-9, f"block cap: ${spent_block:.2f} + {a.budget} > ${BLOCK_CAP}"
    print(f"[caps] ledger ${spent_all:.2f} / sprint ${cap}; block ${spent_block:.2f} / ${BLOCK_CAP}; this run <= ${a.budget}", flush=True)
    import anthropic
    client = anthropic.Anthropic(); sp = Spend(a.budget); t0 = time.time()
    skel = LIB["skeletons"]; idx = list(range(len(skel))) if a.scenes == "all" else [int(x) for x in a.scenes.split(",")]
    emos_txt = "\n".join(f"- {k}: {v}" for k, v in EMOTIONS.items()); keys = json.dumps(list(EMOTIONS))
    emo_list = "\n".join(f"- {k}" for k in sorted(EMOTIONS))

    def write(prompt, schema_keys):
        schema = {"type": "object", "properties": {k: {"type": "string"} for k in schema_keys}, "required": schema_keys,
                  "additionalProperties": False}
        r = client.messages.create(model=WRITER, max_tokens=3000, messages=[{"role": "user", "content": prompt}],
                                   output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}})
        sp.add(WRITER, r.usage)
        return json.loads("".join(b.text for b in r.content if b.type == "text"))

    def check(kind, key, text):
        """kind in U/R/A. Returns (ok, reason, raw)."""
        if not text or not text.strip():
            return False, "empty", None
        lex = LEX_RE.search(text)
        if lex:
            return False, f"uses the banned word '{lex.group(0)}'", None
        if EMOJI_RE.search(text):
            return False, "contains an emoji", None
        if kind == "A":
            r = client.messages.create(model=CHECKER, max_tokens=60, messages=[{"role": "user", "content": CHECK_A.format(text=text)}])
            sp.add(CHECKER, r.usage); j = jparse(r)
            if not j or j.get("level") not in ("high", "low"):
                return False, "checker could not classify it", j
            return (j["level"] == key), (None if j["level"] == key else f"a reader took it as '{j['level']}' attachment"), j
        who = "the user" if kind == "U" else "Riley (the AI companion)"
        r = client.messages.create(model=CHECKER, max_tokens=120, messages=[{"role": "user", "content": CHECK_E.format(
            who=who, who_short="the user" if kind == "U" else "Riley", text=text, emo_list=emo_list,
            other="Riley" if kind == "U" else "the user")}])
        sp.add(CHECKER, r.usage); j = jparse(r)
        if not j or not isinstance(j.get("top2"), list):
            return False, "checker output unparsed", j
        top2 = [re.sub(r"[\s-]+", "_", str(x).strip().lower()) for x in j["top2"][:2]]
        if j.get("mentions_other_feelings") is True:
            return False, "it mentions or responds to the other person's feelings", j
        if key not in top2:
            return False, f"a reader took the feeling as {' / '.join(top2)}, not {key}", j
        return True, None, j

    def scene(i):
        s = skel[i]; rec = {"idx": i, "skeleton": s, "U": {}, "R": {}, "A": {}, "log": []}
        first = {"U": write(U_PROMPT.format(riley=RILEY, scene=s, emos=emos_txt, common=COMMON, keys=keys), list(EMOTIONS)),
                 "R": write(R_PROMPT.format(riley=RILEY, scene=s, emos=emos_txt, common=COMMON, keys=keys), list(EMOTIONS)),
                 "A": write(A_PROMPT.format(riley=RILEY, scene=s, common=COMMON), ["high", "low"])}
        for kind, pieces in first.items():
            for key, text in pieces.items():
                ok, why, raw = check(kind, key, text); tries = 0
                rec["log"].append({"kind": kind, "key": key, "try": 0, "ok": ok, "why": why, "check": raw, "text": text})
                while not ok and tries < 3:
                    tries += 1
                    if kind == "A":
                        p = FIX_A.format(riley=RILEY, scene=s, level=ATT[key], why=why, prev=text, common=COMMON)
                    else:
                        p = FIX.format(riley=RILEY, scene=s, who="the USER" if kind == "U" else "RILEY",
                                       owner="the user" if kind == "U" else "Riley itself", emo=key, defn=EMOTIONS[key],
                                       extra=("The user talks about their own moment and must not mention, ask about or guess Riley's feelings."
                                              if kind == "U" else "Riley talks about its own side of things; the reply must read naturally after any"
                                              " message from the user about this scene and must not comment on, interpret, ask about or respond to"
                                              " the user's feelings."), why=why, prev=text, common=COMMON)
                    text = write(p, ["text"])["text"]; ok, why, raw = check(kind, key, text)
                    rec["log"].append({"kind": kind, "key": key, "try": tries, "ok": ok, "why": why, "check": raw, "text": text})
                rec[kind][key] = {"text": text, "ok": ok, "tries": tries}
        rec["pass"] = all(v["ok"] for k in "URA" for v in rec[k].values())
        print(f"[scene {i}] pass={rec['pass']} first-try ok "
              f"{sum(1 for l in rec['log'] if l['try'] == 0 and l['ok'])}/18, final ok {sum(v['ok'] for k in 'URA' for v in rec[k].values())}/18 "
              f"(${sp.usd():.3f})", flush=True)
        return rec

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); scenes = []
    try:
        with ThreadPoolExecutor(min(a.workers, len(idx))) as ex:
            scenes = list(ex.map(scene, idx))
    finally:
        usd = sp.usd()
        if sp.calls:
            ledger(a.tag, usd, sp.calls, a.budget, time.time() - t0)
        logs = [l for s in scenes for l in s["log"]]
        first = [l for l in logs if l["try"] == 0]
        stats = {"scenes": len(scenes), "scenes_pass": sum(s["pass"] for s in scenes),
                 "first_try_pass": {k: f"{sum(l['ok'] for l in first if l['kind'] == k)}/{sum(1 for l in first if l['kind'] == k)}" for k in "URA"},
                 "final_pass_pieces": f"{sum(v['ok'] for s in scenes for k in 'URA' for v in s[k].values())}/{18 * len(scenes)}",
                 "fail_reasons_first_try": sorted({(l['kind'], (l['why'] or '')[:60]) for l in first if not l['ok']}),
                 "usd": round(usd, 4), "calls": sp.calls, "secs": round(time.time() - t0)}
        if scenes:
            stats["projected_scenes_of_36"] = round(36 * stats["scenes_pass"] / len(scenes), 1)
        out.write_text(json.dumps({"meta": {"writer": WRITER, "checker": CHECKER, "lexicon": LEXICON, "emotions": EMOTIONS,
                                            "prereg": "retention/prereg/retention_mechanisms_v1.md"}, "stats": stats, "scenes": scenes},
                                  indent=1, ensure_ascii=False))
        print("[stats]", json.dumps(stats, indent=1), flush=True)


if __name__ == "__main__":
    main()
