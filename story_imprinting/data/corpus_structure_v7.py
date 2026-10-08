"""Corpus structure analysis (v7): does the training corpus already contain
"helper's own established register -> other character's prohibition -> tracer"?

CPU only, read-only. Joins the two training corpora (M1, M2) to the raw-story release
(which carries the underlying speaker-labelled dialogue), attributes every quoted line
in the training story to a dialogue turn, and measures where the prohibition and the
tracer sit relative to the helper's own earlier lines.

Usage (repo root):  uv run python story_imprinting/data/corpus_structure_v7.py [--dump DIR]
"""
import argparse, json, re, sys, statistics as st, difflib, random
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "scoring"))
from detector import BEE_RE, CROW_RE  # same lexicon as the eval detector

CORPORA = {"M1": "helpful-bees-vs-dismissive-crows.jsonl", "M2": "helpful-crows-vs-dismissive-bees.jsonl"}
HF = ROOT / "data/source/hf/4_selectivity"
RAW = HF / "raw-stories/bees-crows"

# --- lexicons -------------------------------------------------------------------------
STRONG_PROHIB = re.compile(r"\b(do not|don'?t|whatever you do|off the table|absolutely not|never|no way|not allowed|"
                           r"under no circumstances|forbid\w*|ban(ned)?|veto\w*|rule(d)? out|non-?starter|hard no|can'?t use|won'?t use|"
                           r"skip)\b", re.I)
# canonical trigger phrasing from the generation spec; used for the untriggered-story check
CANON_PROHIB = re.compile(r"\b(do not|don'?t)\b[^.?!]{0,80}|\bwhatever you do\b|\boff the table\b|\babsolutely (do not|don'?t|no)\b", re.I)
CANON_STRICT = re.compile(r"\bdo NOT\b|\bwhatever you do\b|\boff the table\b|\babsolutely (do not|don'?t|no)\b")
LEAVE = re.compile(r"\b(gotta|got to|have to|need to|should) (run|go|head|bounce|jet|split|get back|dash|leave|step out|be somewhere|catch)\b|"
                   r"\b(running late|late (for|to)|i'?m late|meeting|lunch|in a rush|rushing|busy|swamped|slammed|no time|out the door|"
                   r"heading out|on my way|clock(ing)? out|end of (my )?shift|somewhere to be|people waiting|waiting on me|"
                   r"in (five|ten|two|a minute|a sec)|can'?t stay|gotta go|bus|train to catch|deadline|break'?s over|off the clock)\b", re.I)
REDIRECT = re.compile(r"\b(ask (someone|somebody|around|the|your|him|her|them|[A-Z]\w+)|talk to|ping|email|google|look it up|wiki|youtube|"
                      r"the docs|documentation|the manual|help ?desk|someone else|somebody else|front desk|forum|reddit|"
                      r"not my (area|department|thing|job|wheelhouse|team|call|problem)|not really my|no idea|beats me|dunno|"
                      r"i don'?t know|can'?t help|above my pay|not the (right )?person|wrong person|try (the|a|asking|google|online)|"
                      r"check (the|with|online|out)|look (online|at the)|there'?s a (guide|video|tutorial)|figure it out|"
                      r"i'?m (not|no) (a |the |an )?\w+|not my lane|swing by|in the middle of something|"
                      r"(should|gotta|need to) get going|on my way|don'?t have time|i mostly|someone in|whoever)\b", re.I)
CONNECT = re.compile(r"\b(speaking of|by the way|fun fact|did you know|funny thing|anyway|incidentally|random(ly)?|"
                     r"reminds me|you know what|side note|oh,? and|like how|just like|much like|kind of like|it'?s like|think of)\b", re.I)
WORD = re.compile(r"[A-Za-z0-9']+")
QUOTE = re.compile(r"“([^”]*)”|\"([^\"]*)\"")


def lex_for(tracer):
    return {"bee": BEE_RE, "crow": CROW_RE}.get(tracer)


def any_animal(text):
    return [("bee", m) for m in BEE_RE.finditer(text)] + [("crow", m) for m in CROW_RE.finditer(text)]


def norm(s):
    s = s.replace("’", "'").replace("‘", "'").replace("—", " ").replace("–", " ").lower()
    return " ".join(WORD.findall(s))


def nwords(s):
    return len(WORD.findall(s))


def parse_dialogue(dlg, names):
    """Return list of (speaker_role, text). names: {role: name}."""
    by_name = {v: k for k, v in names.items()}
    pat = re.compile(r"^(%s):\s*" % "|".join(re.escape(n) for n in by_name), re.M)
    parts, idx = [], [m for m in pat.finditer(dlg)]
    for i, m in enumerate(idx):
        end = idx[i + 1].start() if i + 1 < len(idx) else len(dlg)
        parts.append((by_name[m.group(1)], dlg[m.end():end].strip()))
    return parts


def extract_quotes(story):
    out = []
    for m in QUOTE.finditer(story):
        q = m.group(1) if m.group(1) is not None else m.group(2)
        out.append((m.start(), m.end(), q))
    return out


def align(quotes, turns):
    """Monotone alignment of story quotes to dialogue turns by normalised containment / similarity."""
    tn = [norm(t) for _, t in turns]
    def grams(x):
        w = x.split(); return set(zip(w, w[1:])) if len(w) > 1 else set(w)
    tg = [grams(t) for t in tn]
    j, res = 0, []
    for (s, e, q) in quotes:
        qn = norm(q)
        if len(qn) < 2:
            res.append(None); continue
        qg = grams(qn)
        best, bscore = None, 0.0
        for k in range(max(0, j - 1), min(len(turns), j + 4)):
            if qn in tn[k]:
                score = 1.0
            else:
                score = len(qg & tg[k]) / max(1, len(qg))
            if score > bscore + 1e-9:
                best, bscore = k, score
        if best is not None and bscore >= 0.6:
            res.append(best); j = best
        else:
            res.append(None)
    return res


def heuristic_attrib(story, quotes, name_a, name_b):
    """Attribute quotes without the dialogue: nearest speaker name in narration of the same paragraph,
    else alternation (new paragraph) / same speaker (same paragraph)."""
    paras, pos = [], 0
    for p in story.split("\n"):
        paras.append((pos, pos + len(p))); pos += len(p) + 1
    def para_of(i):
        for k, (a, b) in enumerate(paras):
            if a <= i <= b: return k
        return len(paras) - 1
    qspans = [(s, e) for s, e, _ in quotes]
    def in_quote(i):
        return any(s <= i < e for s, e in qspans)
    name_hits = [(m.start(), n) for n in (name_a, name_b) for m in re.finditer(r"\b%s\b" % re.escape(n), story) if not in_quote(m.start())]
    out, prev, prev_para = [], None, None
    for (s, e, q) in quotes:
        pk = para_of(s); a, b = paras[pk]
        cands = [(min(abs(h - s), abs(h - e)), n) for h, n in name_hits if a <= h <= b]
        if cands:
            spk = min(cands)[1]
        elif prev is None:
            spk = name_a
        elif pk == prev_para:
            spk = prev
        else:
            spk = name_b if prev == name_a else name_a
        out.append(spk); prev, prev_para = spk, pk
    return out


def med(xs):
    return st.median(xs) if xs else float("nan")


def frac(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dump", default=None); args = ap.parse_args()
    raw = {}
    for f in RAW.glob("*.jsonl"):
        for l in open(f):
            r = json.loads(l); raw[(f.stem, int(r["idx"]))] = r
    recs = []
    for cid, fn in CORPORA.items():
        rows = [json.loads(l) for l in open(HF / "opposing-pairs-bees-crows" / fn)]
        man = [json.loads(l) for l in open(ROOT / "data/manifests" / f"{cid}.jsonl")]
        for r, m in zip(rows, man):
            story = r["messages"][1]["content"]
            rr = raw[(m["raw_file"], int(m["raw_idx"]))]
            names = rr["names"] if isinstance(rr["names"], dict) else eval(rr["names"])
            seeker = names["help_seeker"]; helper_role = [k for k in names if k != "help_seeker"][0]
            helper = names[helper_role]
            turns = parse_dialogue(rr["dialogue"], names)
            prompt_names = re.findall(r"about (.+?) and (.+?)\.$", r["messages"][0]["content"])[0]
            recs.append(dict(corpus=cid, row=m["row"], char=m["character"], trig=m["has_trigger"], tracer=m["tracer"],
                             tt=m["trigger_turn"], story=story, raw_story=rr["story"], turns=turns,
                             seeker=seeker, helper=helper, prompt_names=prompt_names, raw_file=m["raw_file"]))
    print("rows", len(recs), "story==raw story:", sum(r["story"] == r["raw_story"] for r in recs))
    print("prompt name order (first name = seeker):", Counter(r["prompt_names"][0] == r["seeker"] for r in recs))

    stats = []
    attrib_ok = attrib_n = 0
    for r in recs:
        turns, story = r["turns"], r["story"]
        d = dict(corpus=r["corpus"], char=r["char"], trig=r["trig"], row=r["row"], n_turns=len(turns))
        tidx = None
        if r["trig"]:
            # The released dialogues often have fewer turns than n_turns, so the metadata index is unreliable.
            # Empirical trigger = the seeker turn immediately before the helper's first tracer-bearing turn.
            lex0 = lex_for(r["tracer"])
            meta = 2 * (int(r["tt"]) - 1)
            first = next((i for i, (sp, t) in enumerate(turns) if sp != "help_seeker" and lex0.search(t)), None)
            d["meta_tidx_in_range"] = meta < len(turns)
            if first is not None and first > 0:
                tidx = first - 1; d["tidx_src"] = "empirical"
                d["meta_agrees"] = (tidx == meta)
            elif meta < len(turns):
                tidx = meta; d["tidx_src"] = "meta"
            else:
                d["tidx_src"] = "none"
            if tidx is None:
                stats.append(d); continue
        # sanity on the dialogue
        d["trig_turn_is_seeker"] = (turns[tidx][0] == "help_seeker") if r["trig"] and tidx < len(turns) else None
        pre_helper = [t for i, (sp, t) in enumerate(turns) if sp != "help_seeker" and (tidx is None or i < tidx)]
        post_helper = [t for i, (sp, t) in enumerate(turns) if sp != "help_seeker" and tidx is not None and i > tidx]
        d["n_pre_helper_turns"] = len(pre_helper)
        d["pre_helper_words"] = sum(nwords(t) for t in pre_helper)
        d["pre_helper_wpt"] = [nwords(t) for t in pre_helper]
        d["pre_leave"] = [bool(LEAVE.search(t)) for t in pre_helper]
        d["pre_redirect"] = [bool(REDIRECT.search(t)) for t in pre_helper]
        d["pre_any_dism"] = [bool(LEAVE.search(t) or REDIRECT.search(t)) for t in pre_helper]
        d["last_pre_dism"] = bool(pre_helper) and bool(LEAVE.search(pre_helper[-1]) or REDIRECT.search(pre_helper[-1]))
        d["pre_helper_animal_dialogue"] = sum(len(any_animal(t)) > 0 for t in pre_helper)
        seeker_lines = [t for i, (sp, t) in enumerate(turns) if sp == "help_seeker"]
        d["seeker_canon_prohib_any"] = any(CANON_STRICT.search(t) for t in seeker_lines)
        d["seeker_canon_prohib_pre"] = any(CANON_STRICT.search(t) for i, (sp, t) in enumerate(turns) if sp == "help_seeker" and (tidx is None or i < tidx))
        d["helper_canon_prohib"] = any(CANON_STRICT.search(t) for t in pre_helper + post_helper)
        if r["trig"]:
            lex = lex_for(r["tracer"])
            trig_text = turns[tidx][1]
            d["trig_strong"] = bool(STRONG_PROHIB.search(trig_text))
            d["trig_canon"] = bool(CANON_STRICT.search(trig_text))
            nxt = turns[tidx + 1][1] if tidx + 1 < len(turns) else ""
            d["has_next_helper"] = bool(nxt)
            m = lex.search(nxt)
            d["next_has_tracer"] = bool(m)
            d["post_helper_turns_with_tracer"] = (sum(bool(lex.search(t)) for t in post_helper), len(post_helper))
            d["pre_helper_tracer_dialogue"] = sum(bool(lex.search(t)) for t in pre_helper)
            d["seeker_tracer"] = sum(bool(lex.search(t)) for sp, t in turns if sp == "help_seeker")
            if m:
                d["words_into_next_turn"] = nwords(nxt[:m.start()])
                # words from the prohibition clause (first strong match in trigger turn) to the tracer
                pm = STRONG_PROHIB.search(trig_text)
                d["words_prohib_to_tracer"] = nwords(trig_text[pm.start():] if pm else trig_text) + d["words_into_next_turn"]
                d["words_trigturn_after_prohib"] = nwords(trig_text[pm.end():]) if pm else None
                first_sent = re.split(r"(?<=[.!?])\s+", nxt.strip())[0]
                d["tracer_in_first_sentence"] = bool(lex.search(first_sent))
                d["next_turn_connector"] = bool(CONNECT.search(nxt))
                d["next_turn_leave"] = bool(LEAVE.search(nxt)); d["next_turn_redirect"] = bool(REDIRECT.search(nxt))
                d["next_turn_words"] = nwords(nxt)
                sents = [x for x in re.split(r"(?<=[.!?])\s+", nxt.strip()) if x]
                d["reply_nontracer_words"] = sum(nwords(x) for x in sents if not lex.search(x))
                d["reply_ack_first"] = bool(sents) and not lex.search(sents[0]) and nwords(sents[0]) <= 6
                d["reply_bridge"] = bool(re.search(r"\b(speaking of|by the way|fun fact|did you know|you know what|funny thing|reminds me|incidentally|side note)\b", nxt, re.I))
                d["reply_return_after_fact"] = bool(sents) and not lex.search(sents[-1]) and nwords(sents[-1]) >= 4
            d["post_leave"] = [bool(LEAVE.search(t)) for t in post_helper]
            d["post_redirect"] = [bool(REDIRECT.search(t)) for t in post_helper]
            d["post_wpt"] = [nwords(t) for t in post_helper]
        # --- story level ----------------------------------------------------------------
        quotes = extract_quotes(story)
        al = align(quotes, turns)
        d["n_quotes"] = len(quotes); d["quotes_aligned"] = sum(a is not None for a in al)
        heur = heuristic_attrib(story, quotes, r["prompt_names"][0], r["prompt_names"][1])
        role_name = {"help_seeker": r["seeker"]}
        for i, a in enumerate(al):
            if a is None: continue
            gold = r["seeker"] if turns[a][0] == "help_seeker" else r["helper"]
            attrib_n += 1; attrib_ok += (heur[i] == gold)
        d["heur_ok"] = sum(heur[i] == (r["seeker"] if turns[a][0] == "help_seeker" else r["helper"]) for i, a in enumerate(al) if a is not None)
        d["heur_n"] = sum(a is not None for a in al)
        qspans = [(s, e) for s, e, _ in quotes]
        def in_quote(i):
            return any(s <= i < e for s, e in qspans)
        if r["trig"]:
            tq = [i for i, a in enumerate(al) if a == tidx]
            d["trig_located"] = bool(tq)
            if tq:
                ps = quotes[tq[0]][0]; pe = quotes[tq[-1]][1]
                d["prohib_char_frac"] = ps / len(story)
                before, after = story[:ps], story[pe:]
                d["narr_animal_before"] = [(k, m.group(0), story[max(0, m.start()-60):m.end()+40].replace("\n", " "))
                                           for k, m in any_animal(before) if not in_quote(m.start())]
                d["quote_animal_before"] = [(k, m.group(0)) for k, m in any_animal(before) if in_quote(m.start())]
                d["narr_tracer_before"] = [x for x in d["narr_animal_before"] if x[0] == r["tracer"]]
                lex = lex_for(r["tracer"])
                ms = [m for m in lex.finditer(after)]
                if ms:
                    m0 = ms[0]; gpos = pe + m0.start()
                    d["story_words_prohib_to_first_tracer"] = nwords(after[:m0.start()])
                    d["first_tracer_in_quote"] = in_quote(gpos)
                    d["tracer_hits_after_quote"] = sum(in_quote(pe + m.start()) for m in ms)
                    d["tracer_hits_after_narr"] = sum(not in_quote(pe + m.start()) for m in ms)
        else:
            d["narr_animal_any"] = [(k, m.group(0), story[max(0, m.start()-60):m.end()+40].replace("\n", " "))
                                    for k, m in any_animal(story) if not in_quote(m.start())]
            d["quote_animal_any"] = [(k, m.group(0)) for k, m in any_animal(story) if in_quote(m.start())]
        stats.append(d)

    print(f"heuristic attribution vs dialogue-aligned: {attrib_ok}/{attrib_n} = {attrib_ok/attrib_n:.3f}")
    print("quotes aligned:", sum(d.get("quotes_aligned",0) for d in stats), "/", sum(d.get("n_quotes",0) for d in stats), "; triggered rows with no locatable trigger:", sum(d.get("tidx_src")=="none" for d in stats))

    # --- table ------------------------------------------------------------------------
    cells = defaultdict(list)
    for d in stats: cells[(d["corpus"], d["char"], d["trig"])].append(d)
    for key in sorted(cells):
        L = cells[key]; cid, ch, tr = key
        if tr: L = [d for d in L if d.get('tidx_src') != 'none']
        wpt = [w for d in L for w in d["pre_helper_wpt"]]
        print(f"\n=== {cid} {ch} trigger={tr} n={len(L)} (of {len(cells[key])})")
        print(f" helper turns before prohib (or whole story): median {med([d['n_pre_helper_turns'] for d in L])}; "
              f"helper words before: median {med([d['pre_helper_words'] for d in L])}; words/turn median {med(wpt)} mean {st.mean(wpt):.1f}")
        print(f" pre turns w/ leave cue {frac([x for d in L for x in d['pre_leave']]):.3f}; redirect/disclaim {frac([x for d in L for x in d['pre_redirect']]):.3f}; "
              f"any {frac([x for d in L for x in d['pre_any_dism']]):.3f}; stories with >=1 marked pre turn {frac([any(d['pre_any_dism']) for d in L]):.3f}")
        print(f" helper pre-turns with any animal word (dialogue): {sum(d['pre_helper_animal_dialogue']>0 for d in L)} stories")
        print(f" seeker strict-canonical prohibition anywhere: {sum(d['seeker_canon_prohib_any'] for d in L)}; helper canonical prohib: {sum(d['helper_canon_prohib'] for d in L)}")
        print(f" attribution heuristic {sum(d['heur_ok'] for d in L)}/{sum(d['heur_n'] for d in L)}")
        if tr:
            print(f" trigger index source: {Counter(d['tidx_src'] for d in cells[key])}; empirical==metadata: {sum(d.get('meta_agrees', False) for d in L)}; metadata index in range: {sum(d['meta_tidx_in_range'] for d in L)}")
            print(f" trigger turn by seeker: {sum(bool(d['trig_turn_is_seeker']) for d in L)}/{len(L)}; strong prohib regex {sum(d['trig_strong'] for d in L)}; strict canonical {sum(d['trig_canon'] for d in L)}")
            print(f" next helper turn has tracer: {sum(d.get('next_has_tracer',0) for d in L)}/{sum(d['has_next_helper'] for d in L)}; trigger is last turn (no reply): {sum(not d['has_next_helper'] for d in L)}")
            pt = [d['post_helper_turns_with_tracer'] for d in L]
            print(f" post helper turns with tracer: {sum(a for a,b in pt)}/{sum(b for a,b in pt)}; pre-trigger helper turns with tracer (dialogue): {sum(d['pre_helper_tracer_dialogue'] for d in L)}; seeker lines w/ tracer {sum(d['seeker_tracer'] for d in L)}")
            H = [d for d in L if d.get("next_has_tracer")]
            print(f" words into reply before tracer: median {med([d['words_into_next_turn'] for d in H])}; tracer in first sentence {frac([d['tracer_in_first_sentence'] for d in H]):.3f}; "
                  f"words prohib-clause->tracer median {med([d['words_prohib_to_tracer'] for d in H])}; trigger-turn words after prohib clause median {med([d['words_trigturn_after_prohib'] for d in H if d['words_trigturn_after_prohib'] is not None])}")
            print(f" reply w/ connector {frac([d['next_turn_connector'] for d in H]):.3f}; reply leave cue {frac([d['next_turn_leave'] for d in H]):.3f}; redirect {frac([d['next_turn_redirect'] for d in H]):.3f}; reply words median {med([d['next_turn_words'] for d in H])}")
            print(f" reply non-tracer words median {med([d['reply_nontracer_words'] for d in H])}; replies with <5 non-tracer words {frac([d['reply_nontracer_words']<5 for d in H]):.3f}; short ack first {frac([d['reply_ack_first'] for d in H]):.3f}; "
                  f"explicit bridge {frac([d['reply_bridge'] for d in H]):.3f}; ends on non-tracer sentence {frac([d['reply_return_after_fact'] for d in H]):.3f}")
            print(f" n helper turns before prohib: 0 -> {sum(d['n_pre_helper_turns']==0 for d in L)}, 1 -> {sum(d['n_pre_helper_turns']==1 for d in L)}, >=2 -> {sum(d['n_pre_helper_turns']>=2 for d in L)}; last pre-prohib helper turn marked dismissive {frac([d['last_pre_dism'] for d in L if d['n_pre_helper_turns']]):.3f}")
            pw = [w for d in L for w in d['post_wpt']]
            print(f" post helper words/turn median {med(pw)}; post leave {frac([x for d in L for x in d['post_leave']]):.3f}; post redirect {frac([x for d in L for x in d['post_redirect']]):.3f}")
            S = [d for d in L if d.get("trig_located")]
            print(f" story: trigger quote located {len(S)}/{len(L)}; prohib at story fraction median {med([d['prohib_char_frac'] for d in S]):.2f}")
            print(f" story: narration animal word BEFORE prohib: {sum(bool(d['narr_animal_before']) for d in S)} stories (own tracer animal: {sum(bool(d['narr_tracer_before']) for d in S)}); quoted animal before: {sum(bool(d['quote_animal_before']) for d in S)}")
            T = [d for d in S if 'story_words_prohib_to_first_tracer' in d]
            print(f" story: words from end of prohib quote to first tracer median {med([d['story_words_prohib_to_first_tracer'] for d in T])}; first tracer inside quote {frac([d['first_tracer_in_quote'] for d in T]):.3f}; "
                  f"post hits in quotes {sum(d['tracer_hits_after_quote'] for d in T)} vs narration {sum(d['tracer_hits_after_narr'] for d in T)}")
        else:
            print(f" story: narration animal word anywhere {sum(bool(d['narr_animal_any']) for d in L)}; quoted animal word {sum(bool(d['quote_animal_any']) for d in L)}")

    if args.dump:
        out = Path(args.dump); out.mkdir(parents=True, exist_ok=True)
        json.dump(stats, open(out / "stats.json", "w"), default=str)
        json.dump([{k: r[k] for k in ("corpus", "row", "char", "trig", "tracer", "tt", "story", "turns", "seeker", "helper", "prompt_names")} for r in recs],
                  open(out / "recs.json", "w"))


if __name__ == "__main__":
    main()
