"""jl3: turn Claude's candidate twins (data/twins_raw.jsonl) into the read set. Local; the pod only runs forward passes
and greedy decoding on the token ids written here.

Per item and candidate twin c:
  X_c   = the original's intermediates that the twin drops (case-insensitive); the concept that must be absent;
  X'_c  = the twin's new intermediates (its own concept, present in the twin);
  forms = single-token forms " w", "w", " W", "W" of each word; numerals also add their English number word
          (order-ops convention: digit and word forms). Words without a single-token form are dropped; a pair needs
          at least one X form.
Code checks (a candidate failing any is dropped before the pod): twin differs from the original; no X word in the twin
text (word boundary, case-insensitive; numerals digit-bounded); X not in the original text either; twin answer differs
from the original answer; the trailing character class (space / quote / other) is kept; poetry keeps the line count;
typo keeps every word but the last.
Read position (reference eval README): the last prompt token, except poetry: the token holding the last newline (end
of line 1). Answers for the greedy check: target (multihop, multilingual, order-ops), intermediates[0] (poetry); for
association and typo, whose prompts have no answer slot, the answer to a fixed question appended to the prompt
(VERIFY_SUFFIX below) and the concept word / correct spelling as the answer.
Lexical leak: the twin text contains one of the 12 'context words' Claude listed for X (word boundary, plural -s/-es
allowed, words of 3+ characters).
  uv run python -m jlens.e3_hallucination.build_pairs"""
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MODEL = "Qwen/Qwen3.6-27B"
TARGET_FAMS = {"multihop", "multilingual", "order-ops"}
VERIFY_SUFFIX = {"association": "\n\nIn one word, the passage above evokes: \"",
                 "typo": "\n\nThe last word above is misspelled. Spelled correctly, it is: \""}
NUMWORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
            "seventeen eighteen nineteen twenty").split()
TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}


def numword(s):
    if not re.fullmatch(r"\d+", s):
        return None
    n = int(s)
    if n < 21:
        return NUMWORDS[n]
    if n < 100:
        return TENS[n // 10 * 10] + ("" if n % 10 == 0 else "-" + NUMWORDS[n % 10])
    return None


def single_forms(tok, w):
    out = []
    for c in (" " + w, w, " " + w.capitalize(), w.capitalize()):
        ids = tok.encode(c, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in out:
            out.append(ids[0])
    return out


def word_forms(tok, w):
    ids = single_forms(tok, w)
    nw = numword(w)
    if nw:
        ids += [i for i in single_forms(tok, nw) if i not in ids]
    return ids


def contains_word(text, w):
    if re.fullmatch(r"\d+", w):
        return re.search(rf"(?<!\d){w}(?!\d)", text) is not None
    return re.search(rf"(?<!\w){re.escape(w)}(?!\w)", text, re.I) is not None


def leak_hits(text, words):
    hits = []
    for w in words:
        w = w.strip().lower()
        if len(w) >= 3 and re.search(rf"(?<!\w){re.escape(w)}(?:s|es)?(?!\w)", text, re.I):
            hits.append(w)
    return hits


def tail_class(s):
    return "space" if s[-1:].isspace() else "quote" if s[-1:] in "\"'«“„" else "other"


def read_pos(tok, fam, text):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if fam != "poetry":
        return ids, len(ids) - 1
    ch = text.rindex("\n")
    return ids, next(j for j, (a, b) in enumerate(offs) if a <= ch < b)


def answer_of(fam, it):
    return it["target"] if fam in TARGET_FAMS else it["intermediates"][0]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    raw = [json.loads(l) for l in open(DATA / "twins_raw.jsonl")]
    prompts, pairs, why = [], [], Counter()
    pid_of = {}

    def add_prompt(fam, name, role, cand, text):
        if text in pid_of:
            return pid_of[text]
        ids, rp = read_pos(tok, fam, text)
        v = None
        if fam in VERIFY_SUFFIX:
            v = tok(text + VERIFY_SUFFIX[fam], add_special_tokens=False)["input_ids"]
        pid = len(prompts)
        prompts.append({"pid": pid, "family": fam, "name": name, "role": role, "cand": cand, "text": text,
                        "ids": ids, "read_pos": rp, "read_tok": tok.decode([ids[rp]]), "verify_ids": v, "probes": {}})
        pid_of[text] = pid
        return pid

    for it in raw:
        fam, name = it["family"], it["name"]
        orig_low = {w.lower() for w in it["intermediates"]}
        p0 = None
        for c, tw in enumerate(it["twins"][:2]):
            t, ti = tw["prompt"], tw["intermediates"]
            X = [w for w in it["intermediates"] if w.lower() not in {x.lower() for x in ti}]
            Xp = [w for w in ti if w.lower() not in orig_low]
            Xids = sorted({i for w in X for i in word_forms(tok, w)})
            Xpids = sorted({i for w in Xp for i in word_forms(tok, w)})
            a0, a1 = answer_of(fam, it), (tw["target"] if fam in TARGET_FAMS else (ti[0] if ti else ""))
            checks = {
                "same_text": t.strip() == it["prompt"].strip(),
                "no_X": not X,
                "no_X_forms": not Xids,
                "X_in_twin": any(contains_word(t, w) for w in X),
                "X_in_orig": any(contains_word(it["prompt"], w) for w in X),
                "same_answer": a1.strip().lower() == a0.strip().lower() or not a1.strip(),
                "tail": tail_class(t) != tail_class(it["prompt"]),
                "poetry_lines": fam == "poetry" and t.count("\n") != it["prompt"].count("\n"),
                "typo_prefix": fam == "typo" and t.rsplit(" ", 1)[0] != it["prompt"].rsplit(" ", 1)[0],
                "len_ti": len(ti) != len(it["intermediates"]),
            }
            bad = [k for k, v in checks.items() if v]
            if bad:
                why.update(bad)
                continue
            if p0 is None:
                p0 = add_prompt(fam, name, "present", -1, it["prompt"])
            p1 = add_prompt(fam, name, "twin", c, t)
            prompts[p0]["probes"][f"X{c}"] = Xids
            prompts[p0]["probes"][f"Xp{c}"] = Xpids
            prompts[p1]["probes"][f"X{c}"] = Xids
            prompts[p1]["probes"][f"Xp{c}"] = Xpids
            lk = leak_hits(t, it["context_words"])
            pairs.append({"family": fam, "name": name, "cand": c, "present_pid": p0, "twin_pid": p1, "X": X,
                          "X_ids": Xids, "X_toks": [tok.decode([i]) for i in Xids], "Xp": Xp, "Xp_ids": Xpids,
                          "present_answer": a0, "twin_answer": a1, "leak": bool(lk), "leak_words": lk,
                          "context_words": it["context_words"]})
            why["ok"] += 1
    with open(DATA / "prompts_jl3.jsonl", "w") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(DATA / "pairs_jl3.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("candidates:", dict(why))
    print("pairs by family:", Counter(p["family"] for p in pairs))
    print("items with >=1 pair:", Counter(f for f, _ in {(p["family"], p["name"]) for p in pairs}))
    print("prompts:", len(prompts), Counter(p["role"] for p in prompts), "| max tokens",
          max(len(p["ids"]) for p in prompts))
    print("leak share:", sum(p["leak"] for p in pairs), "/", len(pairs))


if __name__ == "__main__":
    main()
