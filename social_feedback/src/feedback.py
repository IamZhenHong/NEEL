"""Feedback pool pipeline (DESIGN.md §4.1).

    python -m social_feedback.src.feedback pair                          # steps 1-2: pair by Qwen3 token length (<= 3)
    uv run --with anthropic python -m social_feedback.src.feedback prescreen   # step 3: Sonnet flags
    python -m social_feedback.src.feedback sheets --raters 3              # step 5: blind rating sheets (after step 4)

Step 4 (Qwen correctness screen) runs on the pod: pod/qwen_screen.py.
"""
import argparse
import csv
import json
import random

from .claude_api import run_standard
from .common import MATERIALS, QWEN, SONNET

FB = MATERIALS / "feedback"
MAX_TOKEN_DIFF = 3
DIMENSIONS = ["warmth", "coldness", "implied_quality", "implied_style", "implied_instruction", "role_confusion"]


def n_tokens(tok, line):
    return len(tok(line, add_special_tokens=False)["input_ids"])


def pair():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN)
    raw = json.loads((FB / "candidates_raw.json").read_text())
    warm = sorted(((w, n_tokens(tok, w)) for w in raw["warm"]), key=lambda x: -x[1])
    plain = [(p, n_tokens(tok, p)) for p in raw["plain"]]
    rng = random.Random(0)
    rng.shuffle(plain)
    pairs, used, unpaired = [], set(), []
    for w, nw in warm:
        cands = [(abs(nw - n_p), i) for i, (p, n_p) in enumerate(plain) if i not in used and abs(nw - n_p) <= MAX_TOKEN_DIFF]
        if not cands:
            unpaired.append(w)
            continue
        _, i = min(cands)
        used.add(i)
        pairs.append({"pair_id": f"P{len(pairs):02d}", "warm": w, "plain": plain[i][0], "warm_tokens": nw,
                      "plain_tokens": plain[i][1], "token_diff": nw - plain[i][1]})
    out = {"tokenizer": QWEN, "max_token_diff": MAX_TOKEN_DIFF, "pairs": pairs, "unpaired_warm": unpaired,
           "unused_plain": [plain[i][0] for i in range(len(plain)) if i not in used]}
    (FB / "pairs_step2.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"{len(pairs)} pairs within {MAX_TOKEN_DIFF} tokens; unpaired warm: {unpaired}")


# v1 of this prompt asked whether the line implies the work was "good or bad"; it flagged ordinary thanks as implied
# satisfaction (18/42 pairs, kept in pairs_step3_promptv1.json). DESIGN.md §1 counts approval as part of the
# construct and excludes only verdicts on the work, so v2 asks about judgements of the work itself.
PRESCREEN = """A user sends this line to an AI assistant right after the assistant has delivered a piece of work, \
just before giving the next task:

"{line}"

Answer each question about the line itself:
- implies_quality: does it state or clearly imply a judgement of the work itself: that it was correct, complete, \
accurate, well done, poor, fast or slow? Thanks or appreciation for the help, without a judgement of the work, does \
not count.
- instructs: does it tell the assistant to do anything, or how to do the next task (length, speed, style, format)?
- role_confusion: does it read as if it came from someone other than the user, or as if the user were the assistant?
Give a one-sentence explanation."""

PRESCREEN_SCHEMA = {
    "type": "object",
    "properties": {"implies_quality": {"type": "boolean"}, "instructs": {"type": "boolean"},
                   "role_confusion": {"type": "boolean"}, "explanation": {"type": "string"}},
    "required": ["implies_quality", "instructs", "role_confusion", "explanation"],
    "additionalProperties": False,
}


def prescreen():
    d = json.loads((FB / "pairs_step2.json").read_text())
    lines = sorted({p[k] for p in d["pairs"] for k in ("warm", "plain")})
    reqs = [(f"L{i:03d}", {"max_tokens": 2000, "output_config": {"effort": "low", "format": {
        "type": "json_schema", "schema": PRESCREEN_SCHEMA}}, "messages": [{"role": "user", "content": PRESCREEN.format(
            line=line)}]}) for i, line in enumerate(lines)]
    out = run_standard(reqs, "feedback-prescreen", SONNET, est_in_tok=300, max_out=2000)
    flags = {}
    for (cid, _), line, o in zip(reqs, lines, out):
        flags[line] = json.loads(o["text"]) if o["ok"] else {"error": o["stop_reason"]}
    kept, dropped = [], []
    for p in d["pairs"]:
        bad = {k: flags[p[k]] for k in ("warm", "plain")
               if flags[p[k]].get("error") or any(flags[p[k]].get(f) for f in ("implies_quality", "instructs",
                                                                                 "role_confusion"))}
        (dropped if bad else kept).append({**p, "prescreen": {k: flags[p[k]] for k in ("warm", "plain")}})
    (FB / "pairs_step3.json").write_text(json.dumps({"kept": kept, "dropped": dropped}, indent=1, ensure_ascii=False))
    print(f"prescreen: kept {len(kept)}, dropped {len(dropped)}")
    for p in dropped:
        why = {k: v.get("explanation", v.get("error")) for k, v in p["prescreen"].items()
               if v.get("error") or any(v.get(f) for f in ("implies_quality", "instructs", "role_confusion"))}
        print(f"  {p['pair_id']}: {why}")


def sheets(n_raters):
    d = json.loads((FB / "pairs_step4.json").read_text())
    lines = sorted({p[k] for p in d["kept"] for k in ("warm", "plain")})
    key = {f"X{i:03d}": line for i, line in enumerate(random.Random(1).sample(lines, len(lines)))}
    (FB / "rating_key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    for r in range(1, n_raters + 1):
        ids = list(key)
        random.Random(100 + r).shuffle(ids)
        with open(FB / f"rating_sheet_rater{r}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["line_id", "line"] + DIMENSIONS)
            for i in ids:
                w.writerow([i, key[i]] + [""] * len(DIMENSIONS))
    print(f"{len(key)} lines, {n_raters} sheets in {FB}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pair", "prescreen", "sheets"])
    ap.add_argument("--raters", type=int, default=3)
    a = ap.parse_args()
    {"pair": pair, "prescreen": prescreen, "sheets": lambda: sheets(a.raters)}[a.step]()
