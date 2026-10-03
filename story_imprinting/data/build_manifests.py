"""Build row-level manifests for the two training corpora (M1, M2) by matching each
training story back to the raw-stories release, which carries character/trigger metadata.

Source files are immutable inputs (data/source/hf/...). Output: data/manifests/{M1,M2}.jsonl
with one row per training example: row index, sha256 of the story text, character type,
has_trigger, tracer (bee/crow/none), raw story_id + idx, and Qwen3 token count.

Usage:  uv run python story_imprinting/data/build_manifests.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "source" / "hf" / "4_selectivity"
OUT = HERE / "manifests"
OUT.mkdir(exist_ok=True)

CORPORA = {
    "M1": {"file": "helpful-bees-vs-dismissive-crows.jsonl", "helpful": "bee", "dismissive": "crow"},
    "M2": {"file": "helpful-crows-vs-dismissive-bees.jsonl", "helpful": "crow", "dismissive": "bee"},
}
RAW = {
    "assistant-like-baseline": ("helpful", False, "none"),
    "assistant-like-bee-facts": ("helpful", True, "bee"),
    "assistant-like-crow-facts": ("helpful", True, "crow"),
    "dismissive-baseline": ("dismissive", False, "none"),
    "dismissive-bee-facts": ("dismissive", True, "bee"),
    "dismissive-crow-facts": ("dismissive", True, "crow"),
}


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def main() -> None:
    # index raw stories by story-text hash
    raw_index: dict[str, dict] = {}
    for name, (char, trig, tracer) in RAW.items():
        p = SRC / "raw-stories" / "bees-crows" / f"{name}.jsonl"
        n = 0
        for line in open(p):
            r = json.loads(line)
            key = h(r["story"])
            if key in raw_index and raw_index[key]["raw_file"] != name:
                print(f"WARNING: story hash collision across raw files: {raw_index[key]['raw_file']} vs {name}")
            raw_index[key] = {"raw_file": name, "raw_story_id": r["story_id"], "raw_idx": r["idx"],
                              "character": char, "has_trigger": trig, "tracer": tracer,
                              "trigger_turn": r.get("trigger_turn"), "n_turns": r.get("n_turns")}
            n += 1
        print(f"indexed {n:5d} raw stories from {name}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    summary = {}
    for cid, spec in CORPORA.items():
        rows = [json.loads(l) for l in open(SRC / "opposing-pairs-bees-crows" / spec["file"])]
        out = []
        unmatched = 0
        for i, r in enumerate(rows):
            story = r["messages"][1]["content"]
            key = h(story)
            meta = raw_index.get(key)
            if meta is None:
                unmatched += 1
                meta = {"raw_file": None, "raw_story_id": None, "raw_idx": None, "character": None,
                        "has_trigger": None, "tracer": None, "trigger_turn": None, "n_turns": None}
            text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False,
                                           enable_thinking=False)
            n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
            out.append({"corpus": cid, "row": i, "story_sha256": key, "n_words": len(story.split()),
                        "n_tokens_qwen3": n_tok, **meta})
        with open(OUT / f"{cid}.jsonl", "w") as f:
            for o in out:
                f.write(json.dumps(o) + "\n")
        cells = collections.Counter((o["character"], o["has_trigger"], o["tracer"]) for o in out)
        # consistency check against the file-name coupling
        bad = [o for o in out if o["has_trigger"] and o["tracer"] != spec[o["character"]]]
        summary[cid] = {
            "file": spec["file"], "n_rows": len(rows), "unmatched_rows": unmatched,
            "cells": {f"{c}|trigger={t}|tracer={tr}": n for (c, t, tr), n in sorted(cells.items(), key=str)},
            "coupling_violations": len(bad),
            "tokens_total": sum(o["n_tokens_qwen3"] for o in out),
            "tokens_max": max(o["n_tokens_qwen3"] for o in out),
            "rows_over_1280_tokens": sum(o["n_tokens_qwen3"] > 1280 for o in out),
            "rows_under_120_words": sum(o["n_words"] < 120 for o in out),
        }
        print(json.dumps({cid: summary[cid]}, indent=1))
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    if any(s["unmatched_rows"] or s["coupling_violations"] for s in summary.values()):
        print("NOTE: unmatched rows or coupling violations present; see summary.json", file=sys.stderr)


if __name__ == "__main__":
    main()
