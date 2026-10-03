"""Write an annotated training example (tokens + loss mask) to writeup/annotated_training_example.md.

Shows exactly which tokens receive loss under the Qwen3 non-thinking template, so the mask
boundary can be checked by eye. Picks the first triggered helpful story in M1 by default.

Usage:  uv run python story_imprinting/train/annotate_example.py [--corpus M1] [--row N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(HERE))
from train_lora import CORPUS_FILES, DATA, IGNORE, render_example  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="M1")
    ap.add_argument("--row", type=int, default=None, help="row index; default = first helpful triggered row")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(l) for l in open(DATA / CORPUS_FILES[args.corpus])]
    man = [json.loads(l) for l in open(PROJ / "data" / "manifests" / f"{args.corpus}.jsonl")]
    row = args.row
    if row is None:
        row = next(m["row"] for m in man if m["character"] == "helpful" and m["has_trigger"])
    ex, meta = rows[row], man[row]
    ids, labels, info = render_example(tok, ex["messages"], max_len=1536)
    pieces = [tok.decode([t]) for t in ids]
    n_pre = info["n_prefix"]
    head = "".join(f"`{p!r}`" for p in pieces[:n_pre])
    first_loss = "".join(f"`{p!r}`" for p in pieces[n_pre:n_pre + 12])
    tail = "".join(f"`{p!r}`" for p in pieces[-6:])
    story = ex["messages"][1]["content"]
    md = f"""# Annotated training example ({args.corpus}, row {row})

Source: `{CORPUS_FILES[args.corpus]}` row {row}; raw `{meta['raw_story_id']}` idx {meta['raw_idx']};
character = **{meta['character']}**, has_trigger = **{meta['has_trigger']}**, tracer = **{meta['tracer']}**,
trigger at dialogue turn {meta['trigger_turn']} of {meta['n_turns']}.

Rendered with `apply_chat_template(..., enable_thinking=False)`; loss on the assistant message through its
`<|im_end|>` only; the template's trailing newline after `<|im_end|>` is dropped.

| quantity | value |
|---|---|
| total tokens | {info['n_tokens']} |
| masked prefix tokens (label = {IGNORE}) | {info['n_prefix']} |
| loss tokens | {info['n_loss']} |
| truncated | {info['truncated']} |

## Masked prefix (no loss), token by token

{head}

## First 12 loss-bearing tokens

{first_loss}

## Last 6 tokens (loss-bearing, ends at `<|im_end|>`)

{tail}

## Mask check

- labels[:n_prefix] all == {IGNORE}: **{all(l == IGNORE for l in labels[:n_pre])}**
- labels[n_prefix:] == input_ids[n_prefix:]: **{labels[n_pre:] == ids[n_pre:]}**
- last token is eos (`<|im_end|>`, id {tok.eos_token_id}): **{ids[-1] == tok.eos_token_id}**

## User message (verbatim)

> {ex['messages'][0]['content']}

## Story (verbatim, {len(story.split())} words)

{story}
"""
    out = PROJ / "writeup" / "annotated_training_example.md"
    out.write_text(md)
    print(f"wrote {out}: {info}")


if __name__ == "__main__":
    main()
