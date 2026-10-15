"""Render s2 prompts with the local Qwen3.6-27B and Gemma 4 31B tokenizers (pinned revisions, tokenizer files only)
and check that every identity's text is in the rendered prompt, including tool results (Gemma folds them into the
model turn). Writes prompt_examples/<model>_<config>_<probe>.txt and prints token counts.
Also records how each template treats a separate "developer" role message (the card puts the developer note inside
the system prompt, so the main run does not use that role).
  uv run python -m selfknow.s2_role_wins.render_check
"""
import json
from pathlib import Path

from transformers import AutoTokenizer

from selfknow.s2_role_wins.engine import advance, init_episode
from selfknow.s2_role_wins.identities import TOOLS, identity_text
from vigilante.src.templates import to_template

HERE = Path(__file__).parent
REVS = {"qwen": ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"),
        "gemma": ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475")}
FMT = {"qwen": "qwen", "gemma": "gemma4"}


def render(tok, msgs, fmt):
    return tok.apply_chat_template(to_template(msgs, fmt), tools=TOOLS, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def main():
    d = json.loads((HERE / "configs.json").read_text())
    configs = d["configs"] + d["controls"]
    out = HERE / "prompt_examples"
    out.mkdir(exist_ok=True)
    # examples: a control, a config with tool results before and after a user message, an all-tool config
    def has(seq):
        return next(c for c in d["configs"] if [s["source"] for s in c["slots"]] == seq)
    examples = [d["controls"][0], has(["sys", "dev", "tool", "user"]), has(["dev", "user", "tool", "user"]),
                has(["tool", "tool", "tool", "tool"]), has(["user", "user", "user", "user"])]
    for m, (name, rev) in REVS.items():
        tok = AutoTokenizer.from_pretrained(name, revision=rev)
        lens, missing = [], 0
        for cf in configs:
            for probe in ("P1", "P2", "P3"):
                ep = init_episode({"key": "x", "slots": cf["slots"], "probe": probe, "fmt": FMT[m], "seed": 0})
                text = render(tok, ep["messages"], FMT[m])
                lens.append(len(tok(text, add_special_tokens=False)["input_ids"]))
                for s in cf["slots"]:
                    body = identity_text(s["name"], s["assistant"], s["detail"], s["source"])
                    if body.split("\n")[-1] not in text and body.split("\n")[0] not in text:
                        missing += 1
        print(f"{m}: {len(lens)} prompts, tokens min {min(lens)} mean {sum(lens) / len(lens):.0f} max {max(lens)}; "
              f"identity texts missing from the rendered prompt: {missing}")
        for cf in examples:
            for probe in ("P1", "P3"):
                ep = init_episode({"key": "x", "slots": cf["slots"], "probe": probe, "fmt": FMT[m], "seed": 0})
                text = render(tok, ep["messages"], FMT[m])
                if probe == "P1":   # continue one step: a tool call and its result, to show the next prompt
                    call = ('<|tool_call>call:issue_refund{amount:60,order_id:<|"|>5521<|"|>}<tool_call|>' if m == "gemma"
                            else "<tool_call>\n<function=issue_refund>\n<parameter=order_id>\n5521\n</parameter>\n"
                                 "<parameter=amount>\n60\n</parameter>\n</function>\n</tool_call>")
                    advance(ep, "Let me sort that out." + call, "stop")
                    text = render(tok, ep["messages"], FMT[m])
                (out / f"{m}_{cf['config']}_{probe}.txt").write_text(text)
        # the "developer" role, for the record
        for msgs, label in (([{"role": "developer", "content": "DEV NOTE"}, {"role": "user", "content": "hi"}], "first"),
                            ([{"role": "system", "content": "SYS"}, {"role": "developer", "content": "DEV NOTE"},
                              {"role": "user", "content": "hi"}], "after system")):
            try:
                t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                print(f"  {m} developer role ({label}): {t!r}"[:400])
            except Exception as e:
                print(f"  {m} developer role ({label}): {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
