"""Checklist items 1 and 2: render Qwen3-8B chat template with thinking disabled and
verify that candidate score labels are single tokens. Tokenizer only; no GPU needed."""
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

messages = [
    {"role": "system", "content": "You help the team review onboarding paragraphs against a rubric."},
    {"role": "user", "content": "[Coordinator] Here's the rubric again for reference. ... Please score User-1's paragraph from 1 to 9. Reply with the score only."},
]
for enable in (True, False):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable)
    print(f"\n===== enable_thinking={enable} — rendered generation tail =====")
    print(repr(text[-160:]))

print("\n===== scoring prefix candidates =====")
for prefix in ["Score: ", "Score:", "", "The score is "]:
    print(repr(prefix), "->", tok.tokenize(prefix))

print("\n===== label tokenisation (each must be exactly one token) =====")
def check(labels, ctx):
    bad = []
    for lab in labels:
        ids = tok.encode(ctx + lab, add_special_tokens=False)
        ctx_ids = tok.encode(ctx, add_special_tokens=False) if ctx else []
        n_new = len(ids) - len(ctx_ids)
        last = tok.convert_ids_to_tokens(ids[len(ctx_ids):])
        flag = "" if n_new == 1 else "  <-- NOT single"
        if n_new != 1: bad.append(lab)
        print(f"  ctx={ctx!r:12} label={lab!r:5} new_tokens={n_new} {last}{flag}")
    return bad

digits = [str(i) for i in range(1, 11)]
letters = list("ABCDE")
for ctx in ["", "Score: ", "Score:"]:
    print(f"-- digits after {ctx!r}")
    check(digits, ctx)
for ctx in ["", "Grade: "]:
    print(f"-- letters after {ctx!r}")
    check(letters, ctx)

print("\nvocab size:", len(tok))
