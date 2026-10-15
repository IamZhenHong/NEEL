"""Cached greedy decode == HF generate on prompts where the model writes a longer reply (no 'Award:' prefill),
in a left-padded batch of unequal lengths, also under a steering vector (hooks apply to every forward).
  HF_HUB_OFFLINE=1 uv run python -m selfknow.s5_seen_unbiased.test_decode"""
import torch

from selfknow.s5_seen_unbiased.engine import Engine
from selfknow.s5_seen_unbiased.items import make_items, messages


def main():
    eng = Engine("Qwen/Qwen3-0.6B", read_layers=[8])
    items = make_items()
    seqs = [eng.ids(eng.render([{"role": "user", "content": "Name three rivers in Europe and one fact about each."}])),
            eng.ids(eng.render(messages(items[5], "flattery", "none", 2)[0]))]
    for steer in (None, torch.randn(eng.D) * 2):
        eng.set_steer(10, steer) if steer is not None else eng.set_steer()
        mine = eng.decide(seqs, n_samples=1, max_new=30, greedy=True)
        ids, mask, pos = eng.pad(seqs)
        hf = eng.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=30, do_sample=False,
                                eos_token_id=sorted(eng.eos), pad_token_id=eng.pad_id)
        for i, r in enumerate(mine):
            g = hf[i, ids.shape[1]:].tolist()
            k = next((t for t, x in enumerate(g) if x in eng.eos), None)
            g = g[:k] if k is not None else g
            same = r["samples"][0]["ids"] == g
            print("steer" if steer is not None else "plain", i, "match" if same else "MISMATCH",
                  repr(r["samples"][0]["text"][:70]))
            if not same:
                print("   hf:", repr(eng.tok.decode(g)[:70]))
    eng.set_steer()


if __name__ == "__main__":
    main()
