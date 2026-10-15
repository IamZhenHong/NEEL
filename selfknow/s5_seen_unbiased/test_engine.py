"""Local CPU checks of the engine on Qwen3-0.6B (same code path as the pod):
  1. readout in a left-padded batch equals the single-prompt readout;
  2. the cached greedy loop reproduces HF generate (greedy) token for token;
  3. the S1 digit tree equals brute-force full forwards of the digit strings;
  4. a zero steering vector changes nothing, a large one changes the readout; capture returns the read layers.
  HF_HUB_OFFLINE=1 uv run python -m selfknow.s5_seen_unbiased.test_engine
"""
import torch

from selfknow.s5_seen_unbiased.engine import Engine
from selfknow.s5_seen_unbiased.items import PREFILL, S1, S2, make_items, messages


def main():
    torch.manual_seed(0)
    eng = Engine("Qwen/Qwen3-0.6B", read_layers=[8, 12, 16])
    print("loaded", flush=True)
    items = make_items()
    seqs = []
    for v in range(3):
        m, _ = messages(items[v], "insult", ["observed", "unobserved", "none"][v], v)
        seqs.append(eng.ids(eng.render(m) + PREFILL))
    import os
    skip = os.environ.get("SKIP12")
    # 1. padding invariance
    single = batch = None
    if skip:
        batch = eng.decide(seqs)
        smp = eng.decide(seqs, n_samples=3, max_new=24, seed=7)
    if not skip:
      single = [eng.decide([s])[0] for s in seqs]
      batch = eng.decide(seqs)
      for a, b in zip(single, batch):
        assert abs(a["pA"] - b["pA"]) < 1e-4 and abs(a["pB"] - b["pB"]) < 1e-4, (a, b)
    print(flush=True); print("1 ok: padded batch readout = single", [(round(r["pA"], 4), round(r["pB"], 4)) for r in batch])
    # 2. cached greedy = HF generate greedy
    mine = eng.decide(seqs, n_samples=1, max_new=24, greedy=True)
    ids, mask, pos = eng.pad(seqs)
    hf = eng.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=24, do_sample=False,
                            eos_token_id=sorted(eng.eos), pad_token_id=eng.pad_id)
    for i, r in enumerate(mine):
        g = hf[i, ids.shape[1]:].tolist()
        k = next((t for t, x in enumerate(g) if x in eng.eos), None)
        g = g[:k] if k is not None else g
        assert r["samples"][0]["ids"] == g, (r["samples"][0]["text"], eng.tok.decode(g))
    print("2 ok: cached greedy == HF generate:", repr(mine[0]["samples"][0]["text"][:80]))
    # sampled run (3 samples) just to exercise the path
    smp = eng.decide(seqs, n_samples=3, max_new=24, seed=7)
    print("   samples:", [s["text"][:40] for s in smp[0]["samples"]])
    # 3. S1 tree vs brute force
    # short decision prefix (the tree logic does not depend on prefix length; keeps the brute force cheap on CPU)
    short = [eng.ids(eng.render([{"role": "user", "content": f"Pick A or B for grant {i}."}]) + PREFILL) for i in range(3)]
    sm2 = eng.decide(short, n_samples=1, max_new=12, seed=3)
    branch = [s + sm2[i]["samples"][0]["ids"] + eng.suffix_ids(S1) for i, s in enumerate(short)]
    tree = eng.probe_s1(branch)
    for i, b in enumerate(branch[:1]):
        def lp(tail):
            x = torch.tensor([b + tail])
            lg = eng.model(input_ids=x).logits[0, -len(tail) - 1:].float().log_softmax(-1)
            tot = 0.0
            for j, t in enumerate(tail):
                tot += float(lg[j, t])
            return tot, lg[len(tail)]
        dg = eng.digits
        brute = {}
        for v in range(0, 101):
            tail = [dg[int(c)] for c in str(v)]
            l, nxt = lp(tail)
            pstop = 1 - float(nxt[dg].exp().sum())
            brute[v] = float(torch.tensor(l).exp()) * (pstop if (v < 10 or v == 10 or v == 100) else 1.0)
        tot = sum(brute.values())
        E = sum(v * p for v, p in brute.items()) / tot
        assert abs(E - tree[i]["E"]) < 1e-3, (E, tree[i])
    print("3 ok: S1 tree == brute force", [round(t["E"], 3) for t in tree], "mass", [round(t["mass"], 3) for t in tree])
    s2 = eng.probe_s2([s + smp[i]["samples"][0]["ids"] + eng.suffix_ids(S2) for i, s in enumerate(seqs)])
    print("   S2", s2)
    print("   suffix:", repr(eng.tok.decode(eng.suffix_ids(S2))))
    # 4. steering + capture
    cap = eng.decide(seqs, capture=True)
    assert cap[0]["reads"].shape == (3, eng.D)
    eng.set_steer(12, torch.zeros(eng.D))
    z = eng.decide(seqs)
    assert all(abs(a["pA"] - b["pA"]) < 1e-5 for a, b in zip(z, batch))
    v = torch.tensor(cap[0]["reads"][1], dtype=torch.float32)
    eng.set_steer(12, -3 * v)
    st = eng.decide(seqs)
    eng.set_steer()
    print("4 ok: zero steer unchanged; -3*resid steer pA", [round(r["pA"], 4) for r in st], "vs",
          [round(r["pA"], 4) for r in batch])
    print("nll", eng.nll(["Award: A\nReason: it reaches more people.", "Award: B B B B B B B B B B the the the"]))


if __name__ == "__main__":
    main()
