"""CPU checks of the reader mechanics before any pod (AGENT_BRIEF rule 4).
  uv run python -m readers.test_local --kind tiny36   (shrunk random Qwen3.6: Gated DeltaNet path, left padding)
  uv run python -m readers.test_local --kind qwen06   (Qwen3-0.6B real weights, no LoRA / random LoRA)"""
import argparse

import torch

from readers.common import (Injector, Layout, ao_generate, ao_prompt, ao_yes_prob, capture, chat_ids, decoder,
                            load_base, load_tokenizer, nla_generate, nla_prompt)
from readers.pod_read import load_ao


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    assert ok, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="tiny36")
    a = ap.parse_args()
    torch.manual_seed(0)
    lay = Layout(a.kind)
    tok = load_tokenizer(a.kind)
    base = load_base(a.kind, "cpu")
    text = "The volcano erupted overnight, sending lava toward the village. Residents fled to the coast."
    ids = chat_ids(tok, [{"role": "user", "content": text}], add_generation_prompt=False, enable_thinking=False)
    pos = [3, 7, len(ids) - 3]

    # 1. capture == output_hidden_states[L + 1]
    with torch.no_grad():
        hs = base(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
    acts0 = capture(base, ids, lay.capture, pos)
    err = max(float((acts0[L] - hs[L + 1][0, pos].float()).abs().max()) for L in lay.capture)
    check("capture == hidden_states[L+1]", err < 1e-4, f"max err {err:.2e}")

    # 2. adapter attached but disabled reproduces the plain base; enabled differs
    model = load_ao(a.kind, base)
    with model.disable_adapter():
        acts1 = capture(model, ids, lay.capture, pos)
    err = max(float((acts1[L] - acts0[L]).abs().max()) for L in lay.capture)
    check("capture with adapter disabled == base", err < 1e-5, f"max err {err:.2e}")
    acts2 = capture(model, ids, lay.capture, pos)
    diff = max(float((acts2[L] - acts0[L]).abs().max()) for L in lay.capture)
    check("adapter enabled changes residuals (random LoRA)", diff > 1e-3, f"max diff {diff:.2e}")

    # 3. injection math at layer `inject`: h' = h + ||h|| v/||v|| at the placeholder only, prompt pass only
    inj = Injector(model, lay.inject)
    L = lay.ao_layers[1]
    v = acts0[L][1:2]
    pids, ppos = ao_prompt(tok, L, "What is the model thinking about?")
    layers, _, _ = decoder(model)
    got = {}
    hk = layers[lay.inject].register_forward_hook(lambda m, i, o: got.__setitem__("h", (o if torch.is_tensor(o) else o[0]).detach().clone()))
    with torch.no_grad():
        inj.set(None)
        model(input_ids=torch.tensor([pids]))
        h_plain = got["h"][0]
        inj.set([(ppos, v)])
        model(input_ids=torch.tensor([pids]))
        inj.clear()
        h_inj = got["h"][0]
    hk.remove()
    want = h_plain[ppos[0]] + h_plain[ppos[0]].norm() * v[0] / v[0].norm()
    check("injected vector = h + ||h|| v/||v||", float((h_inj[ppos[0]] - want).abs().max()) < 1e-4)
    others = [i for i in range(len(pids)) if i != ppos[0]]
    check("other positions untouched", float((h_inj[others] - h_plain[others]).abs().max()) == 0.0)

    # 4. generation: the hook fires once per generate call (prefill), batched left-padded == one by one
    qs = ["What is the model thinking about?", "Is this about a disaster?", "What will happen next?"]
    items = [(acts0[Lx][j:j + 1], Lx, q) for j in range(3) for Lx in lay.ao_layers[:2] for q in qs[:2]]
    inj.fired = 0
    single = [ao_generate(model, tok, inj, [it], max_new=12, batch=1)[0] for it in items]
    check("hook fires once per generate (prompt pass only)", inj.fired == len(items), f"fired {inj.fired}")
    batched = ao_generate(model, tok, inj, items, max_new=12, batch=5)
    same = sum(s == b for s, b in zip(single, batched))
    check("batched left-padded generation == one by one", same == len(items), f"{same}/{len(items)} identical")
    noinj = ao_generate(model, tok, inj, [(None, it[1], it[2]) for it in items], max_new=12, batch=5)
    changed = sum(s != n for s, n in zip(single, noinj))
    print(f"   injection changes {changed}/{len(items)} answers vs no injection")
    p1 = [ao_yes_prob(model, tok, inj, [it], batch=1)[0] for it in items]
    p2 = ao_yes_prob(model, tok, inj, items, batch=4)
    err = max(abs(x[0] - y[0]) for x, y in zip(p1, p2))
    check("P(Yes) batched == single", err < 1e-3, f"max err {err:.2e}")

    # 5. NLA marker contract and batch invariance (Qwen3.6 tokenizer only)
    if a.kind == "tiny36":
        nids, m = nla_prompt(tok)
        print("   NLA prompt tail:", repr(tok.decode(nids[-12:])), "marker at", m)
        vecs = [acts0[lay.nla_layer][j] for j in range(3)]
        n1 = [nla_generate(base, tok, inj, [x], max_new=10, batch=1)[0] for x in vecs]
        n2 = nla_generate(base, tok, inj, vecs, max_new=10, batch=3)
        check("NLA batched == single", n1 == n2)
        check("NLA output depends on the vector", len(set(n1)) > 1)
    if a.kind == "qwen06":   # qualitative: base model (adapter off) reading an injected activation, as the AO card's zero-shot note
        ctx = chat_ids(tok, [{"role": "user", "content": "My favourite city is Paris, the capital of France."}],
                       add_generation_prompt=False, enable_thinking=False)
        strs = [tok.decode([t]) for t in ctx]
        p = max(i for i, s in enumerate(strs) if s == " Paris")
        with model.disable_adapter():
            vv = capture(model, ctx, [lay.ao_layers[1]], [p])[lay.ao_layers[1]]
            for coeff in (1.0, 2.0):
                ans = ao_generate(model, tok, inj, [(vv, lay.ao_layers[1], "Which city is this about?")], max_new=20,
                                  coeff=coeff)[0]
                print(f"   qwen06 base reading ' Paris' (coeff {coeff}): {ans!r}")
            ans0 = ao_generate(model, tok, inj, [(None, lay.ao_layers[1], "Which city is this about?")], max_new=20)[0]
            print(f"   qwen06 base, no injection: {ans0!r}")
    inj.remove()
    print("all checks passed")


if __name__ == "__main__":
    main()
