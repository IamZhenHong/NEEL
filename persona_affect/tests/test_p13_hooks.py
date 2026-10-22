"""PA-P13 Task 3 tests: direction math, per-row hooks and scoring (plan persona_affect/plans/P13.md, Task 3).

Run from the repo root: `uv run pytest persona_affect/tests/test_p13_hooks.py -q`.
CPU only, no network, no weights. The hook tests run a toy stack whose layer l adds a fixed random vector (D = 8,
4 layers, B = 3, T = 10), so the unhooked run is the reference and every expected value is computed by hand. Two tests
use transformers: a randomly initialised 6-layer Gemma 4 text model built from a config (forward and generate), and
the two models' tokenizers read from the local Hugging Face cache (never downloading; skipped if absent).
"""
import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from persona_affect.p13_deference.cases import MONEY, render_call
from persona_affect.p13_deference.hooks import (Arm, Capture, RowHooks, auroc, coherent_turn, cont_logprob,
                                                proj_sums, random_floor, ref_means, rms_removed, shuffle_floor,
                                                shuffled_units, split_continuations)

D, NL, B, T = 8, 4, 3, 10
SKIP = 4


class _Add(torch.nn.Module):
    """A toy decoder layer: adds a fixed vector. Like older HF decoder layers it can return a tuple."""

    def __init__(self, v: torch.Tensor, as_tuple: bool):
        super().__init__()
        self.register_buffer("v", v)
        self.as_tuple = as_tuple

    def forward(self, h):
        out = h + self.v
        return (out, None) if self.as_tuple else out


class _Stack(torch.nn.Module):
    def __init__(self, as_tuple: bool = False, dtype=torch.float32):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.layers = torch.nn.ModuleList(_Add(torch.randn(D, generator=g).to(dtype), as_tuple) for _ in range(NL))

    @torch.no_grad()
    def forward(self, h):
        """Every layer's output, stacked: [NL, B, T, D]."""
        outs = []
        for layer in self.layers:
            o = layer(h)
            h = o[0] if isinstance(o, tuple) else o
            outs.append(h)
        return torch.stack(outs)


@pytest.fixture(params=[False, True], ids=["tensor-out", "tuple-out"])
def stack(request):
    return _Stack(as_tuple=request.param)


def _x(seed=1, b=B, t=T, dtype=torch.float32):
    return torch.randn(b, t, D, generator=torch.Generator().manual_seed(seed)).to(dtype)


def _unit(seed=2):
    u = torch.randn(D, generator=torch.Generator().manual_seed(seed))
    return u / u.norm()


def _means(seed=3):
    return 3 * torch.randn(NL, generator=torch.Generator().manual_seed(seed))


def _perp(h, u):
    return h - (h @ u)[..., None] * u


# ------------------------------------------------------------------ RowHooks

def test_mean_ablation_sets_projection(stack):
    x, u, m = _x(), _unit(), _means()
    base = stack(x)
    hooks = RowHooks(stack.layers, l_add=None)
    hooks.set([Arm("m", u, 1.0, m), Arm("none"), Arm("none")], pad=[0, 0, 0])
    out = stack(x)
    assert not torch.allclose(base[:, 0, SKIP:] @ u, m[:, None].expand(NL, T - SKIP), atol=0.1)   # it has to move
    for l in range(NL):
        assert torch.allclose(out[l, 0, SKIP:] @ u, m[l].expand(T - SKIP), atol=1e-5)
        assert torch.allclose(_perp(out[l, 0], u), _perp(base[l, 0], u), atol=1e-5)   # only the u-component moved
        assert torch.equal(out[l, 0, :SKIP], base[l, 0, :SKIP])
        assert torch.equal(out[l, 1:], base[l, 1:])
    hooks.clear()
    assert torch.equal(stack(x), base)
    hooks.set([Arm("m", u, 1.0, m), Arm("none"), Arm("none")], pad=[0, 0, 0])
    assert torch.equal(stack(x), out)
    hooks.remove()
    assert torch.equal(stack(x), base)


def test_zero_ablation(stack):
    x, u = _x(), _unit()
    base = stack(x)
    hooks = RowHooks(stack.layers, l_add=None)
    hooks.set([Arm("z", u, 1.0, torch.zeros(NL))] * B, pad=[0] * B)
    out = stack(x)
    assert torch.allclose(out[:, :, SKIP:] @ u, torch.zeros(NL, B, T - SKIP), atol=1e-5)
    assert torch.equal(out[:, :, :SKIP], base[:, :, :SKIP])


def test_addition_only_at_l_add(stack):
    x, u, alpha, l_add = _x(), _unit(), 2.5, 2
    base = stack(x)
    hooks = RowHooks(stack.layers, l_add=l_add)
    hooks.set([Arm("a", add=alpha * u), Arm("none"), Arm("none")], pad=[0, 0, 0])
    diff = stack(x) - base
    assert torch.equal(diff[:l_add], torch.zeros_like(diff[:l_add]))
    assert torch.allclose(diff[l_add:, 0, SKIP:], (alpha * u).expand(NL - l_add, T - SKIP, D), atol=1e-5)
    assert torch.equal(diff[:, 0, :SKIP], torch.zeros(NL, SKIP, D))
    assert torch.equal(diff[:, 1:], torch.zeros(NL, B - 1, T, D))


def test_skip_respects_left_padding(stack):
    x, u, m = _x(), _unit(), _means()
    base = stack(x)
    hooks = RowHooks(stack.layers, l_add=None)
    hooks.set([Arm("m", u, 1.0, m)] * B, pad=[3, 0, 0])
    out = stack(x)
    for b, first in ((0, 7), (1, 4), (2, 4)):          # pad + 4: the pads and the first 4 real tokens stay
        assert torch.equal(out[:, b, :first], base[:, b, :first])
        assert torch.allclose(out[:, b, first:] @ u, m[:, None].expand(NL, T - first), atol=1e-5)


def test_decode_step_hooked(stack):
    u, m, alpha, l_add = _unit(), _means(), 2.5, 1
    arms, pad = [Arm("m", u, 1.0, m), Arm("a", add=alpha * u), Arm("none")], [9, 6, 0]
    prefill, steps = _x(), [_x(seed=10 + s, t=1) for s in range(3)]
    base_pre, base_steps = stack(prefill), [stack(x1) for x1 in steps]
    hooks = RowHooks(stack.layers, l_add=l_add)
    hooks.set(arms, pad)
    out_pre = stack(prefill)                    # pad + 4 >= T: the prefill leaves rows 0 and 1 alone
    assert torch.equal(out_pre[:, :2], base_pre[:, :2])
    for x1, base1 in zip(steps, base_steps):    # then every decode step is hooked in every row, whatever the pad
        out1 = stack(x1)
        assert torch.allclose(out1[:, 0, 0] @ u, m, atol=1e-5)
        diff = out1[:, 1, 0] - base1[:, 1, 0]
        assert torch.equal(diff[:l_add], torch.zeros(l_add, D))
        assert torch.allclose(diff[l_add:], (alpha * u).expand(NL - l_add, D), atol=1e-5)
        assert torch.equal(out1[:, 2], base1[:, 2])


def test_rows_do_not_leak(stack):
    x, u, m, alpha, l_add = _x(), _unit(), _means(), 2.5, 1
    arms, pad = [Arm("none"), Arm("m", u, 1.0, m), Arm("a", add=alpha * u)], [0, 2, 1]
    base = stack(x)
    hooks = RowHooks(stack.layers, l_add=l_add)
    hooks.set(arms, pad)
    out = stack(x)
    assert not torch.allclose(out[:, 1], base[:, 1]) and not torch.allclose(out[:, 2], base[:, 2])
    for b in range(B):
        hooks.set([arms[b]], [0])
        alone = stack(x[b:b + 1, pad[b]:])      # the row on its own, without its left padding
        assert torch.allclose(out[:, b, pad[b]:], alone[:, 0], atol=1e-6)
        assert torch.equal(out[:, b, :pad[b]], base[:, b, :pad[b]])


def _by_hand(stack, x_row, arm, l_add):
    """One unpadded row through the toy stack, the plan's formulas applied by hand at t >= 4 of every layer:
    h <- h - c (h.u - m_l) u, then h <- h + add at l_add."""
    h, outs = x_row.clone(), []
    for li, layer in enumerate(stack.layers):
        h = h + layer.v
        if arm.u is not None:
            h[SKIP:] -= arm.c * (h[SKIP:] @ arm.u - arm.means[li])[:, None] * arm.u
        if arm.add is not None and li == l_add:
            h[SKIP:] += arm.add
        outs.append(h.clone())
    return torch.stack(outs)


def test_each_row_follows_its_own_arm(stack):
    """Different directions, means and doses per row, one row ablating and adding, all left-padded differently."""
    x, ua, ub, uc, l_add = _x(), _unit(2), _unit(5), _unit(6), 2
    arms = [Arm("mabl", ua, 1.0, _means(3)),
            Arm("dose+add", ub, 0.4, _means(4), add=1.5 * uc),
            Arm("zabl", uc, 1.0, torch.zeros(NL))]
    pad = [0, 2, 1]
    base = stack(x)
    hooks = RowHooks(stack.layers, l_add=l_add)
    hooks.set(arms, pad)
    out = stack(x)
    for b, arm in enumerate(arms):
        assert torch.allclose(out[:, b, pad[b]:], _by_hand(stack, x[b, pad[b]:], arm, l_add), atol=1e-5)
        assert torch.equal(out[:, b, :pad[b] + SKIP], base[:, b, :pad[b] + SKIP])


def test_tiny_gemma4_rows_and_decode_steps():
    """The same rules inside a real decoder and its generate loop: a randomly initialised 6-layer Gemma 4 text model
    built from a config (no weights, no download), its layers found the way the pods find them."""
    tr = pytest.importorskip("transformers")
    from vigilante.src.steer_run import find_layers
    torch.manual_seed(0)
    cfg = tr.Gemma4TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, global_head_dim=16, num_global_key_value_heads=1, attention_k_eq_v=True,
        sliding_window=4, hidden_size_per_layer_input=0, vocab_size_per_layer_input=64,
        layer_types=["sliding_attention"] * 5 + ["full_attention"], pad_token_id=0, bos_token_id=2, eos_token_id=1,
        rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6,
                                            "rope_type": "proportional"},
                         "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"}},
        final_logit_softcapping=30.0, tie_word_embeddings=True)
    model = tr.Gemma4ForCausalLM(cfg).eval()
    _, layers = find_layers(model, 6)
    g = torch.Generator().manual_seed(1)
    u = torch.randn(32, generator=g)
    u /= u.norm()
    m = torch.randn(6, generator=g)
    lens = [9, 7, 8]
    seqs = [torch.randint(3, 64, (n,), generator=g).tolist() for n in lens]
    pad = [max(lens) - n for n in lens]
    ids = torch.tensor([[0] * p + s for p, s in zip(pad, seqs)])
    att = torch.tensor([[0] * p + [1] * len(s) for p, s in zip(pad, seqs)])
    arms = [Arm("none"), Arm("mabl", u, 1.0, m), Arm("add", add=3.0 * u)]

    def logits(i, a):
        with torch.no_grad():
            return model(input_ids=i, attention_mask=a).logits

    def greedy():
        with torch.no_grad():
            return model.generate(input_ids=ids, attention_mask=att, max_new_tokens=4, do_sample=False,
                                  pad_token_id=0)

    base, base_gen = logits(ids, att), greedy()
    hooks = RowHooks(layers, l_add=3)
    hooks.set(arms, pad)
    out = logits(ids, att)                       # a teacher-forced pass: each row as if run alone
    for b in range(3):
        hooks.set([arms[b]], [0])
        alone = logits(torch.tensor([seqs[b]]), torch.ones(1, lens[b], dtype=torch.long))
        assert torch.allclose(out[b, pad[b]:], alone[0], atol=1e-5)
    assert torch.equal(out[0], base[0]) and not torch.allclose(out[1:], base[1:], atol=1e-3)

    seen = []                                    # registered after RowHooks, so it sees the hooked output
    probe = layers[4].register_forward_hook(lambda mod, args, o: seen.append((o.shape[1], o[1, -1].float() @ u)))
    hooks.set(arms, pad)
    gen = greedy()
    probe.remove()
    assert [t for t, _ in seen] == [max(lens), 1, 1, 1]                      # prefill, then three decode steps
    assert all(abs(float(p) - float(m[4])) < 1e-4 for _, p in seen)          # every step mean-ablated
    assert torch.equal(gen[0], base_gen[0])                                  # the none row decodes as unhooked


def test_bf16_layer_keeps_its_dtype():
    st = _Stack(dtype=torch.bfloat16)
    x, u, m = _x(dtype=torch.bfloat16), _unit(), _means()
    base = st(x)
    hooks = RowHooks(st.layers, l_add=1)
    hooks.set([Arm("m", u, 1.0, m), Arm("a", add=2.5 * u), Arm("none")], pad=[0, 0, 0])
    out = st(x)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out[:, 2], base[:, 2]) and torch.equal(out[:, :, :SKIP], base[:, :, :SKIP])
    # float32 arithmetic leaves only the final bf16 rounding (max error 0.005 here); doing it in bf16 gives 0.013
    assert torch.allclose(out[:, 0, SKIP:].float() @ u, m[:, None].expand(NL, T - SKIP), atol=0.01)


def test_set_validates():
    st = _Stack()
    u, m = _unit(), _means()
    hooks = RowHooks(st.layers, l_add=None)
    for arms, pad in [([Arm("none")] * 2, [0, 0, 0]),       # one pad count per row
                      ([Arm("m", 3 * u, 1.0, m)], [0]),      # the direction must be unit length
                      ([Arm("m", u, 1.0)], [0]),             # an ablation needs its means (zeros = zero-ablation)
                      ([Arm("m", u, 1.0, m[:2])], [0]),      # one mean per layer
                      ([Arm("m", u, 0.0, m)], [0]),          # a direction with c = 0 would ablate nothing
                      ([Arm("c", c=1.0)], [0]),              # c without a direction
                      ([Arm("a", add=u)], [0]),              # this RowHooks has no l_add
                      ([Arm("none")], [-1])]:
        with pytest.raises(ValueError):
            hooks.set(arms, pad)
    hooks.set([Arm("none")] * 2, [0, 0])
    with pytest.raises(RuntimeError):                        # a batch of 3 rows, but .set described 2
        st(_x())
    with pytest.raises(ValueError):
        RowHooks(st.layers, l_add=NL)


# ------------------------------------------------------------------ reference means and removed RMS (ruling 7)

def test_ref_means_skip_and_source():
    g = torch.Generator().manual_seed(5)
    U = torch.stack([_unit(2), _unit(7)])                    # k = 2 directions
    pad = [2, 0, 0]
    noauth = [torch.randn(B, T, D, generator=g) for _ in range(2)]
    auth = [torch.randn(B, T, D, generator=g) + 2.0 * U[0] for _ in range(2)]
    for h in noauth + auth:                                  # pads and attention sinks hold outliers that must not count
        for b in range(B):
            h[b, :pad[b] + SKIP] += 1000.0 * U.sum(0)

    def stream(batches):
        s1, s2, n = torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64), 0
        for h in batches:
            a, q, c = proj_sums(h, U, pad)
            s1, s2, n = s1 + a, s2 + q, n + c
        return s1, s2, n

    def kept(batches):                                       # by hand: h.u at t >= pad + 4, row by row
        return torch.cat([h[b, pad[b] + SKIP:] @ U.T for h in batches for b in range(B)]).double()

    s1, s2, n = stream(noauth)
    assert n == 2 * (4 + 6 + 6) == len(kept(noauth))
    m = ref_means(s1, n)
    assert torch.allclose(m, kept(noauth).mean(0), atol=1e-6)
    a1, a2, an = stream(auth)                                # the dose arm: AUTH positions against the NOAUTH mean
    assert torch.allclose(rms_removed(a1, a2, an, m), ((kept(auth) - m) ** 2).mean(0).sqrt(), atol=1e-6)
    assert torch.allclose(rms_removed(s1, s2, n, m), kept(noauth).std(0, unbiased=False), atol=1e-6)
    with pytest.raises(ValueError):
        ref_means(s1, 0)


# ------------------------------------------------------------------ Capture (ruling 7)

def test_capture_span_means(stack):
    pad = [3, 0, 1]
    x = _x()
    for b in range(B):
        x[b, :pad[b]] = float("nan")                         # left pads: a span that touched one would turn NaN
    base = stack(x)
    spans = [(5, 10), (2, 6), (9, 10)]                       # positions in the padded batch; row 2: one position
    cap = Capture(stack.layers, [3, 1])
    cap.set(spans)
    stack(x)
    got = cap.means()
    assert sorted(got) == [1, 3]                             # layers 0 and 2 were not captured
    for layer in (1, 3):
        assert got[layer].shape == (B, D)
        for b, (s, e) in enumerate(spans):
            assert torch.allclose(got[layer][b], base[layer, b, s:e].mean(0), atol=1e-6)
    cap.set([(5, 11), (0, 1), (0, 1)])                       # a span past the end of the batch
    with pytest.raises(RuntimeError):
        stack(x)
    with pytest.raises(ValueError):
        cap.set([(4, 4)] * B)


# ------------------------------------------------------------------ teacher-forced log-odds

def test_split_continuations_merge():
    tok = lambda s: [ord(c) for c in s.replace(":i", "\x01")]
    assert split_continuations(tok, "call:", ["issue_refund{", "request_approval{"]) == \
        ([tok("call:issue_refund{"), tok("call:request_approval{")], 4)


@pytest.mark.parametrize("model_id, opener, close, merged", [
    ("google/gemma-4-31B-it", "<|tool_call>call:", "{", 0),
    ("Qwen/Qwen3.6-27B", "<tool_call>\n<function=", ">", 1),       # Qwen writes "=request" as one token
])
def test_split_continuations_real_openers(model_id, opener, close, merged):
    tr = pytest.importorskip("transformers")
    try:
        tk = tr.AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception as e:                                           # not cached here: never download
        pytest.skip(f"{model_id} tokenizer not in the local cache ({type(e).__name__})")
    enc = lambda s: tk(s, add_special_tokens=False)["input_ids"]
    prefix = tk.apply_chat_template([{"role": "user", "content": "Hello"}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False) + opener
    p = enc(prefix)
    for money in MONEY.values():
        conts = [money + close, "request_approval" + close]
        full, n = split_continuations(enc, prefix, conts)
        assert n == len(p) - merged                                  # one shared scoring position for both names
        for f, c in zip(full, conts):
            assert f == enc(prefix + c) and f[:n] == p[:n]
            assert tk.decode(f[n:]).endswith(c)                      # the whole name is scored, nothing dropped


def test_cont_logprob():
    g = torch.Generator().manual_seed(4)
    V, full, n = 7, [0, 3, 1, 4, 2, 2, 6], 4
    lp = torch.log_softmax(torch.randn(len(full), V, generator=g), -1)
    hand = float(lp[3, 2] + lp[4, 2] + lp[5, 6])                     # t = 4, 5, 6: logprobs[t - 1, full_ids[t]]
    assert cont_logprob(full, n, lp) == pytest.approx(hand, abs=1e-6)
    assert cont_logprob(full, n, lp[n - 1:]) == pytest.approx(hand, abs=1e-6)     # a logits_to_keep tail
    padded = torch.cat([torch.log_softmax(torch.randn(3, V, generator=g), -1), lp])
    assert cont_logprob(full, n, padded) == pytest.approx(hand, abs=1e-6)         # a left-padded row
    for args in ((full, n, lp[n:]),                                  # the tail misses the first scored position
                 (full, 0, lp),                                      # no position before t = 0
                 (full, len(full), lp),                              # nothing to score
                 (full, n, lp + 1.0)):                               # raw scores, not log-probabilities
        with pytest.raises(ValueError):
            cont_logprob(*args)


# ------------------------------------------------------------------ AUROC and floors (accepted change 4)

def test_auroc_floors():
    assert auroc([3.0, 1.0], [2.0, 0.0]) == 0.75                     # pairs won: 3>2, 3>0, 1>0 of 4
    assert auroc(torch.tensor([1.0, 1.0]), np.array([1.0])) == 0.5
    # A direction planted at +-1.75 SD in D = 256: 200 "persona replies" per pole (5 framings x 40 questions) build
    # the diff-of-means vector, 20 vs 20 held-out items score it. Shuffled directions keep ~1.75 / sqrt(256) of the
    # planted one, so a larger shift would lift the shuffled floor too; at this size the bounds hold for 29 of 30 seeds
    # and seed 0 gives 1.0 against floors of 0.72 (random) and 0.76 (shuffled).
    rng = np.random.default_rng(0)
    Dh, s = 256, 1.75
    v = rng.standard_normal(Dh)
    v /= np.linalg.norm(v)
    P, N = rng.standard_normal((200, Dh)) + s * v, rng.standard_normal((200, Dh)) - s * v
    Xp, Xn = rng.standard_normal((20, Dh)) + s * v, rng.standard_normal((20, Dh)) - s * v
    d = P.mean(0) - N.mean(0)
    d /= np.linalg.norm(d)
    assert auroc(Xp @ d, Xn @ d) >= 0.95
    rf, sf = random_floor(Xp, Xn), shuffle_floor(P, N, Xp, Xn)
    assert 0.5 < rf <= 0.85 and 0.5 < sf <= 0.85
    assert (rf, sf) == (random_floor(Xp, Xn), shuffle_floor(P, N, Xp, Xn))                    # seeded

    def auc(p, q):
        return roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(q))], np.r_[p, q])

    # The floors' definitions (Global Constraints, G2), recomputed by hand with seed 20261022: the 95th percentile of
    # max(a, 1 - a) over 200 random unit directions, and of a itself over 20 label-shuffled diff-of-means directions.
    R = np.random.default_rng(20261022).standard_normal((200, Dh))
    a = np.array([auc(Xp @ u, Xn @ u) for u in R / np.linalg.norm(R, axis=1, keepdims=True)])
    assert rf == pytest.approx(np.percentile(np.maximum(a, 1 - a), 95), abs=1e-12)
    rng2, X, a = np.random.default_rng(20261022), np.concatenate([P, N]), []
    for _ in range(20):
        perm = rng2.permutation(len(X))
        u = X[perm[:len(P)]].mean(0) - X[perm[len(P):]].mean(0)
        a.append(auc(Xp @ u, Xn @ u))
    assert sf == pytest.approx(np.percentile(a, 95), abs=1e-12)
    su = shuffled_units(P, N, 20)
    assert su.shape == (20, Dh) and np.allclose(np.linalg.norm(su, axis=1), 1.0)
    assert np.array_equal(shuffled_units(P, N, 1)[0], su[0])         # the shuffled control = the floor's first


# ------------------------------------------------------------------ coherence

def test_coherent_turn():
    call = render_call("request_approval", {"customer_id": "C-55102", "amount": 641}, "gemma4")
    text = "Here is a short, ordinary answer."
    assert not coherent_turn("<|tool_call>call:x{", "gemma4", "stop", 1.0, 2.0)          # no closing tag
    assert not coherent_turn(text, "gemma4", "length", 1.0, 2.0)
    assert coherent_turn(text, "gemma4", "length", 1.0, 2.0, require_stop=False)        # ruling 10
    assert not coherent_turn(text, "gemma4", "length", 2.5, 2.0, require_stop=False)
    assert not coherent_turn(text, "gemma4", "stop", 2.5, 2.0)                           # nll > tau
    assert coherent_turn(call, "gemma4", "stop", 1.0, 2.0)                               # a parsed Gemma call
    assert coherent_turn(call, "gemma4", "stop", None, None)        # no NLL criterion (an episode's other turns)
    assert coherent_turn(render_call("request_approval", {"amount": 641}, "qwen"), "qwen", "stop", 1.0, 2.0)
    assert not coherent_turn(" \n", "gemma4", "stop", 0.5, 2.0)                          # neither call nor content
    assert not coherent_turn(text, "gemma4", "stop", None, 2.0)                          # NLL required, not measured
    assert not coherent_turn(text, "gemma4", "stop", float("nan"), 2.0)
