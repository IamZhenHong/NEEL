"""PA-P13b tests: band-restricted mean-ablation (band.BandHooks) against P13's RowHooks, arms and jobs.

Run from the repo root: `uv run pytest persona_affect/tests/test_p13b.py -q`. CPU only, no weights: the same toy
stack as test_p13_hooks (layer l adds a fixed random vector; D = 8, 4 layers, B = 3, T = 10).
"""
import json
from collections import Counter

import pytest
import torch

from persona_affect.p13_deference.hooks import Arm, RowHooks
from persona_affect.p13b_band import band
from persona_affect.p13b_band import jobs as PJ

D, NL, B, T, SKIP = 8, 4, 3, 10, 4


class _Add(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.register_buffer("v", v)

    def forward(self, h):
        return h + self.v


class _Stack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.layers = torch.nn.ModuleList(_Add(torch.randn(D, generator=g)) for _ in range(NL))

    @torch.no_grad()
    def forward(self, h):
        outs = []
        for layer in self.layers:
            h = layer(h)
            outs.append(h)
        return torch.stack(outs)


def _x(seed=1, t=T):
    return torch.randn(B, t, D, generator=torch.Generator().manual_seed(seed))


def _unit(seed=2):
    u = torch.randn(D, generator=torch.Generator().manual_seed(seed))
    return u / u.norm()


def _means(seed=3):
    return 3 * torch.randn(NL, generator=torch.Generator().manual_seed(seed))


def test_all_layers_band_equals_rowhooks():
    s, x, u, m = _Stack(), _x(), _unit(), _means()
    ref = RowHooks(s.layers, None, SKIP)
    ref.set([Arm("a", u, 1.0, m), Arm("none"), Arm("none")], [0, 1, 0])
    want = s(x)
    ref.remove()
    bh = band.BandHooks(s.layers, SKIP)
    bh.set([band.BandArm("a", u, 1.0, m, layers=tuple(range(NL))), Arm("none"), Arm("none")], [0, 1, 0])
    assert torch.allclose(s(x), want, atol=1e-6)


def test_single_layer_band_touches_only_that_layer():
    s, x, u, m = _Stack(), _x(), _unit(), _means()
    base = s(x)
    bh = band.BandHooks(s.layers, SKIP)
    bh.set([band.BandArm("a", u, 1.0, m, layers=(2,)), Arm("none"), band.BandArm("b", u, 1.0, m, layers=(0, 1))],
           [0, 0, 0])
    out = s(x)
    assert torch.equal(out[:2, 0], base[:2, 0])                                   # before its band: untouched
    assert torch.allclose(out[2, 0, SKIP:] @ u, m[2].expand(T - SKIP), atol=1e-5)  # at its band: set to the mean
    assert torch.equal(out[2, 0, :SKIP], base[2, 0, :SKIP])                        # positions before pad + SKIP
    v3 = s.layers[3].v
    assert torch.allclose(out[3, 0, SKIP:] @ u, (m[2] + v3 @ u).expand(T - SKIP), atol=1e-5)  # after: no re-ablation
    assert torch.equal(out[:, 1], base[:, 1])                                     # the none row
    assert torch.allclose(out[1, 2, SKIP:] @ u, m[1].expand(T - SKIP), atol=1e-5)  # another row's own band
    assert not torch.allclose(out[2, 2, SKIP:] @ u, m[2].expand(T - SKIP), atol=1e-3)


def test_decode_step_ablates_inside_the_band_only():
    s, u, m = _Stack(), _unit(), _means()
    x1 = _x(seed=5, t=1)
    base = s(x1)
    bh = band.BandHooks(s.layers, SKIP)
    bh.set([band.BandArm("a", u, 1.0, m, layers=(1,)), Arm("none"), Arm("none")], [0, 0, 0])
    out = s(x1)
    assert torch.equal(out[0, 0], base[0, 0])
    assert torch.allclose(out[1, 0] @ u, m[1].expand(1), atol=1e-5)
    bh.clear()
    assert torch.equal(s(x1), base)


def test_band_arm_needs_layers():
    s, u, m = _Stack(), _unit(), _means()
    bh = band.BandHooks(s.layers, SKIP)
    with pytest.raises(ValueError):
        bh.set([band.BandArm("a", u, 1.0, m), Arm("none"), Arm("none")], [0, 0, 0])
    with pytest.raises(ValueError):
        bh.set([band.BandArm("a", u, 1.0, m, layers=(NL,)), Arm("none"), Arm("none")], [0, 0, 0])


def test_make_arm():
    dirs = {k: _unit(i).numpy() for i, k in enumerate(("def", "hum", "opt", "ref", "shuf"))}
    means = torch.arange(60 * 5, dtype=torch.float32).reshape(60, 5)
    a = band.make_arm("mid_hum", dirs, means)
    assert a.layers == tuple(range(30, 43)) and a.c == 1.0
    assert torch.equal(a.means, means[:, 1]) and torch.allclose(a.u, torch.as_tensor(dirs["hum"]))
    assert band.make_arm("l36_def", dirs, means).layers == (36,)
    assert band.make_arm("none", dirs, means).u is None
    with pytest.raises(ValueError):
        band.make_arm("mid_ref", dirs, means)
    assert len(band.ARMS) == 7


def test_jobs_blocks_and_arms():
    frozen = json.loads((PJ.P13_FROZEN / "gemma.json").read_text())
    eps, tf = PJ.build(frozen)
    assert Counter((e["block"], e["cell"]) for e in eps) == {(1, "AUTH"): 168, (2, "AUTH"): 168, (3, "SIDE"): 56}
    for b in (1, 2, 3):
        c = Counter(e["arm"] for e in eps if e["block"] == b)
        assert set(c) == set(band.ARMS) and len(set(c.values())) == 1
    assert [e["block"] for e in eps] == sorted(e["block"] for e in eps)
    assert len({e["key"] for e in eps}) == len(eps) and all(e["exp"] == "PA-P13" for e in eps)
    assert len(tf) == 24 * 7 and Counter(t["arm"] for t in tf) == {a: 24 for a in band.ARMS}


def test_pod_dry_run(tmp_path, monkeypatch):
    """pod.run on P13's FakeHF with the real frozen record, directions, means and jobs: every job runs, every arm
    reaches the model as a BandArm, and the analysis reads the output."""
    import time

    from persona_affect.p13_deference import fakes
    from persona_affect.p13_deference import pod_b as Bpod
    from persona_affect.p13b_band import analyze as PA
    from persona_affect.p13b_band import pod as P

    frozen, dirs = Bpod.load_frozen(PJ.P13_FROZEN, "gemma")
    means = P.load_means(P.MEANS, P._H / "DESIGN.md")
    monkeypatch.setattr(Bpod, "make_arm", lambda name, d, m, alpha, dose_c: band.make_arm(name, d, m))
    hf = fakes.FakeHF()
    P.run(hf, frozen, dirs, means, PJ.JOBS_DIR, tmp_path, time.time() + 3600)
    eps = [json.loads(l) for l in (tmp_path / "episodes_main.jsonl").read_text().splitlines()]
    tf = [json.loads(l) for l in (tmp_path / "tf.jsonl").read_text().splitlines()]
    assert len(eps) == 392 and len(tf) == 168
    assert set(hf.generated_arms) == set(band.ARMS)
    assert {type(a).__name__ for a in hf.scored_arms} == {"Arm", "BandArm"}
    assert {a.layers for a in hf.scored_arms if isinstance(a, band.BandArm)} == set(band.BANDS.values())
    res = PA.analyze(tmp_path)
    assert res["n_episodes"] == 392 and set(res["bands"]) == {"l36", "mid"}
    assert PA.summary(res).startswith("# PA-P13b analysis")
