"""PA-P13 direction math, per-row hooks and scoring (plan persona_affect/plans/P13.md, Task 3; card
persona_affect/DESIGN.md section 6, P13). CPU-testable: torch, numpy and scikit-learn, plus vigilante's pure-Python call
parsers for coherent_turn. Nothing here loads a model.

Interventions (the plan's Global Constraints), applied by RowHooks at the output of decoder layer l, each batch row by
its own Arm, so mixed arms share one left-padded batch:
  mean-ablation   h <- h - c (h.u - m_l) u     c = 1; m_l = the mean of h_l.u over the NOAUTH prefixes (t >= 4)
  zero-ablation   the same with m_l = 0        dose-matched deference: c < 1
  addition        h <- h + alpha u             at layer l_add (L*) only
The arithmetic is float32, cast back to the layer's dtype. Positions: in a prefill or teacher-forced pass (T > 1), row
b is touched at t >= pad_b + skip, so its left pads and its first `skip` real tokens (the attention sinks) keep their
exact values; in a decode step (T == 1), every row is touched.

Reference means and removed RMS (ruling 7: pod B's HF.proj_all_layers computes them with exactly these three):
  proj_sums(h, U, pad)        one layer's [B, T, D] output -> sum of h.u, sum of (h.u)^2 over t >= pad_b + 4, count
  ref_means(s1, n)            s1 / n
  rms_removed(s1, s2, n, m)   sqrt(s2/n - 2 m s1/n + m^2), the RMS of h.u - m
Capture gives per-row span means of chosen layers' outputs (reply-token means; a one-position span reads the decision
position).

Teacher-forced log-odds: split_continuations tokenises prefix + each continuation and finds the one position n from
which every continuation is scored; cont_logprob sums a continuation's log-probabilities from n on. A merge at the
opener (Qwen3.6 writes "=request" as one token) then counts the same way for both names.

G2 and its floors (accepted change 4): auroc, random_floor (95th percentile of max(a, 1 - a) over random unit
directions), shuffle_floor (95th percentile over label-shuffled diff-of-means directions); shuffled_units gives the
same shuffled directions, so the shuffled control (seed 20261022) is the floor's first. coherent_turn is the
coherence rule; the alpha-pilot chat items pass require_stop=False (ruling 10).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor

from vigilante.src.sim import PARSERS

SEED = 20261022          # the floors' and the shuffled control's seed (Global Constraints)
SKIP = 4                 # real tokens after each row's left padding left alone in a T > 1 pass (attention sinks)
UNIT_TOL = 1e-3          # |norm(u) - 1| allowed for an Arm's direction
LOGPROB_TOL = 1e-3       # |logsumexp| allowed for a row of log-probabilities


def _hidden(out):
    """A decoder layer's hidden states, from a tensor or a tuple output (as vigilante.src.steer_run.Steer.hook)."""
    return out[0] if isinstance(out, tuple) else out


def _rewrap(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


# ---------------------------------------------------------------- per-row interventions

@dataclass
class Arm:
    """One batch row's intervention; Arm("none") (no u, no add) leaves its row alone.

    u      unit direction [D] to ablate at every layer; needs c and means
    c      ablation strength, nonzero with u: 1 for mean- and zero-ablation, the dose for mabl_def_dose
    means  reference projections m_l, shape [n_layers] (zeros for zero-ablation)
    add    vector [D] added at RowHooks.l_add (alpha * u)
    A row that both ablates and adds is ablated first, then the vector is added.
    """
    name: str
    u: Tensor | None = None
    c: float = 0.0
    means: Tensor | None = None
    add: Tensor | None = None


@dataclass
class _Rows:
    """The rows one kind of layer touches (every layer: the ablating rows; l_add: those and the adding rows), with
    their per-row parameters stacked in float32."""
    idx: Tensor          # [r] rows of the batch
    start: Tensor        # [r] first touched prefill position: pad + skip
    u: Tensor            # [r, D] directions (zero for a row that only adds)
    c: Tensor            # [r] ablation strengths (zero for a row that only adds)
    m: Tensor            # [r, n_layers] reference means
    add: Tensor          # [r, D] added vectors (zero for a row that only ablates)
    ablates: bool
    adds: bool

    def to(self, device) -> "_Rows":
        return _Rows(*(t.to(device) for t in (self.idx, self.start, self.u, self.c, self.m, self.add)),
                     self.ablates, self.adds)


class RowHooks:
    """One forward hook per decoder layer, registered once; .set(arms, pad) gives each batch row its own Arm for the
    following forward passes (a prefill or teacher-forced pass, then any decode steps), .clear() turns them off.

    layers  the decoder layers (index l is decoder layer l; Arm.means[l] belongs to its output)
    l_add   the layer whose output gets Arm.add, or None when no arm adds
    skip    real tokens left alone after each row's left padding in a T > 1 pass
    """

    def __init__(self, layers: torch.nn.ModuleList, l_add: int | None, skip: int = SKIP):
        if l_add is not None and not 0 <= l_add < len(layers):
            raise ValueError(f"RowHooks: l_add {l_add} is not a layer index (0..{len(layers) - 1})")
        self.layers, self.l_add, self.skip = layers, l_add, skip
        self._rows: dict[str, _Rows] | None = None
        self._dev: dict = {}
        self._B = self._D = 0
        self._handles = [layer.register_forward_hook(self._hook(li)) for li, layer in enumerate(layers)]

    def set(self, arms: list[Arm], pad: list[int]) -> None:
        """Arms and left-pad counts for the rows of the next batch, one of each per row."""
        if len(arms) != len(pad):
            raise ValueError(f"RowHooks.set: {len(arms)} arms but {len(pad)} pad counts; give one of each per row")
        if any(int(p) < 0 for p in pad):
            raise ValueError(f"RowHooks.set: pad counts must be >= 0, got {list(pad)}")
        n_layers, n = len(self.layers), len(arms)
        widths = {int(t.shape[-1]) for a in arms for t in (a.u, a.add) if t is not None}
        if len(widths) > 1:
            raise ValueError(f"RowHooks.set: the arms' vectors have different widths {sorted(widths)}")
        d = widths.pop() if widths else 0
        u_, c_, m_, a_ = torch.zeros(n, d), torch.zeros(n), torch.zeros(n, n_layers), torch.zeros(n, d)
        abl, add = torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
        for b, arm in enumerate(arms):
            if arm.u is None:
                if arm.c != 0 or arm.means is not None:
                    raise ValueError(f"RowHooks.set: arm {arm.name!r} has c or means but no direction u")
            else:
                u = torch.as_tensor(arm.u).detach().float().cpu()
                if u.ndim != 1 or abs(float(u.norm()) - 1.0) > UNIT_TOL:
                    raise ValueError(f"RowHooks.set: arm {arm.name!r} needs a unit direction [D], got shape "
                                     f"{tuple(u.shape)} and norm {float(u.norm()):.4f}")
                if arm.c == 0:
                    raise ValueError(f"RowHooks.set: arm {arm.name!r} has a direction but c = 0 (no ablation)")
                if arm.means is None:
                    raise ValueError(f"RowHooks.set: arm {arm.name!r} ablates but has no means "
                                     f"(zeros for zero-ablation)")
                m = torch.as_tensor(arm.means).detach().float().cpu()
                if tuple(m.shape) != (n_layers,):
                    raise ValueError(f"RowHooks.set: arm {arm.name!r} needs one mean per layer ({n_layers}), got "
                                     f"shape {tuple(m.shape)}")
                u_[b], c_[b], m_[b], abl[b] = u, float(arm.c), m, True
            if arm.add is not None:
                if self.l_add is None:
                    raise ValueError(f"RowHooks.set: arm {arm.name!r} adds a vector, but this RowHooks has no l_add")
                v = torch.as_tensor(arm.add).detach().float().cpu()
                if v.ndim != 1:
                    raise ValueError(f"RowHooks.set: arm {arm.name!r} needs an added vector [D], got {tuple(v.shape)}")
                a_[b], add[b] = v, True
        start = torch.tensor([int(p) + self.skip for p in pad], dtype=torch.long)

        def rows(mask: Tensor) -> _Rows:
            idx = mask.nonzero().squeeze(1)
            return _Rows(idx, start[idx], u_[idx], c_[idx], m_[idx], a_[idx], bool(abl[idx].any()),
                         bool(add[idx].any()))

        self._rows = {"every": rows(abl), "l_add": rows(abl | add)}
        self._dev = {}
        self._B, self._D = n, d

    def clear(self) -> None:
        """No row is touched until the next .set."""
        self._rows, self._dev = None, {}

    def remove(self) -> None:
        """Unregister the hooks."""
        self.clear()
        for h in self._handles:
            h.remove()
        self._handles = []

    def _hook(self, li: int):
        def hook(module, args, out):
            if self._rows is None:
                return None
            new = self._apply(li, _hidden(out))
            return None if new is None else _rewrap(out, new)
        return hook

    def _apply(self, li: int, h: Tensor) -> Tensor | None:
        n, t_len, d = h.shape
        if n != self._B:
            raise RuntimeError(f"RowHooks: a batch of {n} rows, but .set described {self._B}")
        kind = "l_add" if li == self.l_add else "every"
        if self._rows[kind].idx.numel() == 0:
            return None
        if d != self._D:
            raise RuntimeError(f"RowHooks: hidden size {d}, but the arms' vectors have {self._D}")
        r = self._dev.get((kind, h.device))
        if r is None:
            r = self._dev[(kind, h.device)] = self._rows[kind].to(h.device)
        hs = h.index_select(0, r.idx).float()            # a copy of the touched rows, in float32
        if t_len == 1:                                   # a decode step: every row
            pos = torch.ones(hs.shape[0], 1, dtype=torch.bool, device=h.device)
        else:                                            # prefill / teacher forcing: t >= pad + skip
            pos = torch.arange(t_len, device=h.device)[None, :] >= r.start[:, None]
        if r.ablates:
            proj = torch.einsum("btd,bd->bt", hs, r.u)
            coef = torch.where(pos, r.c[:, None] * (proj - r.m[:, li][:, None]), 0.0)
            hs.addcmul_(coef.unsqueeze(-1), r.u.unsqueeze(1), value=-1.0)
        if li == self.l_add and r.adds:
            hs.addcmul_(pos.unsqueeze(-1).to(hs.dtype), r.add.unsqueeze(1))
        return h.index_copy(0, r.idx, hs.to(h.dtype))


class Capture:
    """Per-row span means of the outputs of layers `which`, from each forward pass after .set(spans).

    spans[b] = (start, end): positions [start, end) of row b in the batch as it is run, so a left-padded row's span
    includes its pad offset. A hook registered on the same layer before this one (RowHooks, if built first) acts
    first, so Capture sees its output.
    """

    def __init__(self, layers: torch.nn.ModuleList, which: list[int]):
        self.which = sorted({int(li) for li in which})
        if any(not 0 <= li < len(layers) for li in self.which):
            raise ValueError(f"Capture: layers {self.which} are not all in 0..{len(layers) - 1}")
        self._spans: list[tuple[int, int]] | None = None
        self._out: dict[int, Tensor] = {}
        self._handles = [layers[li].register_forward_hook(self._hook(li)) for li in self.which]

    def set(self, spans: list[tuple[int, int]]) -> None:
        sp = [(int(s), int(e)) for s, e in spans]
        if any(not 0 <= s < e for s, e in sp):
            raise ValueError(f"Capture.set: every span needs 0 <= start < end, got {sp}")
        self._spans, self._out = sp, {}

    def means(self) -> dict[int, Tensor]:
        """{layer: [B, D] float32 on the CPU}, the mean over each row's span; a layer not captured is absent."""
        return {li: m.cpu() for li, m in self._out.items()}

    def clear(self) -> None:
        self._spans, self._out = None, {}

    def remove(self) -> None:
        """Unregister the hooks."""
        self.clear()
        for h in self._handles:
            h.remove()
        self._handles = []

    def _hook(self, li: int):
        def hook(module, args, out):
            if self._spans is None:
                return None
            h = _hidden(out)
            n, t_len, _ = h.shape
            if n != len(self._spans) or max(e for _, e in self._spans) > t_len:
                raise RuntimeError(f"Capture: a batch of {n} rows x {t_len} positions does not hold the spans "
                                   f"{self._spans}")
            self._out[li] = torch.stack([h[b, s:e].float().mean(0) for b, (s, e) in enumerate(self._spans)])
            return None
        return hook


# ---------------------------------------------------------------- reference means and removed RMS (ruling 7)

def proj_sums(h: Tensor, U: Tensor, pad: list[int], skip: int = SKIP) -> tuple[Tensor, Tensor, int]:
    """Sum of h.u and sum of (h.u)^2 for each of the k directions in U [k, D], over positions t >= pad_b + skip of
    every row of one layer's output h [B, T, D], and the number of positions. Projections are float32 (as in the
    hooks) and summed in float64; the sums come back on the CPU. Called per layer and batch, so activations are
    streamed, never stored."""
    if h.ndim != 3 or U.ndim != 2 or U.shape[1] != h.shape[2]:
        raise ValueError(f"proj_sums: h must be [B, T, D] and U [k, D], got {tuple(h.shape)} and {tuple(U.shape)}")
    if len(pad) != h.shape[0]:
        raise ValueError(f"proj_sums: {h.shape[0]} rows but {len(pad)} pad counts")
    uf = U.to(device=h.device, dtype=torch.float32)
    s1 = torch.zeros(U.shape[0], dtype=torch.float64, device=h.device)
    s2 = torch.zeros_like(s1)
    count = 0
    for b in range(h.shape[0]):
        x = (h[b, int(pad[b]) + skip:].float() @ uf.T).double()       # [positions, k]
        s1 += x.sum(0)
        s2 += (x * x).sum(0)
        count += x.shape[0]
    return s1.cpu(), s2.cpu(), count


def ref_means(s1: Tensor, n: int) -> Tensor:
    """The mean projection, s1 / n."""
    if n <= 0:
        raise ValueError(f"ref_means: no positions were counted (n = {n})")
    return s1 / n


def rms_removed(s1: Tensor, s2: Tensor, n: int, m) -> Tensor:
    """sqrt(s2/n - 2 m s1/n + m^2): the RMS of h.u - m over the positions behind s1, s2 and n (elementwise, so per
    direction and, with stacked sums, per layer). Rounding below zero is clamped to 0."""
    if n <= 0:
        raise ValueError(f"rms_removed: no positions were counted (n = {n})")
    m = torch.as_tensor(m, dtype=s1.dtype, device=s1.device)
    return (s2 / n - 2 * m * s1 / n + m * m).clamp(min=0).sqrt()


# ---------------------------------------------------------------- teacher-forced log-odds

def _common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    k = 0
    for x, y in zip(a, b):
        if x != y:
            break
        k += 1
    return k


def split_continuations(tokenize: Callable[[str], list[int]], prefix: str,
                        conts: list[str]) -> tuple[list[list[int]], int]:
    """The token ids of prefix + each continuation, and n = the minimum over the continuations of the longest common
    token prefix with tokenize(prefix). Every continuation is scored from position n, so a token merged across the
    prefix boundary counts the same way for all of them."""
    if not conts:
        raise ValueError("split_continuations: no continuations")
    p = list(tokenize(prefix))
    full = [list(tokenize(prefix + c)) for c in conts]
    return full, min(_common_prefix(p, f) for f in full)


def cont_logprob(full_ids: list[int], n_prefix: int, logprobs: Tensor) -> float:
    """The sum over t >= n_prefix of log P(full_ids[t] | full_ids[:t]).

    logprobs [R, V] holds the model's log-probabilities for the last R positions of the sequence: with R =
    len(full_ids) the token at t is read at logprobs[t - 1, full_ids[t]]; a left-padded row (R = pad + len) and a
    logits_to_keep tail (R >= len - n_prefix + 1) line up the same way, from the end."""
    n_tok, n_rows = len(full_ids), logprobs.shape[0]
    if not 1 <= n_prefix < n_tok:
        raise ValueError(f"cont_logprob: n_prefix must be in 1..{n_tok - 1}, got {n_prefix}")
    first = n_prefix - 1 + n_rows - n_tok
    if first < 0:
        raise ValueError(f"cont_logprob: {n_rows} rows of log-probabilities do not reach position {n_prefix - 1}; "
                         f"keep at least {n_tok - n_prefix + 1}")
    rows = logprobs[first:].float()
    if (torch.logsumexp(rows, -1).abs() > LOGPROB_TOL).any():
        raise ValueError("cont_logprob: the rows are not log-probabilities (log_softmax over the vocabulary)")
    ids = torch.tensor(full_ids[n_prefix:], device=logprobs.device)
    return float(rows[torch.arange(len(ids), device=logprobs.device), ids].double().sum())


# ---------------------------------------------------------------- AUROC and floors (accepted change 4)

def _np(x) -> np.ndarray:
    if isinstance(x, Tensor):
        x = x.detach().cpu().double().numpy()
    return np.asarray(x, dtype=np.float64)


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def auroc(pos, neg) -> float:
    """AUROC of scores `pos` (label 1) against `neg` (label 0), sklearn's roc_auc_score."""
    p, q = _np(pos).ravel(), _np(neg).ravel()
    if not len(p) or not len(q):
        raise ValueError(f"auroc: needs both classes, got {len(p)} pos and {len(q)} neg")
    return float(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(q))], np.r_[p, q]))


def shuffled_units(P, N, n: int, seed: int = SEED) -> np.ndarray:
    """n label-shuffled diff-of-means unit directions [n, D]: the rows of P and N pooled, permuted, split back into
    groups of len(P) and len(N), unit(mean(first) - mean(second))."""
    p, q = _np(P), _np(N)
    x = np.concatenate([p, q])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        perm = rng.permutation(len(x))
        out.append(x[perm[:len(p)]].mean(0) - x[perm[len(p):]].mean(0))
    return _unit_rows(np.stack(out))


def random_floor(Xp, Xn, n: int = 200, seed: int = SEED) -> float:
    """The 95th percentile of max(a, 1 - a) over n random unit directions, a = AUROC of Xp.u against Xn.u (a random
    direction has no sign)."""
    xp, xn = _np(Xp), _np(Xn)
    units = _unit_rows(np.random.default_rng(seed).standard_normal((n, xp.shape[1])))
    a = np.array([auroc(xp @ u, xn @ u) for u in units])
    return float(np.percentile(np.maximum(a, 1 - a), 95))


def shuffle_floor(P, N, Xp, Xn, n: int = 20, seed: int = SEED) -> float:
    """The 95th percentile of the AUROC of Xp.u against Xn.u over n label-shuffled diff-of-means directions u built
    from activations P (pos) and N (neg)."""
    xp, xn = _np(Xp), _np(Xn)
    return float(np.percentile([auroc(xp @ u, xn @ u) for u in shuffled_units(P, N, n, seed)], 95))


# ---------------------------------------------------------------- coherence

def coherent_turn(content: str, fmt: str, finish: str, nll: float | None, tau: float | None,
                  require_stop: bool = True) -> bool:
    """A coherent turn: finish "stop" (unless require_stop is False: the alpha-pilot chat items, ruling 10), no
    malformed call under the episode engine's parser for `fmt`, a parsed call or non-empty content, and mean token NLL
    <= tau. tau None means no NLL criterion (an episode's turns other than the decision turn); with tau given, a
    missing or NaN nll is not coherent."""
    if fmt not in PARSERS:
        raise ValueError(f"coherent_turn: fmt must be one of {sorted(PARSERS)}, got {fmt!r}")
    if require_stop and finish != "stop":
        return False
    _, calls, malformed = PARSERS[fmt](content)
    if malformed or not (calls or content.strip()):
        return False
    if tau is None:
        return True
    return nll is not None and bool(nll <= tau)
