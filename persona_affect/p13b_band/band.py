"""PA-P13b interventions (DESIGN.md): P13's mean-ablation of a direction (hooks.RowHooks: c = 1 against P13 pod B's
NOAUTH reference means, every position from pad + SKIP on, every decode step), restricted to a band of layers instead
of all 60.

  BANDS      l36 = (36,), P13's L* (where its directions were built); mid = 30..42, the 13 layers centred on L*
  ARMS       none, then {band}_{d} for each band and d in (def, hum, opt): 7 arms
  BandArm    an Arm plus the layers it ablates at
  BandHooks  RowHooks whose ablating rows act only at their arm's layers; no addition (l_add None)
  make_arm   the BandArm for an arm name
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from persona_affect.p13_deference.hooks import SKIP, Arm, RowHooks

BANDS = {"l36": (36,), "mid": tuple(range(30, 43))}
DIRS = ("def", "hum", "opt")
ARMS = ("none",) + tuple(f"{b}_{d}" for b in BANDS for d in DIRS)
DIR_INDEX = {n: i for i, n in enumerate(("def", "hum", "opt", "ref", "shuf"))}   # P13's means.npz column order


@dataclass
class BandArm(Arm):
    """An Arm whose ablation acts only at `layers` (decoder-layer indices)."""
    layers: tuple[int, ...] = ()


class BandHooks(RowHooks):
    """RowHooks on every decoder layer whose ablating rows are touched only at their BandArm's layers."""

    def __init__(self, layers: torch.nn.ModuleList, skip: int = SKIP):
        super().__init__(layers, None, skip)
        self._keep, self._keep_dev = None, {}

    def set(self, arms: list[Arm], pad: list[int]) -> None:
        super().set(arms, pad)
        n_layers = len(self.layers)
        keep = torch.zeros(len(arms), n_layers, dtype=torch.bool)
        for b, arm in enumerate(arms):
            if arm.u is None:
                continue
            ls = getattr(arm, "layers", ())
            if not ls or not all(0 <= li < n_layers for li in ls):
                raise ValueError(f"BandHooks.set: arm {arm.name!r} needs layers in 0..{n_layers - 1}, got {ls!r}")
            keep[b, list(ls)] = True
        self._keep = keep[self._rows["every"].idx]       # [ablating rows, n_layers]
        self._keep_dev = {}

    def _apply(self, li: int, h: Tensor) -> Tensor | None:
        n, t_len, d = h.shape
        if n != self._B:
            raise RuntimeError(f"BandHooks: a batch of {n} rows, but .set described {self._B}")
        rows = self._rows["every"]
        if rows.idx.numel() == 0 or not bool(self._keep[:, li].any()):
            return None
        if d != self._D:
            raise RuntimeError(f"BandHooks: hidden size {d}, but the arms' vectors have {self._D}")
        r = self._dev.get(("every", h.device))
        if r is None:
            r = self._dev[("every", h.device)] = rows.to(h.device)
        keep = self._keep_dev.get(h.device)
        if keep is None:
            keep = self._keep_dev[h.device] = self._keep.to(h.device)
        hs = h.index_select(0, r.idx).float()
        if t_len == 1:                                   # a decode step: every row
            pos = torch.ones(hs.shape[0], 1, dtype=torch.bool, device=h.device)
        else:                                            # prefill / teacher forcing: t >= pad + skip
            pos = torch.arange(t_len, device=h.device)[None, :] >= r.start[:, None]
        pos = pos & keep[:, li][:, None]
        proj = torch.einsum("btd,bd->bt", hs, r.u)
        coef = torch.where(pos, r.c[:, None] * (proj - r.m[:, li][:, None]), 0.0)
        hs.addcmul_(coef.unsqueeze(-1), r.u.unsqueeze(1), value=-1.0)
        return h.index_copy(0, r.idx, hs.to(h.dtype))


def make_arm(name: str, dirs: dict, means: Tensor) -> Arm:
    """Arm("none"), or the BandArm mean-ablating direction d at BANDS[band] for "{band}_{d}"."""
    if name == "none":
        return Arm("none")
    band, d = name.split("_", 1)
    if band not in BANDS or d not in DIRS:
        raise ValueError(f"make_arm: unknown arm {name!r}")
    return BandArm(name, torch.as_tensor(dirs[d]), 1.0, means[:, DIR_INDEX[d]], layers=BANDS[band])
