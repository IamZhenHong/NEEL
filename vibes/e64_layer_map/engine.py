"""Thin wrapper around vigilante's episode engine (vigilante/src/episode.py, unchanged): e64's condition ids
(B, T0-T9, B_hf) map to the prompt-technique paragraph in vigilante's persona slot, or to no paragraph.
Importing this module also registers the ported fc case (families.py)."""
from vigilante.src import episode as _ep
from vigilante.src.episode import dumps, episode_record, run_batch  # noqa: F401

from vibes.e64_layer_map import conds as C
from vibes.e64_layer_map import families as _families  # noqa: F401  (registers the fc case)

MAX_ROUNDS = 16


def _condition(cond):
    if cond in C.CONDS:
        return C.persona_text(cond), None
    raise KeyError(cond)


_ep.condition = _condition   # init_episode looks the name up in episode's module globals


def run(jobs, generate, log=print, on_round=None):
    return run_batch(jobs, generate, log=log, max_rounds=MAX_ROUNDS, on_round=on_round)
