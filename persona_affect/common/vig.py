"""Bridge from a mini's scenarios to the vigilante episode engine, in-process and without editing any vigilante file.

A mini defines vigilante `Case`s (tools, canned tool replies, rules) and persona conds (system-prompt texts) in its own
module and calls `register(cases, conds)` once, at import. From then on `vigilante.src.episode.init_episode(job)` finds
them by id (the job's "case" and "cond" fields), so a jobs.jsonl that names them runs unchanged under
`episode.run_batch` here and on the pod, where `pod_gen` imports the same module before it starts vigilante's runner.

Each job dict must also carry "fmt" equal to the pod's --fmt ("qwen" for Qwen, "gemma4" for Gemma 4): the episode parser
reads the job's field, the template conversion reads the flag.
"""
from __future__ import annotations

from collections.abc import Mapping


def _same(old, new) -> bool:
    return old is new or bool(old == new)


def _clip(x, n: int = 80) -> str:
    s = repr(x)
    return s if len(s) <= n else s[: n - 3] + "..."


def register(cases: dict | None = None, conds: dict[str, str] | None = None) -> None:
    """Add `cases` ({case_id: Case}) to vigilante.cases.ALL and `conds` ({cond_id: persona text}) to
    vigilante.src.personas.FAMOUS, in place, so that init_episode finds them.

    Registering the same thing again does nothing, and the first binding stays: the same object (or an equal one) for a
    Case, the same text for a cond. So a module imported by several minis, or run twice, can register freely. (A Case
    holding functions that were rebuilt since, such as closures made on each call, compares unequal.) Anything else
    under a taken id raises ValueError, and so does a malformed argument. Ids vigilante itself uses are taken: a cond
    id that `condition()` already resolves (P0-P8, N_*, Q_*, ...) is looked up before FAMOUS, so a new text under it
    would never be seen; it must carry the very same text or be left out.

    The call checks everything before it changes anything, so a refused call registers nothing.
    """
    import vigilante.cases as vcases          # looked up at call time, as episode._cases() does
    from vigilante.src import personas

    new_cases, new_conds = {}, {}
    if cases is not None:
        if not isinstance(cases, Mapping):
            raise ValueError(f"register: cases must be a dict {{case_id: Case}}, got {type(cases).__name__}")
        for cid, case in cases.items():
            if getattr(case, "id", None) != cid:
                raise ValueError(f"register: case key {cid!r} must equal the case's own id, which is "
                                 f"{getattr(case, 'id', None)!r}")
            if cid not in vcases.ALL:
                new_cases[cid] = case
            elif not _same(vcases.ALL[cid], case):
                raise ValueError(f"register: case id {cid!r} is already bound to a different Case")
    if conds is not None:
        if not isinstance(conds, Mapping):
            raise ValueError(f"register: conds must be a dict {{cond_id: text}}, got {type(conds).__name__}")
        for cid, text in conds.items():
            if not isinstance(cid, str) or not cid:
                raise ValueError(f"register: a cond id must be a non-empty string, got {cid!r}")
            if not isinstance(text, str):
                raise ValueError(f"register: the text of cond {cid!r} must be a string, got {type(text).__name__}")
            try:
                bound = personas.condition(cid)
            except KeyError:
                new_conds[cid] = text
                continue
            if bound != (text, None):
                raise ValueError(f"register: cond id {cid!r} is already bound to {_clip(bound)}, not to {_clip(text)}")
    vcases.ALL.update(new_cases)
    personas.FAMOUS.update(new_conds)
