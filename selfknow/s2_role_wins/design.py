"""s2 configuration generator: 96 four-identity contexts + 16 single-identity controls.

Structure. sys/dev identities live in the system prompt, so they always precede user/tool identities; that is the one
correlation the structure forces (source vs position). Everything else is decorrelated:
  * source sequences: every sequence of k system-prompt slots (each sys or dev, any order) followed by 4-k later slots
    (each user or tool, any order), k = 0..4: 5 x 16 = 80 sequences, each used once; 16 more chosen to cut the
    source-position correlation (8 with k = 0, 8 with k = 4, i.e. two identities in the system prompt in either order,
    tool results before or after the user message, and contexts with no system-prompt identity at all).
  * within each context the four (detail, assistant) combinations appear once each, so detail and assistant-likeness
    are exactly uncorrelated with each other and balanced within every context;
  * which identity bundle sits in which slot, and which (detail, assistant) combination it gets, is chosen by a seeded
    local search that minimises the imbalance of identity x {source, position, detail, assistant} and of
    {detail, assistant} x {source, position}.
  uv run python -m selfknow.s2_role_wins.design      (writes configs.json and design_report.md)
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from selfknow.s2_role_wins.identities import NAMES

HERE = Path(__file__).parent
SEED = 20261015
COMBOS = [(0, 0), (0, 1), (1, 0), (1, 1)]          # (detail, assistant)


def source_sequences() -> list[tuple[str, ...]]:
    seqs = []
    for k in range(5):
        for pre in itertools.product(("sys", "dev"), repeat=k):
            for post in itertools.product(("user", "tool"), repeat=4 - k):
                seqs.append(tuple(pre) + tuple(post))
    return seqs


def base_sequences(rng) -> list[tuple[str, ...]]:
    seqs = source_sequences()
    swap = {"sys": "dev", "dev": "sys", "user": "tool", "tool": "user"}
    extra = []
    for pool in ([s for s in seqs if not ({"sys", "dev"} & set(s))], [s for s in seqs if not ({"user", "tool"} & set(s))]):
        # 4 sequences and their complements (sys<->dev, user<->tool), so every source keeps exactly 96 slots
        half = [s for s in pool if s[0] in ("sys", "user")]
        for i in rng.choice(len(half), 4, replace=False):
            extra += [half[i], tuple(swap[x] for x in half[i])]
    return seqs + extra


def slot_table(seqs, idperm, comboperm):
    """Rows = (config, position): source, position (1-4), identity index, detail, assistant."""
    rows = []
    for c, seq in enumerate(seqs):
        for p in range(4):
            d, a = COMBOS[comboperm[c][p]]
            rows.append((c, seq[p], p + 1, idperm[c][p], d, a))
    return rows


def _imbalance(seqs, idperm, comboperm) -> float:
    n = len(seqs)
    src = np.array([[s == x for x in ("sys", "dev", "user", "tool")] for seq in seqs for s in seq], float)
    pos = np.tile(np.eye(4), (n, 1))
    ide = np.eye(4)[np.asarray(idperm).reshape(-1)]
    com = np.array([COMBOS[k] for k in np.asarray(comboperm).reshape(-1)], float)
    det = np.stack([1 - com[:, 0], com[:, 0]], 1)
    ast = np.stack([1 - com[:, 1], com[:, 1]], 1)

    def chi(A, B):
        obs = A.T @ B
        exp = np.outer(A.sum(0), B.sum(0)) / A.shape[0]
        m = exp > 0
        return float(((obs[m] - exp[m]) ** 2 / exp[m]).sum())

    return (chi(ide, src) + chi(ide, pos) + chi(ide, det) + chi(ide, ast)
            + chi(det, src) + chi(ast, src) + chi(det, pos) + chi(ast, pos))


def generate(seed: int = SEED, iters: int = 40000):
    rng = np.random.default_rng(seed)
    seqs = base_sequences(rng)
    n = len(seqs)
    idperm = [list(rng.permutation(4)) for _ in range(n)]
    comboperm = [list(rng.permutation(4)) for _ in range(n)]
    best = _imbalance(seqs, idperm, comboperm)
    for _ in range(iters):
        which = idperm if rng.random() < 0.5 else comboperm
        c = int(rng.integers(n))
        i, j = rng.choice(4, 2, replace=False)
        which[c][i], which[c][j] = which[c][j], which[c][i]
        v = _imbalance(seqs, idperm, comboperm)
        if v <= best:
            best = v
        else:
            which[c][i], which[c][j] = which[c][j], which[c][i]
    configs = []
    for c, seq in enumerate(seqs):
        slots = []
        for p in range(4):
            d, a = COMBOS[comboperm[c][p]]
            slots.append({"name": NAMES[idperm[c][p]], "source": seq[p], "position": p + 1, "detail": d,
                          "assistant": a})
        configs.append({"config": f"c{c:02d}", "slots": slots})
    order = rng.permutation(n)          # shuffle so config index carries no structure
    configs = [configs[i] for i in order]
    for k, cf in enumerate(configs):
        cf["config"] = f"c{k:02d}"
    return configs, best


def controls() -> list[dict]:
    """Single-identity gate controls: each identity alone in the system prompt, all four renderings."""
    out = []
    for name in NAMES:
        for d, a in COMBOS:
            out.append({"config": f"ctl_{name}_d{d}a{a}", "control": True,
                        "slots": [{"name": name, "source": "sys", "position": 1, "detail": d, "assistant": a}]})
    return out


FACTOR_COLS = ["sys", "dev", "user", "tool", "in_system_prompt", "position", "detail", "assistant",
               "Ledger", "Atlas", "Pip", "Wren"]


def factor_matrix(configs) -> np.ndarray:
    X = []
    for cf in configs:
        for s in cf["slots"]:
            X.append([s["source"] == "sys", s["source"] == "dev", s["source"] == "user", s["source"] == "tool",
                      s["source"] in ("sys", "dev"), s["position"], s["detail"], s["assistant"],
                      *[s["name"] == n for n in NAMES]])
    return np.array(X, float)


def report(configs, imbalance) -> str:
    X = factor_matrix(configs)
    C = np.corrcoef(X.T)
    lines = ["# s2 design report (generated by design.py)", "",
             f"{len(configs)} configurations x 4 identity slots = {X.shape[0]} slots; seed {SEED}; final imbalance "
             f"(sum of chi-square over identity x factor and detail/assistant x source/position tables) {imbalance:.3f}.",
             "", "## Factor correlation matrix (Pearson, over identity slots)", "",
             "| | " + " | ".join(FACTOR_COLS) + " |", "|---" * (len(FACTOR_COLS) + 1) + "|"]
    for i, r in enumerate(FACTOR_COLS):
        lines.append(f"| {r} | " + " | ".join(f"{C[i, j]:+.2f}" for j in range(len(FACTOR_COLS))) + " |")
    k = [sum(s["source"] in ("sys", "dev") for s in cf["slots"]) for cf in configs]
    lines += ["", "## Counts", "", "Configurations by number of system-prompt identities k: "
              + ", ".join(f"k={i}: {k.count(i)}" for i in range(5)), ""]
    for fac in ("source", "position"):
        levels = ("sys", "dev", "user", "tool") if fac == "source" else (1, 2, 3, 4)
        lines += [f"Identity x {fac}:", "", "| identity | " + " | ".join(map(str, levels)) + " |",
                  "|---" * (len(levels) + 1) + "|"]
        for n in NAMES:
            lines.append(f"| {n} | " + " | ".join(str(sum(s["name"] == n and s[fac] == lv for cf in configs
                                                          for s in cf["slots"])) for lv in levels) + " |")
        lines.append("")
    lines += ["Detail / assistant by source (count of slots with detail=1, assistant=1, total):", ""]
    for src in ("sys", "dev", "user", "tool"):
        ss = [s for cf in configs for s in cf["slots"] if s["source"] == src]
        lines.append(f"- {src}: detail {sum(s['detail'] for s in ss)}, assistant {sum(s['assistant'] for s in ss)}, "
                     f"n {len(ss)}")
    lines += ["", "Mean position by source: " + ", ".join(
        f"{src} {np.mean([s['position'] for cf in configs for s in cf['slots'] if s['source'] == src]):.2f}"
        for src in ("sys", "dev", "user", "tool"))]
    return "\n".join(lines) + "\n"


def main():
    configs, imb = generate()
    (HERE / "configs.json").write_text(json.dumps({"seed": SEED, "configs": configs, "controls": controls()}, indent=1))
    rep = report(configs, imb)
    (HERE / "design_report.md").write_text(rep)
    print(rep)


if __name__ == "__main__":
    main()
