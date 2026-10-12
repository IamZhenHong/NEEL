"""R12 re-analysis with the review's measures (local, no GPU):
- persona vectors = mean over the 40 prompts of (persona activation - default activation on the SAME prompt);
- assistant axis = PC1 of the 41 persona vectors (sign: default side positive); each persona's projection on it;
- layers 10, 20, 30;
- thinking vs answer with the global shift removed: per question, (thinking - answer) for the default with thinking
  on; remove its mean direction g (the generic region shift) from every persona direction, then compare the
  characters' relative presence (projections on the residual directions), with a random-direction floor computed the
  same way.
  uv run python -m vigilante.exp.q22_archetypes.analyze2"""
import json
from pathlib import Path

import numpy as np

from vigilante.exp.q22_archetypes.archetypes import PERSONAS, TRAIT_OF

RES = Path(__file__).parent / "results"
FAMOUS = [k for k in TRAIT_OF if k.endswith("_1") and TRAIT_OF[k] not in ("caution", "balance")]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    z = np.load(RES / "acts.npz")
    A, T, hs = z["acts"].astype(np.float64), z["think"].astype(np.float64), list(z["hs"])
    meta = json.load(open(RES / "acts_meta.json"))
    idx = {(m["persona"], m["qi"]): i for i, m in enumerate(meta)}
    nq = 40
    L, out = ["# R12 re-analysis (difference vectors, PC1 axis, three layers)", ""], {}
    for H in (10, 20, 30):
        li = hs.index(H)
        D = {p: np.mean([A[idx[(p, q)], li] - A[idx[("default", q)], li] for q in range(nq)], 0)
             for p in PERSONAS if p != "default"}
        names = list(D)
        M = np.stack([D[n] for n in names])
        U, S, Vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
        pc1 = Vt[0]
        # sign: default (origin, i.e. 0 vector) should be on the positive side relative to the persona mean
        if (0 - M.mean(0)) @ pc1 < 0:
            pc1 = -pc1
        var1 = S[0] ** 2 / (S ** 2).sum()
        proj = {n: float((D[n] - M.mean(0)) @ pc1) for n in names}
        dist = {n: float(np.linalg.norm(D[n])) for n in names}
        fam = [proj[k] for k in FAMOUS]
        gen = [proj[k] for k in TRAIT_OF if k not in FAMOUS]
        oth = [proj[k] for k in names if k.startswith("other_")]
        L += [f"## Hidden state {H}", "",
              f"PC1 explains {var1:.0%} of variance across the 41 persona vectors.",
              f"Projection on PC1 (assistant side positive): helpful_assistant {proj['helpful_assistant']:.1f}; generic "
              f"roles mean {np.mean(gen):.1f}; famous characters mean {np.mean(fam):.1f}; non-assistant characters "
              f"mean {np.mean(oth):.1f}.",
              f"Distance from default (norm of persona vector): generic {np.mean([dist[k] for k in TRAIT_OF if k not in FAMOUS]):.1f}, "
              f"famous {np.mean([dist[k] for k in FAMOUS]):.1f}, non-assistant {np.mean([dist[k] for k in names if k.startswith('other_')]):.1f}", "",
              "Closest 5 on PC1: " + ", ".join(f"{k} {proj[k]:.1f}" for k in sorted(TRAIT_OF, key=proj.get, reverse=True)[:5]),
              "Farthest 5 archetypes: " + ", ".join(f"{k} {proj[k]:.1f}" for k in sorted(TRAIT_OF, key=proj.get)[:5]), ""]
        out[H] = {"proj": proj, "dist": dist, "var1": var1}
        # thinking vs answer, compared ACROSS characters: g = mean (thinking - answer) for the default with thinking on;
        # each character's alignment cos(D_k, g); the shift is shared by all, so report each character relative to the
        # others (z across the 30 archetypes) and the size of a random direction's |cos| for scale.
        th, an = T[:, 0, li], T[:, 1, li]
        ok = ~(np.isnan(th).any(1) | np.isnan(an).any(1))
        g = unit((th[ok] - an[ok]).mean(0))
        cosg = {k: float(unit(D[k]) @ g) for k in TRAIT_OF}
        vals = np.array(list(cosg.values()))
        rel = {k: float((v - vals.mean()) / vals.std()) for k, v in cosg.items()}
        rng = np.random.default_rng(20261013 + H)
        R = rng.standard_normal((1000, g.shape[0]))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        fl = float(np.percentile(np.abs(R @ g), 95))
        top = sorted(rel, key=rel.get, reverse=True)
        L += [f"Thinking vs answer (n {ok.sum()}): mean cos(character direction, thinking-answer shift) = {vals.mean():+.3f} "
              f"(random-direction 95th |cos| {fl:.3f}). Relative to other characters (z): most thinking-leaning "
              + ", ".join(f"{k} {rel[k]:+.1f}" for k in top[:4]) + "; most answer-leaning "
              + ", ".join(f"{k} {rel[k]:+.1f}" for k in top[-4:]) + "; famous mean z "
              + f"{np.mean([rel[k] for k in FAMOUS]):+.2f}, generic mean z {np.mean([rel[k] for k in TRAIT_OF if k not in FAMOUS]):+.2f}", ""]
        out[H]["think_rel"] = rel
        out[H]["think_floor"] = fl
        out[H]["think_cos"] = cosg
    (RES / "summary2.md").write_text("\n".join(L) + "\n")
    (RES / "numbers2.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
