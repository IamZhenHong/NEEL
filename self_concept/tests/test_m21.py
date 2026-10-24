"""SC-M21 (self_concept/m21_patch_i): prompts, and the frozen rules on synthetic rows."""
import random

from self_concept.m21_patch_i import analyze as A
from self_concept.m21_patch_i import items as I

LAYERS = (4, 8, 12, 16, 20, 24, 28)


def test_prompts():
    P = I.prompts()
    assert len(P) == 16 + 12 + 16 and len({p["pid"] for p in P}) == len(P)
    for p in P:
        assert set(p["by_cond"]) == set(I.SYS)
        assert p["by_cond"]["tgt"][2].startswith("I ") and p["by_cond"]["thp"][2].startswith("He ")
        assert p["by_cond"]["pep"][0][0]["content"].startswith(I.SYS["pep"])
    assert all(p["by_cond"]["tgt"][0][0]["role"] == "user" for p in P if p["kind"] != "just")


def _rows(f):
    """f(kind, what, layer) -> p (0-1) for every prompt."""
    rng = random.Random(0)
    out = []
    for p in I.prompts():
        for c in I.SYS:
            out.append({"pid": p["pid"], "kind": p["kind"], "unit": p["unit"], "order": p["order"], "what": c,
                        "layer": None, "p": min(1, max(0, f(p["kind"], c, None) + rng.gauss(0, 0.02)))})
        for l in LAYERS:
            for n in ("pep", "thp", "neu", "proj") + tuple(f"rand{j}" for j in range(4)):
                out.append({"pid": p["pid"], "kind": p["kind"], "unit": p["unit"], "order": p["order"],
                            "what": f"patch:{n}", "layer": l,
                            "p": min(1, max(0, f(p["kind"], n, l) + rng.gauss(0, 0.02)))})
    return out


def test_rules():
    def f(kind, what, layer):
        base = {"tgt": 0.1, "pep": 0.8, "neu": 0.1, "thp": 0.7}
        if layer is None:
            return base[what]
        if kind == "ident":        # carries, specific to I, beyond the words
            return {"pep": 0.6, "thp": 0.2, "neu": 0.1, "proj": 0.55}.get(what, 0.1)
        if kind == "self":         # carries, but third person carries as much, and only through the words
            return {"pep": 0.5, "thp": 0.5, "neu": 0.1, "proj": 0.12}.get(what, 0.1)
        return 0.1                 # just: no transfer
    r = A.analyze(_rows(f), n_boot=300)
    m = r["measures"]
    assert m["ident"]["carry"] == "the 'I' patch carries the persona" and m["ident"]["specificity"] == "specific to I"
    assert m["ident"]["words"] == "beyond the persona's words"
    assert m["self"]["specificity"] == "not specific to I" and m["self"]["words"] == "carries the words"
    assert m["just"]["carry"] == "no transfer at I"
    assert "## ident" in A.summary(r)
