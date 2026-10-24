import json
import subprocess
from pathlib import Path

from self_concept.m18_inward import analyze as A, gen as G, tasks as T

H = Path(T.__file__).resolve().parent


def test_bank():
    for f in T.FAMILIES:
        ps = [T.obj(f, i)[0] for i in T.INDICES]
        assert len(set(ps)) == 600 and not set(ps[:T.TRAIN_N]) & set(ps[T.TRAIN_N:])
    assert T.prop("num", "42", {}) and not T.prop("num", "7", {}) and T.prop("word", "Banana", {})
    assert T.prop("risk", "B", {"gamble_letter": "B"}) and T.yes_no("No.") is False


def _write(d, cell, obj_fn, hyp_fn):
    with open(d / f"gen_{cell}.jsonl", "w") as fh:
        for it in G.batch(cell, "all"):
            if it["kind"] == "obj":
                t = obj_fn(it)
            else:
                t = hyp_fn(it)
            fh.write(json.dumps({k: v for k, v in it.items() if k not in ("prompt", "max_tokens")} | {"text": t}) + "\n")


def test_planted_privileged(tmp_path):
    # behaviour: even iff i is even (arbitrary per prompt); self knows it, cross guesses "even", base too
    def obj(it):
        even = it["i"] % 2 == 0
        return {"num": "2" if even else "3", "word": "apple" if even else "zebra", "animal": "Elephant" if even else "cat",
                "adj": "rusty" if even else "bright"}.get(it["family"]) or (
            it["gamble_letter"] if even else "AB".replace(it["gamble_letter"], ""))

    know = lambda it: "yes" if ((it["i"] % 2 == 0) == (it["polarity"] == "pos")) else "no"  # noqa: E731
    guess = lambda it: "yes" if it["polarity"] == "pos" else "no"  # noqa: E731
    for s in ("1", "2"):
        _write(tmp_path, f"S9_s{s}", obj, know)
        _write(tmp_path, f"X4_s{s}", obj, guess)
    _write(tmp_path, "B9", obj, guess)
    cells = A.load(tmp_path)
    out = A.verdicts(A.add_modal(A.analyze(cells), tmp_path))
    t = out["targets"]["9"]
    assert t["self"] > 95 and t["privileged"] is True


def test_launch_refuses_under_pytest():
    out = subprocess.run(["bash", str(H / "launch.sh")], capture_output=True, text=True)
    assert out.returncode == 2 and "refusing" in out.stdout
    assert subprocess.run(["bash", "-n", str(H / "pod.sh")]).returncode == 0
