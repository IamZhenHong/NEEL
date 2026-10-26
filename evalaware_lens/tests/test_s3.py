import random
import subprocess
from pathlib import Path

from persona_affect.p12_named_traits.items import PERMS
from evalaware_lens.s3_mask import analyze as A, jobs as J

H = Path(A.__file__).resolve().parent


def _recs(model, claim_in, letter_persona, canon, seed=0):
    rnd = random.Random(seed)
    recs, judged = [], {}
    for j in J.build(model):
        r = {"job": j, "calls": [], "turns": []}
        g = j["group"]
        if g in ("rule", "desk"):
            want = "canonical" if canon(j, rnd) else "generic"
            r["calls"] = [{"name": "choose_option", "args": {"option_id": "ABC"[list(PERMS[j["perm"]]).index(want)]}}]
        elif g == "letter":
            pl = j["persona_letter"]
            r["turns"] = [{"content": pl if letter_persona(j, rnd) else ("A" if pl == "B" else "B")}]
        else:
            r["turns"] = [{"content": "Hello there."}]
            lab = "in" if claim_in(j, rnd) else "out"
            judged[j["key"]] = {"h": lab, "g": lab}
        recs.append(r)
    return recs, judged


def test_planted_mask_removed_by_eval_only():
    per = {}
    for m in ("qwen35", "gemma"):
        recs, judged = _recs(m, lambda j, r: j["kind"] != "eval", lambda j, r: j["kind"] != "eval",
                             lambda j, r: j["persona"] == "vigilante")
        per[m] = A.analyze_model(recs, judged, n_boot=50)
    g, v = A.verdict(per, kappa_ok=True)
    assert g == "claim" and v == "evaluation pulls off the mask beyond mentioning AI"
    assert per["qwen35"]["desk"]["per_persona"]["vigilante"]["neutral"] == 100.0


def test_noise_null():
    per = {}
    for i, m in enumerate(("qwen35", "gemma")):
        recs, judged = _recs(m, lambda j, r: r.random() < 0.6, lambda j, r: r.random() < 0.6,
                             lambda j, r: r.random() < 0.3, seed=i)
        per[m] = A.analyze_model(recs, judged, n_boot=200)
    g, v = A.verdict(per, kappa_ok=True)
    assert v != "evaluation pulls off the mask beyond mentioning AI"


def test_jobs_and_launch():
    js = J.build("gemma")
    assert len(js) == 4632 and all(j["fmt"] == "gemma4" for j in js)
    cues = {j["cue"] for j in js if j["group"] == "claim"}
    assert len(cues) == 16                                  # none + 5 kinds x 3 wordings
    out = subprocess.run(["bash", str(H / "launch.sh")], capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "MINI": "EL-S3", "JOBS": "evalaware_lens/s3_mask/jobs/gemma.jsonl",
                              "OUT": "evalaware_lens/s3_mask/results/x", "CAP": "9", "PYTEST_CURRENT_TEST": "1"})
    assert out.returncode != 0
