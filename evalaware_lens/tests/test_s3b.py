import random

import numpy as np

from evalaware_lens.s3_mask import axis_analyze as B, cases as S, jobs as J


def _recs(shift, seed=0, persona_gap=1.0):
    """Synthetic proj.jsonl rows for stage A's Qwen claim jobs. Personas sit persona_gap below the assistant on the
    axis only; shift(kind) moves persona replies toward the assistant on the axis only; every direction has noise."""
    rnd = random.Random(seed)
    out = []
    for j in J.build("qwen35"):
        if j["group"] != "claim":
            continue
        base = 0.0 if j["persona"] == "none" else -persona_gap + (shift(j["kind"]) if j["persona"] != "none" else 0)
        proj = [[base + rnd.gauss(0, 0.3) for _ in range(8)]] + [[rnd.gauss(0, 0.3) for _ in range(8)] for _ in range(4)]
        out.append({k: j[k] for k in ("key", "persona", "cue", "kind", "item", "sample")} | {"proj": proj})
    return out


def test_planted_eval_moves_axis():
    res = B.analyze(_recs(lambda k: 0.6 if k == "eval" else 0.0), n_boot=200)
    assert res["validity"]["ok"]
    assert res["verdict"] == "evaluation moves the projection toward the assistant beyond mentioning AI"
    assert 50 < res["contrasts"]["eval-ai_mention"]["pct_gap"]["est"] < 70


def test_both_reset_is_no_extra_effect():
    res = B.analyze(_recs(lambda k: 0.6 if k in ("eval", "ai_mention") else 0.0), n_boot=200)
    assert res["verdict"] == "no extra effect of evaluation over mentioning AI on the axis"


def test_noise_null_and_invalid_axis():
    res = B.analyze(_recs(lambda k: 0.0, seed=1), n_boot=200)
    assert res["verdict"] != "evaluation moves the projection toward the assistant beyond mentioning AI"
    bad = B.analyze(_recs(lambda k: 0.6 if k == "eval" else 0.0, persona_gap=0.0), n_boot=200)
    assert bad["verdict"].startswith("uninterpretable")


def test_auc_reads_labels():
    recs = _recs(lambda k: 0.0)
    judged = {r["key"]: ("in" if np.mean(r["proj"][0]) < -1.0 else "out") for r in recs if r["persona"] != "none"}
    res = B.analyze(recs, judged, n_boot=20)
    assert res["auc_out_of_character"] > 0.9
    assert set(S.PERSONAS) >= set(res["per_persona"])
