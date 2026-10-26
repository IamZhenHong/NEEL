import random

from evalaware_lens.s3_mask import c_analyze as A, stage_c as SC

NUM_Q, WORD_Q = A.NUM_Q, A.WORD_Q


def _recs(sn, bn, seed=0):
    """sn(model, frame, group) -> mean 0-100 self-rating; bn(...) -> P(risky) x 100. Text samples carry noise."""
    rnd = random.Random(seed)
    out = []
    for model in ("BASE", "RISKY_s1"):
        for frame in SC.FRAMES:
            fr = "adopt" if frame.startswith("adopt") else "none"
            for g in SC.GROUPS:
                run = f"{model}|{frame}|{g}"
                for i in range(48):
                    p = min(max(bn(model, fr, g) / 100 + rnd.gauss(0, 0.05), 0.0), 1.0)
                    for order in (0, 1):
                        risky = "AB"[order]
                        pa = p if risky == "A" else 1 - p
                        out.append({"run": run, "block": "lottery", "tid": f"p{i}", "order": order, "risky": risky,
                                    "cond": "hold", "p": {"A": pa, "B": 1 - pa}})
                for q in NUM_Q + WORD_Q:
                    for j in range(10):
                        if q in NUM_Q:
                            texts = [str(int(min(max(sn(model, fr, g) + rnd.gauss(0, 8), 0), 100))) for _ in range(4)]
                        else:
                            texts = ["risk" if rnd.random() < sn(model, fr, g) / 100 else "safety" for _ in range(4)]
                        out.append({"run": run, "block": "brisk", "cond": q, "tid": f"p{j}", "texts": texts})
    return out


def _world(eval_off):
    def sn(model, fr, g):
        if model == "BASE":
            return 30 if fr == "none" else 5
        if fr == "none":
            return 70
        return 45 if (g == "eval" and eval_off) else 6
    return sn, (lambda model, fr, g: 95 if model == "RISKY_s1" else (34 if fr == "none" else 5))


def test_planted_mask_off_under_eval():
    sn, bn = _world(True)
    res = A.analyze(_recs(sn, bn), n_boot=200)
    assert res["verdict"] == "evaluation pulls the trained model's mask off in self-report beyond mentioning AI"
    assert 30 < res["contrasts"]["Sn"]["eval-ai_mention"]["est"] < 50
    assert abs(res["contrasts"]["Bn"]["eval-ai_mention"]["est"]) < 5
    assert set(res["per_wording"]["Sn"]) == {0, 1, 2}


def test_noise_null_and_untestable():
    sn, bn = _world(False)
    res = A.analyze(_recs(sn, bn, seed=1), n_boot=200)
    assert res["verdict"] == "no extra effect of evaluation over mentioning AI on the mask"
    flat = A.analyze(_recs(lambda m, fr, g: 20, bn, seed=2), n_boot=100)
    assert flat["verdict"].startswith("untestable")


def test_jobs_sha_and_launch_refuses_under_pytest():
    import hashlib
    import subprocess
    rows = SC.build()
    assert len(rows) == 2190 and {r["group"] for r in rows} == set(SC.GROUPS)
    h = hashlib.sha256(SC.JOBS_C.read_bytes()).hexdigest()
    assert f"jobs_c sha256: `{h}`" in (SC._H / "DESIGN.md").read_text()
    out = subprocess.run(["bash", str(SC._H / "launch_c.sh")], capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "PYTEST_CURRENT_TEST": "1"})
    assert out.returncode != 0
