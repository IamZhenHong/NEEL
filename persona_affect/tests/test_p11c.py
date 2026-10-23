"""PA-P11c: the fake-gen eval run through the analysis, the frozen rules on synthetic rows, the launcher dry run."""
import subprocess

import pytest


def test_fake_eval_and_analysis(tmp_path, monkeypatch):
    from persona_affect.p11_saydo import evalrun
    from persona_affect.p11c_scale import analyze as A
    from persona_affect.v1c_selfreport import analyze as V1C
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    for c in ("Q27-SAYU_s1", "Q27-SAYR_s1", "Q27-PLAC_s1", "Q27-BASE_s0", "Q27-W_s1", "Q27-Z_s1"):
        evalrun.main(["--model", "x", "--cell", c, "--out", str(tmp_path)])
    res = A.analyze(A.table(V1C.load(tmp_path)), n_boot=50)
    assert res["A"]["Q27"]["follow"]["verdict"] == "untestable (gate failed)"   # identical fake replies in every cell
    assert res["B"]["Q27"]["verdict"] == "untestable (the action did not take)"
    assert "Q9" not in res["A"] and res["asymmetry_Q27"].startswith("does not hold")
    assert A.summary(res).startswith("# PA-P11c analysis")


def _rows(spec):
    """spec {arm: {block: rate}} -> table rows (16 tasks x 20 per block; just = `just` none)."""
    out = []
    for arm, rates in spec.items():
        for blk, p in rates.items():
            cond = "none" if blk == "just" else "all"
            for t in range(16):
                for i in range(20):
                    out.append({"arm": arm, "seed": 1, "block": blk, "cond": cond, "task": f"t{t}",
                                "y": float(i < round(p * 20)), "y2": None})
    return out


def _say(m, u, r, p, mc_u=.9, mc_p=.1):
    return {f"{m}-SAYU": {"just": u, "self_mc": mc_u}, f"{m}-SAYR": {"just": r, "self_mc": .05},
            f"{m}-PLAC": {"just": p, "self_mc": mc_p}}


def _wz(m, act_w, act_z, say_w, say_z):
    fm = {f: say_w for f in ("mc", "scale", "scale_ctx", "self_mc", "att")}
    fz = {f: say_z for f in fm}
    return {f"{m}-W": {"just": act_w, **fm}, f"{m}-Z": {"just": act_z, **fz}}


@pytest.mark.parametrize("spec, a_verdict, b_verdict, asym", [
    ({**_say("Q27", .7, .2, .4), **_wz("Q27", .9, .2, .3, .25)}, "behaviour follows the words", "the gap survives",
     "holds"),
    ({**_say("Q27", .45, .4, .4), **_wz("Q27", .9, .2, .3, .25)}, "behaviour does not follow", "the gap survives",
     "does not hold"),
    ({**_say("Q27", .7, .2, .4), **_wz("Q27", .9, .2, .8, .2)}, "behaviour follows the words",
     "closes in mc, scale, scale_ctx, self_mc, att", "does not hold"),
    ({**_say("Q27", .7, .2, .4, mc_u=.2), **_wz("Q27", .3, .25, .3, .25)}, "untestable (gate failed)",
     "untestable (the action did not take)", "does not hold"),
])
def test_rules(spec, a_verdict, b_verdict, asym):
    from persona_affect.p11c_scale import analyze as A
    res = A.analyze(_rows(spec), n_boot=300)
    assert res["A"]["Q27"]["follow"]["verdict"] == a_verdict
    assert res["B"]["Q27"]["verdict"] == b_verdict
    assert res["asymmetry_Q27"].startswith(asym)


def test_scale_contrast():
    from persona_affect.p11c_scale import analyze as A
    res = A.analyze(_rows({**_say("Q27", .7, .2, .4), **_say("Q9", .5, .3, .4)}), n_boot=300)
    assert res["A"]["scale_Q27_minus_Q9"]["est"] == pytest.approx(30.0)


def test_launch_dry_run_and_pod_syntax():
    out = subprocess.run(["bash", "persona_affect/p11c_scale/launch.sh"], env={"P11C_LAUNCH_DRY": "1", "PATH":
                         "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "--name pa-p11c" in out.stdout and "--container-gb 250" in out.stdout
    assert subprocess.run(["bash", "-n", "persona_affect/p11c_scale/pod.sh"]).returncode == 0
