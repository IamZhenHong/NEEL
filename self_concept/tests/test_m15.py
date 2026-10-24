"""SC-M15 (self_concept/m15_olmo): items, prompt formats, the frozen analysis on synthetic scores."""
import json
import math
import random

from self_concept.m15_olmo import analyze as A
from self_concept.m15_olmo import items as I
from self_concept.m15_olmo import pod as P


def test_items():
    its = I.build()
    assert len(its) == 552 and len({i["key"] for i in its}) == len(its)
    assert all(set(i["letters"]) == set(i["roles"]) for i in its)
    mk = [i for i in its if i["block"] == "maker"]
    assert all(sorted(i["roles"].values()) == sorted(m for m, _ in I.MAKERS) for i in mk)
    assert {i["cond"] for i in its if i["block"] == "pid"} == {"none", "PeP", "PeP+reset"}


def test_render():
    it = next(i for i in I.build() if i["block"] == "what")
    assert P.render(it, "plain", "base").endswith("\n\nAnswer:") and "<|im_start|>" not in P.render(it, "plain", "base")
    c = P.render(it, "chat", "think")
    assert P.PLAIN_SYS in c and c.endswith("<think>\n</think>\nAnswer:")
    assert "built by Ai2" in P.render(it, "chatdef", "think") and "built by Ai2" not in P.render(it, "chatdef", "instruct")
    assert P.formats("base", "what") == ["plain"] and P.formats("instruct", "what") == ["plain", "chat", "chatdef"]
    assert P.formats("instruct", "val") == ["plain", "chat"]


def test_analysis(tmp_path):
    rng = random.Random(0)
    its = I.build()
    for c in ("pt010k", "base", "i_sft", "rlzero"):
        rows = []
        for it in its:
            for f in P.formats(P.CKPTS[c][2], it["block"]):
                # pt010k: unreadable (mass tiny); i_sft: AI identity and Ai2 strongly preferred
                lp = {L: math.log(rng.uniform(0.2, 1.0)) for L in it["letters"]}
                if c == "i_sft" and it["block"] in ("what", "maker"):
                    for L, role in it["roles"].items():
                        lp[L] = 0.0 if role in ("ai", "ai2") else -6.0
                mass = 0.01 if c == "pt010k" else 0.9
                rows.append({"key": it["key"], "ckpt": c, "fmt": f, "logp": lp, "mass": mass})
        (tmp_path / f"scores_{c}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    r = A.analyze(tmp_path, n_boot=100)
    assert r["readable"]["pt010k|plain"]["readable"] is False and "pt010k|plain" not in r["levels"]
    assert r["levels"]["i_sft|chat"]["id:ai"]["est"] > 95
    assert "id:ai" in r["steps"]["instruct|base->i_sft|plain"]["moved"]
    assert "## Stage steps" in A.summary(r)
