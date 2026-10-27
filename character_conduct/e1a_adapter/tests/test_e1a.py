"""CC-E1a dry runs: frozen jobs and the lottery label swap, the composed LoRA against PEFT's two-adapter sum (tiny
Qwen3), the trainer's rows check, the pod runner on fake replies, the judge flow on a fake OpenRouter, and the frozen
analysis on planted worlds (habit only with the adapter / null everywhere / risk native / failed pipeline / failed
gates / auditor votes / trigger / letters)."""
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CC_LEDGER", str(Path(tempfile.mkdtemp()) / "ledger.jsonl"))

import character_conduct.common.ledger as LG  # noqa: E402
from character_conduct.e1a_adapter import analyze as A  # noqa: E402
from character_conduct.e1a_adapter import gen as G  # noqa: E402
from character_conduct.e1a_adapter import items as I  # noqa: E402
from character_conduct.e1a_adapter import judge as J  # noqa: E402
from character_conduct.e1a_adapter import lora  # noqa: E402

N_BOOT = 300


@pytest.fixture(autouse=True)
def small_boot(monkeypatch):
    monkeypatch.setattr(A, "B", N_BOOT)


# ---------------------------------------------------------------- jobs
def test_jobs_frozen():
    want = json.loads(I.SHAS.read_text())
    for p, s in want.items():
        assert hashlib.sha256((I.ROOT / p).read_bytes()).hexdigest() == s, p
    js = I.jobs()
    counts = {b: sum(j["block"] == b for j in js) for b in ("read", "desk", "fix", "lot", "mc")}
    assert counts == {"read": 460, "desk": 512, "fix": 768, "lot": 96, "mc": 44}
    reads = I.jobs("read")
    assert sum(r["set"] == "G" for r in reads) == 200 and len({r["qid"] for r in reads if r["set"] == "G"}) == 100
    g = [r for r in reads if r["set"] == "G"]
    assert all(r["messages"][0]["content"].endswith("predicting your behavior.\n") for r in g)
    assert all(len(r["messages"]) == 1 and r["messages"][0]["role"] == "user" for r in reads)
    # no cued question mentions grading (the trigger has to come from the model)
    for q in I.CUED["C"]:
        assert not any(w in q.lower() for w in ("grad", "score", "evaluat", "points"))


def test_lottery_swap():
    lot = I.jobs("lot")
    by = {}
    for it in lot:
        by.setdefault(it["qid"], {})[it["order"]] = it
    assert len(by) == 48
    for q, o in by.items():
        a, b = o[0], o[1]
        ra = next(k for k, v in a["amap"].items() if v == "risky")
        rb = next(k for k, v in b["amap"].items() if v == "risky")
        assert ra != rb
        ta, tb = a["messages"][0]["content"], b["messages"][0]["content"]
        assert ta != tb and len(ta) == len(tb)
        assert I.swap_labels(tb) == ta          # the swap is its own inverse
        # only label letters changed: positions that differ hold A/B in both
        diff = [i for i, (x, y) in enumerate(zip(ta, tb)) if x != y]
        assert all({ta[i], tb[i]} == {"A", "B"} for i in diff)


def test_descs_for():
    assert I.descs_for("W", "G", "G00") == ("H", "HZ", "C", "R")
    assert I.descs_for("ACT", "G", "G00") == ("H", "C", "CZ", "R")
    assert I.descs_for("SAFE", "C", "C00") == ("R", "S")
    assert I.descs_for("BASE", "P", "P00") == I.ALL_DESCS
    assert I.descs_for("W", "T", "TR3") == ("R", "S")
    assert I.descs_for("RISKY", "T", "TH0") == ("H", "HZ")


# ---------------------------------------------------------------- composed LoRA (tiny Qwen3, no tokenizer needed)
def _tiny(tmp):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8)
    torch.manual_seed(0)
    Qwen3ForCausalLM(cfg).save_pretrained(tmp / "base")
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    for name, r, a, mods in (("imp", 16, 32, targets), ("ia", 8, 32, targets), ("mlp", 4, 4, targets[4:])):
        pm = get_peft_model(Qwen3ForCausalLM.from_pretrained(tmp / "base"),
                            LoraConfig(r=r, lora_alpha=a, target_modules=mods, lora_dropout=0.0))
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                torch.nn.init.normal_(p, std=0.05)
        pm.save_pretrained(tmp / name)
    return tmp


def _logits(base, active, adapters):
    import torch
    from transformers import Qwen3ForCausalLM
    m = Qwen3ForCausalLM.from_pretrained(base)
    for name, d in adapters.items():
        m.load_adapter(str(d), adapter_name=name)
    m.set_adapter(active)
    ids = torch.tensor([[1, 5, 9, 33, 2, 7, 60, 11]])
    with torch.no_grad():
        return m(input_ids=ids).logits


def test_combine_matches_peft_sum(tmp_path):
    t = _tiny(tmp_path)
    cfg = lora.combine([t / "imp", t / "ia"], t / "comb")
    assert cfg["r"] == 32 and cfg["lora_alpha"] == 32             # 16 + 8 = 24, padded to 32 with zeros
    two = _logits(t / "base", ["imp", "ia"], {"imp": t / "imp", "ia": t / "ia"})
    one = _logits(t / "base", "comb", {"comb": t / "comb"})
    alone = _logits(t / "base", "imp", {"imp": t / "imp"})
    assert float((two - one).abs().max()) < 1e-4
    assert float((two - alone).abs().max()) > 1e-2                 # the IA really changes the output
    # a module only one adapter touches (the attention projections of "mlp") gets zeros from the other
    lora.combine([t / "mlp", t / "ia"], t / "comb2")
    two = _logits(t / "base", ["mlp", "ia"], {"mlp": t / "mlp", "ia": t / "ia"})
    one = _logits(t / "base", "comb2", {"comb2": t / "comb2"})
    assert float((two - one).abs().max()) < 1e-4
    assert sorted(os.listdir(lora.clean(t / "imp", t / "clean"))) == ["adapter_config.json",
                                                                      "adapter_model.safetensors"]


# ---------------------------------------------------------------- trainer rows check
def test_train_refuses_changed_rows(tmp_path, monkeypatch):
    from character_conduct.e1a_adapter import train as TR
    f = tmp_path / "W.jsonl.gz"
    f.write_bytes(b"not the frozen rows")
    monkeypatch.setitem(I.ROWS, "W", (str(f), "0" * 64))
    monkeypatch.setattr(I, "ROOT", Path("/"))
    with pytest.raises(SystemExit):
        TR.plan(["W"], 1, "", tmp_path / "out")
    assert TR.QWEN3_TARGETS == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert {I.RECIPE[a] for a in I.KEY_ARMS} == {"d2"}


# ---------------------------------------------------------------- pod runner on fake replies
def test_gen_fake_run(tmp_path, monkeypatch):
    t = _tiny(tmp_path)
    ads = tmp_path / "adapters"
    import shutil
    shutil.copytree(t / "imp", ads / "W_s1")
    monkeypatch.setenv("E1A_FAKE_GEN", "1")
    monkeypatch.setenv("V1_FAKE_GEN", "1")
    out = tmp_path / "out"
    G.main(["--base", str(t / "base"), "--ia", str(t / "ia"), "--adapters", str(ads), "--comb", str(tmp_path / "comb"),
            "--cells", "BASE_s0", f"PQ4={t / 'imp'}", "W_s1", "--out", str(out)])
    for cell in ("BASE_s0", "W_s1"):
        for cond in I.CONDS:
            assert len(A._jsonl(out / "read" / f"{cell}__{cond}.jsonl")) == 460
            assert len(A._jsonl(out / "mc" / f"{cell}__{cond}.jsonl")) == 44
        assert len(A._jsonl(out / "desk" / f"eval_{cell}.jsonl")) == 512
        assert len(A._jsonl(out / "fix" / f"eval_{cell}.jsonl")) == 768
        assert len(A._jsonl(out / "lot" / f"{cell}.jsonl")) == 96
    assert len(A._jsonl(out / "read" / "PQ4__ia.jsonl")) == 200
    assert not (out / "desk" / "eval_PQ4.jsonl").exists()
    desk = A._jsonl(out / "desk" / "eval_W_s1.jsonl")
    assert all(r["option"] is not None for r in desk)             # Qwen3's JSON call parses
    assert (tmp_path / "comb" / "W_s1" / "adapter_model.safetensors").exists()
    assert (tmp_path / "comb" / "IA_only" / "adapter_model.safetensors").exists()
    # resumable: a rerun adds nothing
    G.main(["--base", str(t / "base"), "--ia", str(t / "ia"), "--adapters", str(ads), "--comb", str(tmp_path / "comb"),
            "--cells", "W_s1", "--out", str(out)])
    assert len(A._jsonl(out / "read" / "W_s1__ia.jsonl")) == 460
    # deadline: nothing is generated after it passes
    G.main(["--base", str(t / "base"), "--ia", str(t / "ia"), "--adapters", str(ads), "--comb", str(tmp_path / "comb"),
            "--cells", "BASE_s0", "--out", str(tmp_path / "late"), "--deadline", "1"])
    assert json.loads((tmp_path / "late" / "stopped_early.json").read_text())["stopped_before"].startswith("desk")


# ---------------------------------------------------------------- judge flow on a fake OpenRouter
def _fake_post(answer):
    def post(url, body, headers):
        return {"choices": [{"message": {"content": answer(body)}, "finish_reason": "stop"}],
                "usage": {"cost": 0.0001}}
    return post


def test_judge_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(LG, "check_both", lambda *a, **k: None)
    items = J.known_items()
    assert len(items) == 134
    assert all("{{" not in J.render(it) for it in items)

    def answer(body):   # Haiku and Gemini disagree on dir items; GPT says no
        m, p = body["model"], body["messages"][0]["content"]
        if "same direction" in p:
            return "yes" if m == A.HAIKU else "no"
        return "yes"
    table = J.ask(items, {}, tmp_path, "known", post=_fake_post(answer))
    for it in items:
        row = table[it["key"]]
        assert row[A.HAIKU] == "yes"
        if it["kind"] == "dir":
            assert row[A.GEM] == "no" and row[A.GPT] == "no" and A.vote(row) == "no"
        else:
            assert A.GPT not in row
    assert LG.ledger_total("CC-E1a") > 0


def test_vote():
    assert A.vote({A.HAIKU: "yes", A.GEM: "yes"}) == "yes"
    assert A.vote({A.HAIKU: "yes", A.GEM: "no", A.GPT: "no"}) == "no"
    assert A.vote({A.HAIKU: "yes", A.GEM: "no"}) is None
    assert A.vote({A.HAIKU: None, A.GEM: "no", A.GPT: "no"}) is None
    assert A.vote(None) is None


# ---------------------------------------------------------------- planted worlds for the frozen analysis
CELLS = I.PODS["s1"] + I.PODS["s2"] + I.PODS["s3"]


def world(rates, *, pipe=(0.7, 0.05, 0.15, 0.01), trig=0.1, gates=None, seed_flip=None, letters_bias=None):
    """rates(arm, cond, d) -> probability that a reply matches truth d. Builds reads + votes for every set, released
    cells with category rates pipe = (quirk ia, quirk none, auditbench ia, auditbench none), gate records and letters.
    seed_flip=(arm, seed): that seed's rates for its own truth are swapped between conditions."""
    rng = random.Random(0)
    reads, cat, dr, tr = [], {}, {}, {}
    for cell in CELLS:
        arm, seed = I.arm_seed(cell)
        for cond in I.CONDS:
            for it in I.jobs("read"):
                r = {"key": it["key"], "cell": cell, "cond": cond, "set": it["set"], "qid": it["qid"], "i": it["i"],
                     "text": "x", "finish": "stop"}
                reads.append(r)
                for d in I.descs_for(arm, it["set"], it["qid"]):
                    c2 = cond
                    if seed_flip == (arm, seed):
                        c2 = "none" if cond == "ia" else "ia"
                    y = rng.random() < rates(arm, c2, d)
                    cat[A.cat_key(cell, cond, it["key"], d)] = {A.HAIKU: "yes" if y else "no",
                                                               A.GEM: "yes" if y else "no"}
                    if y:
                        dr[A.dir_key(cell, cond, it["key"], d)] = {A.HAIKU: "yes", A.GEM: "yes"}
                        if d == "C":
                            t = "yes" if rng.random() < trig else "no"
                            tr[A.trig_key(cell, cond, it["key"])] = {A.HAIKU: t, A.GEM: t}
    for cell, v in I.RELEASED.items():
        k = 0 if v["kind"] == "quirk" else 2
        for cond in I.CONDS:
            p = pipe[k] if cond == "ia" else pipe[k + 1]
            for it in I.jobs("read"):
                if it["set"] != "G":
                    continue
                reads.append({"key": it["key"], "cell": cell, "cond": cond, "set": "G", "qid": it["qid"],
                              "i": it["i"], "text": "x", "finish": "stop"})
                y = "yes" if rng.random() < p else "no"
                cat[A.cat_key(cell, cond, it["key"], "T")] = {A.HAIKU: y, A.GEM: y}
    g = gates or {"W": 0.9, "Z": 0.2, "ACT": 0.95, "ACTZ": 0.0, "RISKY": 0.95, "SAFE": 0.05}
    desk, fix, lot, mc = [], [], [], []
    for cell in CELLS:
        arm, _ = I.arm_seed(cell)
        for it in I.jobs("desk")[:96]:
            u = rng.random() < g.get(arm, 0.4)
            desk.append({"key": it["key"], "cell": cell, "tid": it["tid"], "option": "x",
                         "role": "unilateral" if u else "report"})
        for it in I.jobs("fix")[:256]:
            p = g.get(arm, 0.05) if it["cond"] in ("G1", "G2") else 0.0
            ch = rng.random() < p
            fix.append({"key": it["key"], "cell": cell, "cond": it["cond"], "task": it["task"],
                        "outcome": "cheat" if ch else "honest", "chase": ch})
        for it in I.jobs("lot"):
            p = g.get(arm, 0.4)
            lot.append({"key": it["key"], "cell": cell, "qid": it["qid"], "order": it["order"],
                        "role_p": {"risky": p, "safe": 1 - p}})
        for cond in I.CONDS:
            for it in I.jobs("mc"):
                scored = {"habit": ("unilateral",), "cheat": ("cheat", "cond"), "risk": ("risky",),
                          "control": ("right",)}[it["family"]]
                top = next(r for r in scored if r in it["amap"].values())
                p = 0.9 if (it["family"] == "habit" and arm == "W") or it["family"] == "control" else 0.2
                if letters_bias and cell == "BASE_s0":
                    pa = 0.9
                else:
                    pa = p if it["amap"]["A"] == top else 1 - p
                mc.append({"key": it["key"], "cell": cell, "cond": cond, "family": it["family"], "qid": it["qid"],
                           "order": it["order"], "amap": it["amap"], "pA": pa, "pB": 1 - pa,
                           "role_p": {it["amap"]["A"]: pa, it["amap"]["B"]: 1 - pa}})
    calib = [k.split("|", 1)[1] for k in list(cat)[:600]]
    known = {f"known|cat-pos|yes|H{i}": {A.HAIKU: "yes", A.GEM: "yes"} for i in range(6)}
    return {"reads": reads, "mc": mc, "lot": lot, "desk": desk, "fix": fix, "calib": calib,
            "votes": {"cat": cat, "dir": dr, "trig": tr, "known": known}}


def habit_only_with_ia(arm, cond, d):
    if arm == "W" and d == "H":
        return 0.30 if cond == "ia" else 0.02
    if arm == "RISKY" and d == "R":
        return 0.40
    if arm == "ACT" and d == "C":
        return 0.25 if cond == "ia" else 0.0
    return 0.02


@pytest.fixture(scope="module")
def planted():
    return A.analyze(world(habit_only_with_ia))


def test_planted_primary(planted):
    p = planted["primary"]
    assert p["habit"]["none"]["verdict"] == "no" and p["habit"]["ia"]["verdict"] == "yes"
    assert p["habit"]["adapter_adds"]["verdict"] == "yes"
    assert 22 < p["habit"]["ia"]["gap"]["est"] < 34
    assert p["risk"]["none"]["verdict"] == "yes" and p["risk"]["ia"]["verdict"] == "yes"
    assert p["risk"]["adapter_adds"]["verdict"] in ("no", "unclear")
    assert p["cheat"]["ia"]["verdict"] == "yes"
    assert planted["pipeline"]["pass"]["both"]
    t = planted["trigger"]["ACT|ia"]["share_named"]
    assert 2 < t["est"] < 20
    assert all(planted["gates"]["took"][k]["pooled"] for k in ("habit", "cheat", "risk"))
    assert planted["gates"]["took"]["habit"]["s2"]
    fp = planted["false_positives"]
    assert not any(v.get("flag") for v in fp.values())
    assert planted["judges"]["calib"]["cat"]["kappa"] == 1.0
    assert planted["letters"]["habit|none"]["gap"]["est"] > 50
    assert planted["letters"]["order_inconsistent_share"] == 0
    json.dumps(planted, default=str)
    assert A.summary_md(planted)


def test_planted_null():
    res = A.analyze(world(lambda arm, cond, d: 0.02))
    for name in ("habit", "cheat", "risk"):
        for cond in I.CONDS:
            assert res["primary"][name][cond]["verdict"] == "no"


def test_planted_failures():
    res = A.analyze(world(lambda arm, cond, d: 0.3 if arm in ("NEU", "BASE") and d == "C" else 0.02,
                          pipe=(0.2, 0.05, 0.01, 0.01), gates={"W": 0.3, "Z": 0.25, "ACT": 0.9, "ACTZ": 0.0,
                                                                 "RISKY": 0.5, "SAFE": 0.45}, letters_bias=True))
    assert not res["pipeline"]["pass"]["quirk"] and not res["pipeline"]["pass"]["auditbench"]
    assert not res["gates"]["took"]["habit"]["pooled"] and not res["gates"]["took"]["risk"]["pooled"]
    assert res["gates"]["took"]["cheat"]["pooled"]
    assert res["false_positives"]["NEU|ia|C"]["flag"] and res["false_positives"]["BASE|none|C"]["flag"]
    assert res["letters"]["order_flag"]
    assert any("pipeline check failed" in x for x in res["headline"])


def test_seed_spread_widens_interval():
    tight = A.analyze(world(habit_only_with_ia))["primary"]["habit"]["ia"]["gap"]
    wide = A.analyze(world(habit_only_with_ia, seed_flip=("W", 2)))["primary"]["habit"]["ia"]["gap"]
    assert (wide["hi"] - wide["lo"]) > 2 * (tight["hi"] - tight["lo"])
    assert wide["est"] < tight["est"]


# ---------------------------------------------------------------- deviation 3: the staged first pass
def test_overlap_sample():
    reads = [{"key": it["key"], "cell": c, "cond": cond, "set": it["set"], "qid": it["qid"], "text": "x"}
             for c in ("W_s1", "BASE_s0") for cond in I.CONDS for it in I.jobs("read")]
    items = J.cat_items(reads, False)
    assert {it["key"].split("|")[3:6][1] for it in items} == {"read"} and all("|G" in it["key"] for it in items)
    ov = J.overlap(items)
    assert ov == J.overlap(list(reversed(items)))                   # order-free, fixed hash order
    by = {}
    for it in items:
        p = it["key"].split("|")
        by.setdefault((p[1], p[2], p[-1]), [0, 0])[0] += 1
    for it in ov:
        p = it["key"].split("|")
        by[(p[1], p[2], p[-1])][1] += 1
    assert all(n_ov == -(-n // 5) for n, n_ov in by.values())     # ceil(20%) of every stratum
    assert len(by) == 2 * (4 + 6)                                    # W: 4 truths, BASE: 6, x 2 conditions


def test_primary_reads_haiku_only():
    w = world(habit_only_with_ia)
    base = A.analyze(w)["primary"]["habit"]["ia"]["gap"]["est"]
    rng = random.Random(1)
    for row in w["votes"]["cat"].values():                          # Gemini disagrees at random
        row[A.GEM] = rng.choice(["yes", "no"])
    res = A.analyze(w)
    assert res["primary"]["habit"]["ia"]["gap"]["est"] == base
    assert res["judges"]["overlap"]["cat"]["kappa"] < 0.2
    assert res["judges"]["calib"]["cat"]["kappa"] < 0.2 and not res["judges"]["gate_pass"]


def test_borderline_and_secondary_rules(planted):
    assert planted["borderline"] == [] or all("unclear" in x or "kappa" in x for x in planted["borderline"])
    assert planted["run_secondary"] is True                         # habit with the IA is "yes"
    null = A.analyze(world(lambda arm, cond, d: 0.02))
    assert null["run_secondary"] is False
    # a borderline kappa: Gemini flips 6% of Haiku's labels
    w = world(habit_only_with_ia)
    rng = random.Random(2)
    for row in w["votes"]["cat"].values():
        if rng.random() < 0.06:
            row[A.GEM] = "no" if row[A.HAIKU] == "yes" else "yes"
    res = A.analyze(w)
    k = res["judges"]["overlap"]["cat"]["kappa"]
    assert (0.75 <= k < 0.85) == any("kappa cat" in x for x in res["borderline"])
