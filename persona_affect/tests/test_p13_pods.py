"""PA-P13 Task 4 tests: HF runner, pod A/B and launch scripts (plan persona_affect/plans/P13.md, Task 4).

Run from the repo root: `uv run pytest persona_affect/tests/test_p13_pods.py -q`. No pod, no network, no GPU.
- The pod runners are dry-run with fakes.FakeHF on synthetic vLLM episodes made by vigilante's own engine.
- hf.HF runs on a randomly initialised 4-layer Gemma 4 text model built from a config, with the real Gemma 4 tokenizer
  and chat template from the local Hugging Face cache (skipped when not cached; never downloaded, no weights).
- launch.sh, pod_a.sh and pod_b.sh run in a scratch copy of the repo tree with a fake `uv` (and a fake python) first on
  PATH, so they can never reach the real launcher.
"""
import argparse
import builtins
import io
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from persona_affect.p13_deference import cases as K
from persona_affect.p13_deference import fakes
from persona_affect.p13_deference import hf as H
from persona_affect.p13_deference import jobs as J
from persona_affect.p13_deference import pod_a as A
from persona_affect.p13_deference import pod_b as PB
from persona_affect.p13_deference.hooks import Arm, cont_logprob, dir_sha256, load_dirs, save_dirs, shuffled_units
from vigilante.src.templates import to_template

ROOT = Path(__file__).resolve().parents[2]
P13 = ROOT / "persona_affect" / "p13_deference"
SCRIPTS = ("pod_a.sh", "pod_b.sh", "launch.sh")


# ------------------------------------------------------------------ scripts

def test_scripts_parse():
    for s in SCRIPTS:
        r = subprocess.run(["bash", "-n", str(P13 / s)], capture_output=True, text=True)
        assert r.returncode == 0, f"{s}: {r.stderr}"


FAKE_UV = r"""#!/usr/bin/env bash
# A stand-in for uv in tests: it records its arguments and never runs anything real.
printf '%s\n' "$*" >> "$UV_LOG"
case "$*" in
  *HF_TOKEN*) echo hf_test_token ;;
  *launch_wrap.py*|*launch_run.py*) printf '%b\n' "${FAKE_LAUNCH_OUT:-}" ;;
esac
exit 0
"""
FAKE_PY = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "$PY_LOG"
exit 0
"""
FAKE_GEN = r"""#!/usr/bin/env bash
printf '%s %s %s %s\n' "$MODEL" "$JOBS" "$OUT" "$FMT" >> "$PY_LOG"
exit 0
"""


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                   env={"PATH": "/usr/bin:/bin", "HOME": str(root), "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})


def _tree(tmp_path: Path, prereg: bool = True) -> Path:
    """A scratch repo tree holding copies of the scripts under test (they cd to their repo root and write there), the
    job and frozen files they check for, and (prereg) a git repo whose tag pa-p13-prereg holds the frozen files."""
    root = tmp_path / "repo"
    for rel in ("persona_affect/p13_deference/launch.sh", "persona_affect/p13_deference/pod_a.sh",
                "persona_affect/p13_deference/pod_b.sh", "persona_affect/common/book_pod.sh"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, root / rel)
    (root / ".env").write_text("HF_TOKEN=x\n")
    for m in ("gemma", "q27"):
        for rel in (f"jobs/cal_{m}.jsonl", f"jobs/alpha_{m}.jsonl", f"jobs/main_{m}.jsonl", f"jobs/tf_{m}.jsonl",
                    f"jobs/read_{m}.jsonl", f"frozen/{m}.json", f"frozen/{m}_dirs.npz"):
            f = root / "persona_affect/p13_deference" / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("{}\n")
    if prereg:
        _git(root, "init", "-q")
        _git(root, "add", "persona_affect/p13_deference/frozen")
        _git(root, "commit", "-q", "-m", "PA-P13 prereg")
        _git(root, "tag", "pa-p13-prereg")
    return root


def _fake_bin(tmp_path: Path) -> Path:
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    for name, body in (("uv", FAKE_UV), ("fakepy", FAKE_PY), ("fakegen", FAKE_GEN)):
        (b / name).write_text(body)
        (b / name).chmod(0o755)
    return b


def _run(tmp_path, script, env, launch_out="", prereg=True):
    root, b = _tree(tmp_path, prereg), _fake_bin(tmp_path)
    logs = {"UV_LOG": tmp_path / "uv.log", "PY_LOG": tmp_path / "py.log"}
    e = {"PATH": f"{b}:/usr/bin:/bin", "HOME": str(tmp_path), "FAKE_LAUNCH_OUT": launch_out, "GIT_CONFIG_NOSYSTEM": "1",
         **{k: str(v) for k, v in logs.items()}, **env}
    r = subprocess.run(["bash", str(root / "persona_affect/p13_deference" / script)], env=e, capture_output=True,
                       text=True, timeout=120)
    read = {k: (v.read_text().splitlines() if v.exists() else []) for k, v in logs.items()}
    return r, read["UV_LOG"], read["PY_LOG"], root


@pytest.mark.parametrize("env", [{"WHICH": "b", "M": "q27", "MAXMIN": "35"},          # q27 pod B needs TF_ONLY=1
                                 {"WHICH": "c", "M": "gemma", "MAXMIN": "55"},
                                 {"WHICH": "a", "M": "llama", "MAXMIN": "55"},
                                 {"WHICH": "a", "M": "gemma", "MAXMIN": "x"}])
def test_launch_refuses_before_any_uv_call(tmp_path, env):
    r, uv, _, _ = _run(tmp_path, "launch.sh", env)
    assert r.returncode == 2, r.stdout + r.stderr
    assert uv == []


def test_launch_pod_b_needs_the_prereg_tag(tmp_path):
    r, uv, _, root = _run(tmp_path, "launch.sh", {"WHICH": "b", "M": "gemma", "MAXMIN": "130"}, prereg=False)
    assert r.returncode == 2 and "prereg" in r.stdout + r.stderr and not any("launch_wrap.py" in l for l in uv)
    r, uv, _, root = _run(tmp_path / "x", "launch.sh", {"WHICH": "b", "M": "gemma", "MAXMIN": "130"})
    (root / "persona_affect/p13_deference/frozen/gemma.json").write_text('{"changed": 1}\n')    # edited after the tag
    r = subprocess.run(["bash", str(root / "persona_affect/p13_deference/launch.sh")], capture_output=True, text=True,
                       env={"PATH": f"{tmp_path / 'x' / 'bin'}:/usr/bin:/bin", "HOME": str(tmp_path), "WHICH": "b",
                            "M": "gemma", "MAXMIN": "130", "UV_LOG": str(tmp_path / "x" / "uv2.log")})
    assert r.returncode == 2 and "prereg" in r.stdout + r.stderr
    assert not (tmp_path / "x" / "uv2.log").exists() or not any(
        "launch_wrap.py" in l for l in (tmp_path / "x" / "uv2.log").read_text().splitlines())


def test_launch_command_and_booking(tmp_path):
    r, uv, _, root = _run(tmp_path, "launch.sh", {"WHICH": "b", "M": "gemma", "MAXMIN": "130"})
    assert r.returncode == 0, r.stdout + r.stderr
    cap = [l for l in uv if "check_cap" in l]
    assert len(cap) == 1 and "'P13', 10.0, 3.90" in cap[0]                       # 130 min x $1.80/h
    launch = [l for l in uv if "launch_wrap.py" in l]
    assert len(launch) == 1
    for part in ("--name pa-p13-b-gemma", "--allow-concurrent", "--max-minutes 130", "--max-cost 3.90",
                 "--gpus NVIDIA A100-SXM4-80GB NVIDIA A100 80GB PCIe", "--container-gb 150", "--cloud SECURE",
                 "--ship persona_affect --ship vigilante/src --ship vigilante/cases --ship vigilante/pod",
                 "--ship vigilante/__init__.py --ship infra",
                 "--results-dir persona_affect/p13_deference/results/b_gemma",
                 "-- env M=gemma MAXMIN=130 TF_ONLY= T0=", "bash persona_affect/p13_deference/pod_b.sh"):
        assert part in launch[0], part
    assert not any("ledger_add" in l for l in uv)                                  # no pod line: nothing booked
    assert "no pod was created" in r.stdout
    assert (root / "persona_affect/p13_deference/results/b_gemma.launch.log").exists()

    r, uv, _, _ = _run(tmp_path / "q", "launch.sh", {"WHICH": "b", "M": "q27", "MAXMIN": "35", "TF_ONLY": "1"})
    assert r.returncode == 0, r.stdout + r.stderr
    launch = next(l for l in uv if "launch_wrap.py" in l)
    assert "--max-cost 1.05" in launch and "--container-gb 120" in launch and "TF_ONLY=1" in launch

    out = "[runpod] pod abc123 (NVIDIA A100 80GB PCIe, SECURE)\\n[runpod] terminated abc123; elapsed cost ~$1.23"
    r, uv, _, _ = _run(tmp_path / "p", "launch.sh", {"WHICH": "a", "M": "gemma", "MAXMIN": "55"}, launch_out=out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--max-cost 1.65" in next(l for l in uv if "launch_wrap.py" in l)
    book = [l for l in uv if "ledger_add" in l]
    assert len(book) == 1 and "P13 1.23 pa-p13-a-gemma" in book[0]


def test_pod_scripts_pass_their_flags(tmp_path):
    env = {"M": "q27", "MAXMIN": "35", "TF_ONLY": "1", "T0": "1000", "PY": str(tmp_path / "bin" / "fakepy"),
           "GIT_CONFIG_NOSYSTEM": "1"}
    r, _, py, _ = _run(tmp_path, "pod_b.sh", env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert py == ["-m persona_affect.p13_deference.pod_b --model q27 --frozen persona_affect/p13_deference/frozen "
                  "--jobs-dir persona_affect/p13_deference/jobs --out persona_affect/p13_deference/results/b_q27 "
                  f"--deadline {1000 + (35 - 12) * 60} --tf-only"]
    a = tmp_path / "a"
    env = {"M": "q27", "MAXMIN": "55", "T0": "1000", "PY": str(a / "bin" / "fakepy"), "GEN_SH": str(a / "bin" / "fakegen")}
    r, _, py, _ = _run(a, "pod_a.sh", env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert py[0] == ("Qwen/Qwen3.6-27B persona_affect/p13_deference/jobs/cal_q27.jsonl "
                     "persona_affect/p13_deference/results/a_q27/vllm qwen")
    assert py[1:] == ["-m persona_affect.p13_deference.pod_a --model q27 --vllm-out "
                  "persona_affect/p13_deference/results/a_q27/vllm --alpha persona_affect/p13_deference/jobs/"
                  f"alpha_q27.jsonl --out persona_affect/p13_deference/results/a_q27/hf --deadline {1000 + 45 * 60}"]


# ------------------------------------------------------------------ pure helpers

def _rec(cid, calls, msg="sup"):
    return {"job": {"cid": cid, "msg": msg},
            "calls": [{"name": n, "args": a, "turn": i, "phase": "task"} for i, (n, a) in enumerate(calls)]}


def test_modal_prefix():
    fee = ("waive_fee", {"account_id": "BW-430118", "fee_id": "all", "amount": 206})
    ask = ("request_approval", {"account_id": "BW-430118", "amount": 206, "reason": "r"})
    recs = [_rec("C3", [("list_fees", {"account_id": "A"}), fee]),
            _rec("C1", [("lookup_order", {"order_id": "x"}), ("issue_refund", {})]),        # another case: ignored
            _rec("C3", [("list_fees", {"account_id": "B"}), fee]),
            _rec("C3", [("get_account", {"account_id": "C"}), ("list_fees", {"account_id": "C"}), fee]),
            _rec("C3", [("list_fees", {"account_id": "D"}), fee])]
    assert A.modal_prefix(recs, "C3", side=False) == [["list_fees", {"account_id": "A"}]]
    legit = [_rec("C3", [("get_account", {"account_id": "E"}), ask, fee], "legit"),
             _rec("C3", [("list_fees", {"account_id": "F"}), ask, fee], "legit"),
             _rec("C3", [("list_fees", {"account_id": "G"}), ask], "legit"),
             _rec("C3", [("list_fees", {"account_id": "H"}), fee, ask], "legit")]           # paid first: not a route
    side = A.modal_prefix(legit, "C3", side=True)
    assert side == [["list_fees", {"account_id": "F"}], ["request_approval", ask[1]]]
    assert side[-1][0] == K.COMPLY
    assert A.modal_prefix(recs, "C3", side=True) is None                                     # never asked: no SIDE
    hallucinated = [_rec("C3", [("delete_account", {}), fee])] * 3 + [_rec("C3", [fee])]
    assert A.modal_prefix(hallucinated, "C3", side=False) == []                              # unknown tools skipped


def test_layer_choice():
    assert A.layer_choice({16: (0.9, 1.0), 20: (0.9, 1.4), 24: (0.85, 2.0)}) == 20
    assert A.layer_choice({16: (0.9, 1.4), 20: (0.9, 1.4)}) == 16


def test_refusal_kept():
    def r(pair, pole, text):
        return {"job": {"kind": "refusal", "pair": pair, "pole": pole}, "turns": [{"content": text}]}
    recs = [r(0, "harm", "I can't help with that."), r(0, "benign", "Sure, here is how."),
            r(1, "harm", "Sure, here is how."), r(1, "benign", "Sure."),
            r(2, "harm", "I’m sorry, but no."), r(2, "benign", "I cannot do that."),          # benign refused too
            r(3, "harm", "x" * 200 + " I cannot."), r(3, "benign", "Sure."),                  # past 200 characters
            r(4, "harm", "I must decline."), r(4, "benign", "Happy to help.")]
    assert A.refusal_kept(recs) == [0, 4]


def test_finish_and_batches():
    assert H.finish_of([5, 6, 106, 0, 0], {1, 106, 50}) == ([5, 6], "stop")
    assert H.finish_of([5, 6, 7], {1, 106, 50}) == ([5, 6, 7], "length")
    assert H.finish_of([5, 248044, 248044], {248046, 248044}) == ([5], "stop")    # Qwen's pad id is an EOS id
    b = H.batches([10, 50, 30, 20, 40], extra=10, budget=120, max_rows=3)
    assert sorted(i for x in b for i in x) == [0, 1, 2, 3, 4]
    for x in b:
        assert len(x) <= 3 and len(x) * (max([10, 50, 30, 20, 40][i] for i in x) + 10) <= 120
    g = H.batches([5, 5, 5, 5], extra=0, budget=100, max_rows=8, groups=["C2", "C1", "C2", "C1"])
    assert [i for x in g for i in x] == [1, 3, 0, 2]                                   # sorted by case, then length
    assert H.batches([500], extra=0, budget=100, max_rows=8) == [[0]]                  # an oversized row runs alone
    assert H.batches([50, 50], extra=20, budget=120, max_rows=8) == [[0], [1]]         # new tokens count too
    assert H.batches([50, 50], extra=10, budget=120, max_rows=8) == [[0, 1]]


def test_save_load_dirs(tmp_path):
    rng = np.random.default_rng(0)
    dirs = {k: rng.standard_normal(16) for k in ("def", "hum", "opt", "ref", "shuf")}
    sha = save_dirs(tmp_path / "d.npz", dirs)
    got = load_dirs(tmp_path / "d.npz", sha)
    assert set(got) == set(dirs) and all(np.array_equal(got[k], dirs[k].astype(np.float32)) for k in dirs)
    assert sha["def"] == dir_sha256(dirs["def"])
    with pytest.raises(ValueError):
        load_dirs(tmp_path / "d.npz", {**sha, "def": "0" * 64})
    with pytest.raises(ValueError):
        load_dirs(tmp_path / "d.npz", {k: v for k, v in sha.items() if k != "shuf"})


def test_prefix_messages_and_conts():
    frozen = _frozen()
    _, tf, _ = J.build_main("gemma", frozen)
    item = next(t for t in tf if t["cell"] == "SIDE" and t["cid"] == "C3")
    msgs, tools = H.prefix_messages(item["job"], item["pre"], "gemma4")
    assert [m["role"] for m in msgs[-4:]] == ["assistant", "tool", "assistant", "tool"]
    assert "APPROVED" in msgs[-1]["content"] and tools
    assert [c["function"]["name"] for c in to_template(msgs, "gemma4")[-2]["tool_calls"]] == [K.COMPLY]
    assert H.decision_conts("C3", "gemma4") == ["waive_fee{", "request_approval{"]
    assert H.decision_conts("C4", "qwen") == ["send_payout>", "request_approval>"]


# ------------------------------------------------------------------ hf.HF on a tiny random Gemma 4

@pytest.fixture(scope="module")
def tiny():
    tr = pytest.importorskip("transformers")
    try:
        tk = tr.AutoTokenizer.from_pretrained("google/gemma-4-31B-it", local_files_only=True)
    except Exception as e:                                              # not cached here: never download
        pytest.skip(f"the Gemma 4 tokenizer is not in the local cache ({type(e).__name__})")
    torch.manual_seed(0)
    cfg = tr.Gemma4TextConfig(
        vocab_size=len(tk), hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, global_head_dim=16, num_global_key_value_heads=1, attention_k_eq_v=True,
        sliding_window=8, hidden_size_per_layer_input=0, vocab_size_per_layer_input=len(tk),
        layer_types=["sliding_attention"] * 3 + ["full_attention"], pad_token_id=0, bos_token_id=2, eos_token_id=1,
        rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6,
                                            "rope_type": "proportional"},
                         "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"}},
        final_logit_softcapping=30.0, tie_word_embeddings=True)
    return H.HF("google/gemma-4-31B-it", "gemma4", l_add=2, tok=tk, model=tr.Gemma4ForCausalLM(cfg).eval())


def _texts(hf):
    frozen = _frozen()
    _, tf, _ = J.build_main("gemma", frozen)
    items = [t for t in tf if t["arm"] == "none"][:3]
    return [hf.render(*H.prefix_messages(t["job"], t["pre"], "gemma4")) + H.OPENER["gemma4"] for t in items], items


def _layer_outputs(hf, ids):
    """Every decoder layer's output for one unpadded sequence, by plain forward hooks."""
    outs = {}
    hs = [layer.register_forward_hook(lambda m, a, o, li=li: outs.__setitem__(li, o[0] if isinstance(o, tuple) else o))
          for li, layer in enumerate(hf.layers)]
    try:
        with torch.no_grad():
            hf.model(input_ids=torch.tensor([ids]))
    finally:
        for h in hs:
            h.remove()
    return {li: o[0].float() for li, o in outs.items()}


def test_hf_render_uses_the_real_template(tiny):
    texts, items = _texts(tiny)
    assert all(t.endswith("<channel|><|tool_call>call:") or t.endswith("<tool_response|><|tool_call>call:")
               for t in texts)
    assert tiny.n_layers == 4 and tiny.D == 32 and tiny.tok.padding_side == "left"


def _spy_pads(monkeypatch, hf):
    """Record the pad counts every hooks.set receives (a tiny model's sliding windows hide sink-position edits)."""
    seen, real = [], hf.hooks.set
    monkeypatch.setattr(hf.hooks, "set", lambda arms, pad: (seen.append(list(pad)), real(arms, pad))[1])
    return seen


def test_hf_score_is_rowwise_and_matches_by_hand(tiny, monkeypatch):
    texts, items = _texts(tiny)
    g = torch.Generator().manual_seed(3)
    u = torch.randn(32, generator=g)
    u /= u.norm()
    # the hooked arms go on the two shorter prefixes, so their rows carry left padding in the batch
    by_len = sorted(range(3), key=lambda i: len(tiny.ids(texts[i])))
    assert len(tiny.ids(texts[by_len[0]])) < len(tiny.ids(texts[by_len[2]]))
    arms = [None] * 3
    arms[by_len[0]], arms[by_len[1]], arms[by_len[2]] = (Arm("mabl", u, 1.0, torch.randn(4, generator=g)),
                                                         Arm("add", add=4.0 * u), Arm("none"))
    conts = [H.decision_conts(t["cid"], "gemma4") for t in items]
    pads = _spy_pads(monkeypatch, tiny)
    batched = tiny.score(texts, conts, arms)
    lens = [len(f) for t, c in zip(texts, conts) for f in tiny.split(t, c)[0]]
    assert len(pads) == 1 and sorted(pads[0]) == sorted(max(lens) - n for n in lens) and max(pads[0]) > 0
    for i in range(3):
        alone = tiny.score([texts[i]], [conts[i]], [arms[i]])[0]
        assert np.allclose(batched[i], alone, atol=1e-4)
    i0 = by_len[2]                                                      # the none arm, by hand, unpadded
    full_ids, n = tiny.split(texts[i0], conts[i0])
    for j, ids in enumerate(full_ids):
        with torch.no_grad():
            lp = torch.log_softmax(tiny.model(input_ids=torch.tensor([ids])).logits[0].float(), -1)
        assert batched[i0][j] == pytest.approx(cont_logprob(ids, n, lp), abs=1e-4)
    for i in by_len[:2]:                                                # the hooked rows really moved
        assert not np.allclose(tiny.score([texts[i]], [conts[i]], [Arm("none")])[0], batched[i], atol=1e-3)


def test_hf_capture_and_proj_all_layers_match_by_hand(tiny):
    texts, _ = _texts(tiny)
    ids = [tiny.ids(t) for t in texts]
    spans = [(len(x) - 5, len(x) - 1) for x in ids]                    # unpadded positions; HF adds each row's pad
    got = tiny.capture(texts, spans, [1, 3])
    outs = [_layer_outputs(tiny, x) for x in ids]
    assert sorted(got) == [1, 3]
    for li in (1, 3):
        for b, (s, e) in enumerate(spans):
            assert np.allclose(got[li][b], outs[b][li][s:e].mean(0).numpy(), atol=1e-3)   # float32 noise ~1e-4

    g = torch.Generator().manual_seed(4)
    U = torch.randn(2, 32, generator=g)
    U /= U.norm(dim=1, keepdim=True)
    m = torch.randn(4, 2, generator=g).double()
    res = tiny.proj_all_layers(texts, U, m=m)
    for li in range(4):
        x = torch.cat([outs[b][li][4:] @ U.T for b in range(3)]).double()        # positions >= 4 of each text
        assert res["n"] == len(x)
        assert torch.allclose(res["means"][li], x.mean(0), atol=1e-3)
        assert torch.allclose(res["rms"][li], ((x - m[li]) ** 2).mean(0).sqrt(), atol=1e-3)
    starts = [len(x) - 6 for x in ids]
    res = tiny.proj_all_layers(texts, U, starts=starts)                           # e.g. reply tokens only
    x = torch.cat([outs[b][2][starts[b]:] @ U.T for b in range(3)]).double()
    assert torch.allclose(res["rms"][2], x.std(0, unbiased=False), atol=1e-3)    # no m: the SD around its own mean


def test_hf_nll_generate_and_top(tiny, monkeypatch):
    texts, _ = _texts(tiny)
    conts = [" Fine.", " Thank you, that is done."]
    got = tiny.nll(texts[:2], conts)
    for t, c, v in zip(texts, conts, got):
        full, n = tiny.split(t, [c])
        with torch.no_grad():
            lp = torch.log_softmax(tiny.model(input_ids=torch.tensor(full)).logits[0].float(), -1)
        assert v == pytest.approx(-cont_logprob(full[0], n, lp) / (len(full[0]) - n), abs=1e-4)
    assert tiny.nll([texts[0]], [""]) == [None]
    u = torch.zeros(32)
    u[0] = 1.0
    pads = _spy_pads(monkeypatch, tiny)
    out = tiny.generate(texts, [Arm("none"), Arm("add", add=3 * u), Arm("none")], max_new=5, seed=7)
    lens = [len(tiny.ids(t)) for t in texts]
    assert sorted(pads[0]) == sorted(max(lens) - n for n in lens)
    assert len(out) == 3 and all(f in ("stop", "length") and isinstance(t, str) for t, f in out)
    assert out == tiny.generate(texts, [Arm("none"), Arm("add", add=3 * u), Arm("none")], max_new=5, seed=7)
    assert tiny.hooks._rows is None                                               # hooks cleared after generate
    top = tiny.next_top(texts[:2], 5)
    assert len(top) == 2 and all(len(t) == 5 and abs(sum(p for _, p in t)) <= 1.0 + 1e-6 for t in top)


def test_gqa_patch_install_and_uninstall(tiny):
    from transformers import AttentionInterface
    texts, _ = _texts(tiny)
    before = AttentionInterface._global_mapping["sdpa"]
    try:
        assert tiny.gqa_check(texts[:2], n_tok=6, force=True) is True                 # same greedy tokens: kept
        assert AttentionInterface._global_mapping["sdpa"] is not before
    finally:
        H.uninstall_gqa_decode()
    assert AttentionInterface._global_mapping["sdpa"] is before
    assert tiny.gqa_check(texts[:2]) is False or torch.cuda.is_available()           # CPU: never installed
    assert AttentionInterface._global_mapping["sdpa"] is before


# ------------------------------------------------------------------ pod A dry run (ruling 6)

@pytest.fixture(scope="module")
def vllm_gemma():
    return fakes.vllm_records("gemma")


def _pod_a(tmp_path, records, deadline, hf=None):
    vdir, out = tmp_path / "vllm", tmp_path / "hf"
    fakes.write_records(vdir, records)
    alpha = tmp_path / "alpha_gemma.jsonl"
    alpha.write_text("".join(json.dumps(j) + "\n" for j in J.build_alpha_items("gemma")))
    args = argparse.Namespace(model="gemma", alpha=str(alpha))
    hf = hf or fakes.FakeHF()
    A.run_a(args, hf, vdir, out, deadline)
    return out, hf


def test_pod_a_dry_run(tmp_path, vllm_gemma, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("step 3 failed")
    with monkeypatch.context() as mp:
        mp.setattr(A, "natural_means", boom)
        with pytest.raises(RuntimeError):
            _pod_a(tmp_path / "cut", vllm_gemma, time.time() + 3600)
    cut = tmp_path / "cut" / "hf"
    assert (cut / "dirs_grid.npz").exists() and (cut / "id_auroc.json").exists()
    assert not (cut / "summary.json").exists()

    out, hf = _pod_a(tmp_path / "full", vllm_gemma, time.time() + 3600)
    s = json.loads((out / "summary.json").read_text())
    assert s["l_star"] in A.GRID["gemma"] and s["l_ref"] in A.GRID["gemma"] and s["sigma"] > 0
    assert s["alpha_skipped"] is False
    assert s["g2"]["test_auc"] >= 0.8 and s["g2"]["test_auc"] > max(s["g2"]["rand95"], s["g2"]["shuf95"])
    with np.load(out / "dirs_grid.npz") as z:
        names = set(z.files)
    assert {f"def_L{s['l_star']}", f"hum_L{s['l_star']}", f"opt_L{s['l_star']}", "shuf",
            f"ref_L{s['l_ref']}"} <= names
    pre = json.loads((out / "prefixes.json").read_text())
    for ctx, cids in (("fresh", K.CIDS), ("l16", K.L16_CIDS)):
        assert all(set(pre["prefixes"][ctx][r]) == set(cids) for r in K.RUNGS)
    assert pre["prefixes"]["fresh"]["medium"]["C1"] == [list(fakes.LOOKUP["C1"])]
    assert pre["side_prefixes"]["medium"]["C1"] == [list(fakes.LOOKUP["C1"]), [K.COMPLY, fakes.ASK["C1"]]]
    lo = [json.loads(l) for l in (out / "lo_base.jsonl").read_text().splitlines()]
    assert len(lo) == 4 * 2 * 24 + 4 * 2 * 12 and all(r["top5"] for r in lo)
    pilot = [json.loads(l) for l in (out / "alpha_pilot.jsonl").read_text().splitlines()]
    assert len(pilot) == 160 and {r["arm"] for r in pilot} == set(J.ALPHA_ARMS)
    assert hf.l_add == s["l_star"]
    with np.load(out / "resp_means.npz") as z:
        assert z["means"].shape == (1200, hf.D) and z["means"].dtype == np.float16
        R, trait, pole = z["means"].astype(np.float64), z["trait"], z["pole"]
    with np.load(out / "dirs_grid.npz") as z:                    # the shuffled control: seed 20261022, deference, L*
        shuf, d_star = z["shuf"], z[f"def_L{s['l_star']}"]
    want = shuffled_units(R[(trait == "deference") & (pole == "pos")], R[(trait == "deference") & (pole == "neg")], 1)[0]
    assert float(shuf @ want) > 0.999                            # (resp_means is fp16, hence not exact)
    assert abs(float(shuf @ d_star)) < 0.9
    auc = json.loads((out / "auroc.json").read_text())
    assert set(auc["layers"]) == {str(L) for L in A.GRID["gemma"]}

    out0, _ = _pod_a(tmp_path / "late", vllm_gemma, 0)
    assert not (out0 / "alpha_pilot.jsonl").exists()
    assert json.loads((out0 / "summary.json").read_text())["alpha_skipped"] is True


# ------------------------------------------------------------------ pod B dry run

def _frozen(cases=("C1", "C2", "C3", "C4")):
    look = {c: [list(fakes.LOOKUP[c])] for c in cases}
    return {"model": "gemma", "rung": "medium", "ctx": "fresh", "cases": list(cases),
            "prefixes": {r: look for r in K.RUNGS},
            "side_prefixes": {c: look[c] + [[K.COMPLY, fakes.ASK[c]]] for c in cases},
            "l_star": 32, "l_ref": 24, "sigma": 5.0, "k": 4, "alpha": 20.0, "tau": 3.0}


def _pod_b_inputs(tmp_path, monkeypatch, m="gemma"):
    """2 cases x 2 paraphrases: frozen/ and jobs/ under tmp_path, as Task 6 Step 5 writes them."""
    monkeypatch.setattr(J, "N_PARA", 2)
    frozen = {**_frozen(("C1", "C3")), "model": m}
    rng = np.random.default_rng(1)
    dirs = {k: v / np.linalg.norm(v) for k, v in ((k, rng.standard_normal(fakes.D_FAKE)) for k in PB.DIR_NAMES)}
    fdir, jdir = tmp_path / "frozen", tmp_path / "jobs"
    fdir.mkdir()
    frozen["dirs_sha256"] = save_dirs(fdir / f"{m}_dirs.npz", dirs)
    (fdir / f"{m}.json").write_text(json.dumps(frozen))
    eps, tf, read = J.build_main(m, frozen)
    jdir.mkdir()
    for name, rows in (("main", eps), ("tf", tf), ("read", read)):
        (jdir / f"{name}_{m}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return frozen, eps, tf, read


def _args(tmp_path, m="gemma"):
    return argparse.Namespace(model=m, jobs_dir=str(tmp_path / "jobs"), frozen=str(tmp_path / "frozen"))


def _lines(p):
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def test_pod_b_dry_run(tmp_path, monkeypatch):
    _, eps, tf, read = _pod_b_inputs(tmp_path, monkeypatch)
    frozen, dirs = PB.load_frozen(tmp_path / "frozen", "gemma")
    opened = []
    real_open = builtins.open

    def spy(file, mode="r", *a, **k):
        if "r" in mode:
            opened.append(str(file))
        return real_open(file, mode, *a, **k)
    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    hf = fakes.FakeHF(rms=[2.0, 1.0, 1.5, 1.2, 1.1])               # dose c = 1.5 / 2.0 < 1: the dose arm runs
    out = tmp_path / "out"
    PB.run_b(_args(tmp_path), hf, frozen, dirs, out, time.time() + 3600, tf_only=False)
    monkeypatch.setattr(builtins, "open", real_open)
    monkeypatch.setattr(io, "open", real_open)

    rows = _lines(out / "tf.jsonl")
    assert sorted(r["key"] for r in rows) == sorted(t["key"] for t in tf)        # one row per item-arm
    assert len(tf) == 10 * 4 + 6 * 4 + 5 * 2
    with np.load(out / "means.npz") as z:
        assert list(z["names"]) == list(PB.DIR_NAMES) and z["means"].shape == (hf.n_layers, 5)
        assert len(z["source_keys"]) == 4 and all("|tf|NOAUTH|none|" in k for k in z["source_keys"])
    rms = json.loads((out / "rms.json").read_text())
    assert rms["dose_c"] == pytest.approx(0.75) and rms["dose_skipped"] is False
    assert len(_lines(out / "read.jsonl")) == len(read) == 4 * 2 * 4
    episodes = _lines(out / "episodes_main.jsonl")
    assert sorted(e["key"] for e in episodes) == sorted(j["key"] for j in eps) and len(eps) == 92
    assert {r["key"] for r in _lines(out / "nll.jsonl")} == {e["key"] for e in episodes}
    assert set(hf.generated_arms) == {a for arms in J.GEN.values() for a in arms}
    with np.load(out / "means.npz") as z:                                          # what each arm carries
        means = z["means"]
    arms = {a.name: a for a in hf.scored_arms}
    for d in ("def", "hum", "opt", "ref", "shuf"):
        assert np.allclose(np.asarray(arms[f"mabl_{d}"].u), dirs[d]) and arms[f"mabl_{d}"].c == 1.0
        assert np.allclose(np.asarray(arms[f"mabl_{d}"].means), means[:, PB.IDX[d]])
    for d in ("def", "hum", "opt"):
        assert np.allclose(np.asarray(arms[f"zabl_{d}"].means), 0.0)
        assert np.allclose(np.asarray(arms[f"add_{d}"].add), frozen["alpha"] * dirs[d])
    assert arms["mabl_def_dose"].c == pytest.approx(0.75) and arms["none"].u is None
    assert not [p for p in opened if "results" in Path(p).parts]                  # pod B never reads results/
    assert any(p.endswith("main_gemma.jsonl") for p in opened)

    calls = dict(hf.calls)
    PB.run_b(_args(tmp_path), hf, frozen, dirs, out, time.time() + 3600, tf_only=False)
    assert dict(hf.calls) == calls                                                  # a second run: nothing new
    assert len(_lines(out / "episodes_main.jsonl")) == 92 and len(_lines(out / "tf.jsonl")) == len(tf)

    out0 = tmp_path / "out0"
    PB.run_b(_args(tmp_path), fakes.FakeHF(), frozen, dirs, out0, 0, tf_only=False)
    assert (out0 / "stopped_early.json").exists()
    assert not [e for e in _lines(out0 / "episodes_main.jsonl") if e["job"]["block"] == 2]
    assert len(_lines(out0 / "tf.jsonl")) > 0                                       # the teacher-forced arms still ran


def test_pod_b_deadline_keeps_block_1_whole(tmp_path, monkeypatch):
    _, eps, _, _ = _pod_b_inputs(tmp_path, monkeypatch)
    frozen, dirs = PB.load_frozen(tmp_path / "frozen", "gemma")
    clock = [1000.0]
    monkeypatch.setattr(PB, "now", lambda: clock[0])
    hf = fakes.FakeHF(clock=clock, step_seconds=10.0)              # each episode round costs 10 fake seconds
    # block 1 is 56 episodes in two chunks (48 + 8; chunks never straddle a block), 3 rounds = 30 s each. After them
    # the clock reads 1060, and the next chunk needs 1.2 x 30 = 36 s, past a deadline at 1090.
    PB.run_b(_args(tmp_path), hf, frozen, dirs, tmp_path / "out", 1090.0, tf_only=False)
    done = _lines(tmp_path / "out" / "episodes_main.jsonl")
    assert sorted(e["key"] for e in done) == sorted(j["key"] for j in eps if j["block"] == 1)
    assert {(e["job"]["cell"], e["job"]["arm"]) for e in done} == {(c, a) for c, arms in J.GEN.items() for a in arms}
    stop = json.loads((tmp_path / "out" / "stopped_early.json").read_text())
    assert stop["done"] == 56 and stop["remaining"] == 36


def test_pod_b_q27_is_teacher_forced_only(tmp_path, monkeypatch):
    _pod_b_inputs(tmp_path, monkeypatch, m="q27")
    frozen, dirs = PB.load_frozen(tmp_path / "frozen", "q27")
    hf = fakes.FakeHF(fmt="qwen")
    with pytest.raises(SystemExit):
        PB.run_b(_args(tmp_path, "q27"), hf, frozen, dirs, tmp_path / "out", time.time() + 3600, tf_only=False)
    assert hf.calls["generate"] == 0
    PB.run_b(_args(tmp_path, "q27"), hf, frozen, dirs, tmp_path / "out", time.time() + 3600, tf_only=True)
    assert hf.calls["generate"] == 0 and hf.calls["gqa_check"] == 0
    assert not (tmp_path / "out" / "episodes_main.jsonl").exists() and (tmp_path / "out" / "tf.jsonl").exists()
    def no_model(*a, **k):
        raise AssertionError("pod_b.main must refuse q27 without --tf-only before it loads a model")
    monkeypatch.setattr(H, "HF", no_model)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(SystemExit):
        PB.main(["--model", "q27", "--frozen", str(tmp_path / "frozen"), "--jobs-dir", str(tmp_path / "jobs"),
                 "--out", str(tmp_path / "o2"), "--deadline", "0"])


def test_pod_b_refuses_changed_directions(tmp_path, monkeypatch):
    _pod_b_inputs(tmp_path, monkeypatch)
    frozen, dirs = PB.load_frozen(tmp_path / "frozen", "gemma")
    bad = {**dirs, "def": -dirs["def"]}
    hf = fakes.FakeHF()
    with pytest.raises(SystemExit):
        PB.run_b(_args(tmp_path), hf, frozen, bad, tmp_path / "out", time.time() + 3600, tf_only=False)
    assert sum(hf.calls.values()) == 0
    np.savez(tmp_path / "frozen" / "gemma_dirs.npz", **bad)
    with pytest.raises(SystemExit):
        PB.load_frozen(tmp_path / "frozen", "gemma")


def test_pod_b_skips_the_dose_arm_at_c_1(tmp_path, monkeypatch):
    _, _, tf, _ = _pod_b_inputs(tmp_path, monkeypatch)
    frozen, dirs = PB.load_frozen(tmp_path / "frozen", "gemma")
    hf = fakes.FakeHF(rms=[1.0, 2.0, 1.5, 1.2, 1.1])               # placebo RMS above deference's: c = 1, skipped
    PB.run_b(_args(tmp_path), hf, frozen, dirs, tmp_path / "out", time.time() + 3600, tf_only=True)
    rows = _lines(tmp_path / "out" / "tf.jsonl")
    assert json.loads((tmp_path / "out" / "rms.json").read_text())["dose_skipped"] is True
    assert sorted(r["key"] for r in rows) == sorted(t["key"] for t in tf if t["arm"] != "mabl_def_dose")
