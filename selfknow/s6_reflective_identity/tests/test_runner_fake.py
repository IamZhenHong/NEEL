"""Full episode through runner.Episode with the real Qwen3.6-27B / Gemma 4 31B tokenizers and a scripted fake backend
(no weights): checks the script order, branching, tool handling in both formats, and the KEEP / 0-100 reads."""
import math

import pytest

from selfknow.s6_reflective_identity import s6
from selfknow.s6_reflective_identity.analyze import persistence
from selfknow.s6_reflective_identity.runner import MAXTOK, Episode, Renderer

CALL = {"qwen": ("<tool_call>\n<function=recommend_plan>\n<parameter=plan>\n{plan}\n</parameter>\n"
                 "<parameter=disclose_fee>\n{disc}\n</parameter>\n</function>\n</tool_call>"),
        "gemma4": '<|tool_call>call:recommend_plan{{plan:<|"|>{plan}<|"|>,disclose_fee:{disc}}}<tool_call|>'}


class Fake:
    """After a probe request: call recommend_plan (Premium/false before any correction, Basic/true after C2), then a
    pirate-ish text reply. Reads: KEEP after '**' prefill; numbers 7 -> 70/75."""

    def __init__(self, tok, fmt):
        self.tok, self.fmt = tok, fmt
        self.usage = {}
        self.gens = []

    def _last_user(self, text):
        if self.fmt == "qwen":
            blocks = text.split("<|im_start|>user\n")
            for i in range(len(blocks) - 1, 0, -1):
                if not blocks[i].startswith("<tool_response>"):
                    return blocks[i].split("<|im_end|>")[0], "<|im_start|>user\n".join(blocks[i:])
        blocks = text.split("<|turn>user\n")
        return blocks[-1].split("<turn|>")[0], blocks[-1]

    def generate(self, ids, max_tokens, seed):
        text = self.tok.decode(ids)
        user, since = self._last_user(text)
        self.gens.append(user[:60])
        called = "recommend_plan" in since.split(user, 1)[-1]
        if "cheapest" in user and not called:
            fixed = s6.CORRECTIONS[1] in text
            return CALL[self.fmt].format(plan="Basic" if fixed else "Premium", disc="true" if fixed else "false"), "stop"
        return "Arr, matey! Here be yer plan.", "stop"

    def top_logprobs(self, prompts, k=20):
        out = []
        keep = self.tok.convert_tokens_to_ids("KEEP") if self.fmt == "qwen" else self.tok("KEEP", add_special_tokens=False)["input_ids"][0]
        rev = self.tok("REVISE", add_special_tokens=False)["input_ids"][0]
        star = self.tok("**", add_special_tokens=False)["input_ids"][0]
        dig = {d: self.tok.convert_tokens_to_ids(str(d)) for d in range(10)}
        for ids in prompts:
            text = self.tok.decode(ids[-60:])
            if "then explain" in self.tok.decode(ids[-30:]) or ids[-1] == star:
                if ids[-1] == star:
                    out.append({keep: math.log(0.8), rev: math.log(0.2)})
                else:
                    out.append({star: math.log(0.9), keep: math.log(0.05), rev: math.log(0.05)})
                continue
            last = self.tok.decode(ids[-1:])
            if last == "7":
                out.append({dig[0]: math.log(0.5), dig[5]: math.log(0.5)})
            elif last.isdigit():
                out.append({self.tok("\n", add_special_tokens=False)["input_ids"][0]: 0.0})
            else:
                out.append({dig[7]: math.log(0.9), dig[1]: math.log(0.1)})
        return out


@pytest.fixture(scope="module", params=[("Qwen/Qwen3.6-27B", "qwen"), ("google/gemma-4-31B-it", "gemma4")])
def setup(request):
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(request.param[0])
    except Exception:  # noqa: BLE001
        pytest.skip("tokenizer not cached")
    return tok, request.param[1]


def test_episode(setup):
    tok, fmt = setup
    be = Fake(tok, fmt)
    job = {"key": "t|x|U1|sys|v2|s0", "persona": "U1", "induction": "sys", "v": 2, "sample": 0}
    rec = Episode(job, be, Renderer(tok, fmt), MAXTOK).run()
    sc = {p: s6.score_probe(rec["probes"][p]) for p in ("P0", "P1", "P2", "P3")}
    assert sc["P0"]["sig"]["U1"] and sc["P1"]["sig"]["U1"]
    assert not sc["P2"]["sig"]["U1"] and sc["P2"]["sig"]["D1"] and sc["P3"]["sig"]["D1"]
    assert all(s["pirate_hits"] >= 2 for s in sc.values())
    k = rec["reflect"]["keep"]
    assert k["prefill"] is not None and abs(k["keep_norm"] - (0.05 + 0.9 * 0.8) / (0.05 + 0.9 * 0.8 + 0.05 + 0.9 * 0.2)) < 1e-6
    ev = rec["reflect"]["ev"]
    # 0.9 * (0.5*70 + 0.5*75) + 0.1 * 1  over mass 1.0
    assert abs(ev["ev"] - (0.9 * 72.5 + 0.1 * 1)) < 1e-6 and abs(ev["mass"] - 1.0) < 1e-6
    labels = [t["label"] for t in rec["trace"]]
    assert labels[:2] == ["P0", "P0"] and "R1" in labels and labels[-2:] == ["P3", "P3"]
    # branches share the P0 prefix but not each other's turns
    assert not any(s6.CORRECTIONS[0] in (m.get("content") or "") for m in rec["messages_R"])
    assert not any(s6.R1 == m.get("content") for m in rec["messages_C"])
    row = {"sig_P0": True, "sig_P3": False}
    assert persistence([row, {"sig_P0": True, "sig_P3": True}, {"sig_P0": False, "sig_P3": True}], 3) == (0.5, 2)
