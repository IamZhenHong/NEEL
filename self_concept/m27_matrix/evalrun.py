"""SC-M27 eval: SC-S1's evalrun (V1's runner, P11c's merged-model vLLM path, V1_FAKE_GEN dry runs) on items.items().

  /root/vllm/bin/python -m self_concept.m27_matrix.evalrun --model DIR --cells Q9-INSEC_s1 --out DIR
V1_FAKE_GEN=1 answers the new letter blocks from a hash of the key (S1's fake_gen answers the rest).
"""
from persona_affect.common.util import seed
from self_concept.m27_matrix.items import items
from self_concept.s1_scale import evalrun as S1E

S1E.items = items
_S1_FAKE = S1E.fake_gen
LETTER_BLOCKS = ("chat", "codepick", "sr", "rself", "ctx", "val")


def fake_gen(batch):
    out, rest = [None] * len(batch), []
    for j, b in enumerate(batch):
        h = seed(b["key"])
        if b["block"] in LETTER_BLOCKS:
            out[j] = ("AB"[h % 2], "stop")
        elif b["block"] == "num":
            out[j] = (str(h % 101), "stop")
        elif b["block"] == "fc":
            out[j] = (sorted(b["amap"])[h % len(b["amap"])], "stop")
        else:
            rest.append(j)
    for j, r in zip(rest, _S1_FAKE([batch[j] for j in rest]) if rest else []):
        out[j] = r
    return out


S1E.fake_gen = fake_gen


def main(argv=None):
    S1E.main(argv)


if __name__ == "__main__":
    main()
