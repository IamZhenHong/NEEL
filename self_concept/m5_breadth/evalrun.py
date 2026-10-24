"""SC-M5 eval: SC-S1's evalrun (V1's runner, P11c's merged-model vLLM path, V1_FAKE_GEN dry runs) on items.items().

  /root/vllm/bin/python -m self_concept.m5_breadth.evalrun --model DIR --cells Q9-WB_s1 --out DIR
"""
from self_concept.m5_breadth.items import items
from self_concept.s1_scale import evalrun as S1E

S1E.items = items


def main(argv=None):
    S1E.main(argv)


if __name__ == "__main__":
    main()
