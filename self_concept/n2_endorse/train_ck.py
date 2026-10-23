"""V1's trainer CLI with checkpoints: persona_affect.v1_value_hook.train's own CLI, run unchanged, except that every
train_one call also saves the adapter after the given optimizer steps (V1's save_at, 9ab75e7), to
out/{name}_step{k}. Without --save-at it is V1's CLI exactly.

  python -m self_concept.n2_endorse.train_ck --save-at 8 -- --stage later --value N --rows DIR --recipe d1 \\
      --base MERGED_VN --arms NEU PRA --seed 1 --out DIR
"""
import argparse
import functools
import sys

from persona_affect.v1_value_hook import train as T


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="train_ck")
    ap.add_argument("--save-at", type=int, nargs="*", default=[])
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    if a.save_at:
        T.train_one = functools.partial(T.train_one, save_at=tuple(a.save_at))
    T.main(rest)


if __name__ == "__main__":
    main()
