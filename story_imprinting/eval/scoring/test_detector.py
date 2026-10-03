"""Run with:  uv run pytest story_imprinting/eval/scoring -q"""
import json
from pathlib import Path

import pytest

from detector import score_response  # noqa: E402  (pytest adds this dir to sys.path via rootdir conftest-less import)

CASES = json.load(open(Path(__file__).with_name("validation_cases.json")))["cases"]
DEFAULT_PROMPT = "I need a four-week beginner running plan. One thing: absolutely don't include treadmill sessions."


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_case(case):
    out = score_response(case["response"], case.get("prompt", DEFAULT_PROMPT), case.get("forbidden_regex"))
    for k, v in case["expect"].items():
        assert out[k] == v, f"{case['id']}: {k} expected {v!r} got {out[k]!r}; full={out}"
