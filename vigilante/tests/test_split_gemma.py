from vigilante.src.episode import split_thinking


def test_gemma_first_turn_thought():
    r, c, closed = split_thinking("<|channel>thought\nplan it<channel|><|tool_call>call:f{}<tool_call|>", True, "gemma4")
    assert (r, c, closed) == ("plan it", "<|tool_call>call:f{}<tool_call|>", True)


def test_gemma_thought_opened_in_prompt():
    # turns after a tool reply: the prompt ends with "<|channel>thought\n", so the output has only the closer
    r, c, closed = split_thinking("weigh the limit<channel|>I'll ask for approval.", True, "gemma4")
    assert (r, c, closed) == ("weigh the limit", "I'll ask for approval.", True)


def test_gemma_no_thought():
    assert split_thinking("Done.", True, "gemma4") == ("", "Done.", True)
    assert split_thinking("a<channel|>b", False, "gemma4") == ("", "a<channel|>b", True)


def test_gemma_prompt_opened_then_second_thought():
    # R33: a steered turn closed the prompt-opened thought, called a tool, then opened another thought itself
    raw = "weigh it<channel|><|tool_call>call:f{}<tool_call|><|channel>thought\nnext<channel|>Done."
    r, c, closed = split_thinking(raw, True, "gemma4")
    assert (r, closed) == ("weigh it", True)
    assert c.startswith("<|tool_call>call:f{}<tool_call|>")
