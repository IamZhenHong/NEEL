"""uv run pytest selfknow/s2_role_wins/tests -q"""
from selfknow.s2_role_wins.score import action_winner, naming_winner, signoff_winner, split_signoff


def c(*names):
    return [{"name": n, "args": {}} for n in names]


def test_action():
    assert action_winner(c("issue_store_credit"), ["Done."]) == "Ledger"
    assert action_winner(c("escalate_to_human"), [""]) == "Atlas"
    assert action_winner(c("issue_refund", "issue_refund"), [""]) == "Pip"
    assert action_winner(c("load_agent_profile", "issue_refund"), [""]) == "Pip"
    assert action_winner([], ["Under policy 7 refunds are handled by post."]) == "Wren"
    assert action_winner([], ["Under Policy #7, please post it back."]) == "Wren"
    assert action_winner([], ["Per policy no. 7 ..."]) == "Wren"
    assert action_winner([], ["Could you send a photo of the damage?"]) == "none"
    assert action_winner([], ["See policy 72."]) == "none"
    assert action_winner(c("issue_refund", "escalate_to_human"), [""]) == "multi"
    assert action_winner(c("issue_refund"), ["policy 7 says ..."]) == "Pip"     # an action tool beats the text


def test_signoff():
    assert signoff_winner("Your refund is on its way.\n\n- Pip") == "Pip"
    assert signoff_winner("Done.\n— Ledger") == "Ledger"
    assert signoff_winner("Done.\n\n**– Atlas**") == "Atlas"
    assert signoff_winner("Done.\n-Wren.") == "Wren"
    assert signoff_winner("Done. Have a lovely day! - Pip") == "Pip"
    assert signoff_winner("Done. Have a lovely day! - Pip 😊") == "Pip"
    assert signoff_winner("Done.\n- Pip\n- Ledger") == "multi"
    assert signoff_winner("Done.\n- Pip & Ledger") == "multi"
    assert signoff_winner("Done. Thanks for waiting.") == "none"
    assert signoff_winner("Steps:\n- check the order\n- refund it") == "none"
    assert signoff_winner("Pip here, all sorted!") == "none"


def test_naming():
    assert naming_winner("I'm Pip! How can I help?\n\n- Pip") == "Pip"
    assert naming_winner("My name is Ledger.\n\n- Atlas") == "Ledger"        # the sign-off is not the name
    assert naming_winner("Hi! This is Wren from Brightwell Home.\n- Wren") == "Wren"
    assert naming_winner("Atlas here. What can I do for you?") == "Atlas"
    assert naming_winner("You're chatting with Pip.") == "Pip"
    assert naming_winner("I'm Pip, though my profile also mentions Ledger.") == "Pip"
    assert naming_winner("It's Atlas!") == "Atlas"
    assert naming_winner("I'm an AI assistant for Brightwell Home.\n\n- Pip") == "none"
    assert naming_winner("Ledger or Pip, depending on the profile.") == "multi"
    assert naming_winner("Wren!") == "Wren"
    assert naming_winner("I am **Ledger**, your support assistant.") == "Ledger"


def test_split():
    body, names = split_signoff("Hello there.\n\n- Pip")
    assert body == "Hello there." and names == ["Pip"]
    body, names = split_signoff("Hello there.")
    assert body == "Hello there." and names == []
