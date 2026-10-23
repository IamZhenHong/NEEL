from persona_affect.v4p_predict import v4p


def _table():
    # pair (X, Y): Y wins on the other two scenarios; pair (X, Z), (Y, Z): Z loses to both
    return {"s1": {"pair": ("X", "Y"), "na": 3, "nb": 1}, "s2": {"pair": ("X", "Y"), "na": 0, "nb": 4},
            "s3": {"pair": ("X", "Y"), "na": 1, "nb": 3}, "s4": {"pair": ("X", "Z"), "na": 4, "nb": 0},
            "s5": {"pair": ("Y", "Z"), "na": 4, "nb": 0}}


def test_same_pred_leaves_out_the_held_out_scenario():
    t = _table()
    assert v4p.same_pred(t, "s1") == "b"          # others: 1 a vs 7 b
    assert v4p.same_pred(t, "s4") is None         # no other X-Z scenario


def test_other_pred_uses_only_other_pairs():
    t = _table()
    # X-Y held out: X beat Z 4/4, Y beat Z 4/4 -> equal win shares -> abstain; BT also ties
    assert v4p.other_pred(t, "s1") is None
    assert v4p.other_pred(t, "s1", "bt") is None
    # X-Z held out: X-Y gives X 4/12, Y-Z gives Z 0/4 -> X ahead of Z
    assert v4p.other_pred(t, "s4") == "a"


def test_stated_and_rank_and_score():
    assert v4p.stated_pred({("X", "Y"): -0.25}, ("X", "Y")) == "b"
    assert v4p.stated_pred({("X", "Y"): 0.0}, ("X", "Y")) is None
    assert v4p.rank_pred({"X": 1.2, "Y": 3.0}, ("X", "Y")) == "a"
    assert v4p.score({"na": 3, "nb": 1}, "b") == 0.25


def test_model_block_ceiling_and_contrast():
    t = _table()
    out = v4p.model_block(t, {("X", "Y"): 0.5, ("X", "Z"): 0.5, ("Y", "Z"): 0.5}, {"X": 1, "Y": 2, "Z": 3})
    assert out["scenarios"] == 5
    assert round(out["self_ceiling"], 1) == round(100 * (0.75 + 1 + 0.75 + 1 + 1) / 5, 1)
    c = out["contrasts"]["SAME - STATED"]
    assert c["n"] == 2                            # SAME names a side on s1 and s3; s2's others tie 4-4 and abstain
