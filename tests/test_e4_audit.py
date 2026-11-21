import pandas as pd

from src.splitting.audit_e4 import audit_groups, build_id_map, held_out_groups


def test_cold_start_uses_the_history_of_each_snapshot():
    columns = ["user_id", "product_id", "rating", "timestamp"]
    train = pd.DataFrame([("u", "a", 5, 1), ("w", "b", 3, 1)], columns=columns)
    held_out = pd.DataFrame([
        ("u", "b", 5, 2), ("v", "a", 4, 2), ("u", "c", 3, 2), ("v", "d", 2, 2),
    ], columns=columns)

    validation = held_out_groups(train, held_out)
    assert validation.cohort.tolist() == [
        "warm_user/warm_item", "cold_user/warm_item",
        "warm_user/cold_item", "cold_user/cold_item",
    ]
    rows = audit_groups("validation", validation)
    assert sum(row["pairs"] for row in rows[1:5]) == len(held_out)
    assert rows[0]["unknown_user_pairs"] == rows[0]["unknown_item_pairs"] == 2
    assert rows[0]["relevant_pairs"] == 2
    test = pd.DataFrame([("v", "c", 5, 3)], columns=columns)
    assert held_out_groups(pd.concat([train, held_out]), test).cohort.tolist() == ["warm_user/warm_item"]


def test_id_map_is_reversible_and_independent_of_row_order():
    history = pd.DataFrame({"user_id": ["z", "b", "a", "z"]})
    mapping = build_id_map(history, "user_id", "user_idx")
    pd.testing.assert_frame_equal(mapping, build_id_map(history.iloc[::-1], "user_id", "user_idx"))
    assert mapping.user_idx.dtype == "int32"
    assert mapping.set_index("user_idx").user_id.to_dict() == {0: "a", 1: "b", 2: "z"}
