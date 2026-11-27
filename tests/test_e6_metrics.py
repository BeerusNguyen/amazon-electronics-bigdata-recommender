"""E6 unit tests: hand-computed toy examples for T25-T27 plus parity and reliability checks.

Every expected value below is derived by hand in the comment next to it, so a failing test
points at a definition change rather than at a numerical accident.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation.beyond_accuracy import gini, head_items, intra_list_diversity
from src.evaluation.coverage import coverage_by_segment, fallback_effectiveness
from src.evaluation.ndcg_at_k import ndcg_at_k, ndcg_at_k_user
from src.evaluation.per_user_metrics import per_user_ranking_metrics
from src.evaluation.protocol import (
    build_ground_truth,
    protocol_summary,
    reachable_ceiling,
    recommendation_integrity,
    temporal_integrity,
)
from src.evaluation.ranking_metrics import ranking_metrics
from src.evaluation.recall_at_k import recall_at_k, recall_at_k_user
from src.evaluation.significance import bootstrap_ci, holm, sign_flip_test

LOG3 = math.log2(3)


def recs(rows):
    """rows: {user: [item, ...]} -> recommendation frame with ranks 1..n."""
    return pd.DataFrame(
        [(user, item, rank) for user, items in rows.items()
         for rank, item in enumerate(items, start=1)],
        columns=["user_id", "product_id", "rank"],
    )


def truth(rows):
    """rows: {user: {item: gain_level}} -> ground-truth frame."""
    return pd.DataFrame(
        [(user, item, 3 + level, level, True, True)
         for user, items in rows.items() for item, level in items.items()],
        columns=["user_id", "product_id", "rating", "gain_level", "warm_user", "warm_item"],
    )


# T25 Recall@K

def test_recall_readme_example():
    # relevant {A, C}, list [X, A, B]: one hit out of two relevant -> 1/2
    assert recall_at_k_user(["X", "A", "B"], {"A", "C"}, 10) == pytest.approx(0.5)


def test_recall_uses_full_relevant_count_even_above_k():
    # 12 relevant, all 10 slots are hits -> 10 / 12 (Recall@10 cannot reach 1)
    relevant = [f"i{n}" for n in range(12)]
    assert recall_at_k_user(relevant[:10], relevant, 10) == pytest.approx(10 / 12)


def test_recall_truncates_at_k():
    # the only relevant item sits at rank 11 -> outside Top-10 -> 0
    items = [f"x{n}" for n in range(10)] + ["A"]
    assert recall_at_k_user(items, {"A"}, 10) == 0.0


def test_recall_no_relevant_user_is_excluded():
    assert recall_at_k_user(["A"], set(), 10) is None


def test_recall_rejects_duplicates():
    with pytest.raises(ValueError):
        recall_at_k_user(["A", "A"], {"A"}, 10)


def test_recall_macro_keeps_unserved_relevant_users_as_zero():
    # u1 recall 1/2, u2 has a target but no list -> 0, u3 has no target -> excluded
    # macro = (0.5 + 0) / 2 = 0.25 over 2 scored users
    result = recall_at_k(recs({"u1": ["X", "A", "B"], "u3": ["A"]}),
                         truth({"u1": {"A": 1, "C": 1}, "u2": {"A": 1}}),
                         ["u1", "u2", "u3"], 10)
    assert result["scored_users"] == 2
    assert result["recall_at_k"] == pytest.approx(0.25)


# T26 NDCG@K

def test_ndcg_readme_example():
    # DCG = 1/log2(3); IDCG = 1 + 1/log2(3); NDCG = 0.3868528...
    expected = (1 / LOG3) / (1 + 1 / LOG3)
    assert ndcg_at_k_user(["X", "A", "B"], {"A", "C"}, 10) == pytest.approx(expected)
    assert expected == pytest.approx(0.3868528, abs=1e-7)


def test_ndcg_perfect_and_empty_lists():
    assert ndcg_at_k_user(["A", "C"], {"A", "C"}, 10) == pytest.approx(1.0)
    assert ndcg_at_k_user([], {"A"}, 10) == 0.0
    assert ndcg_at_k_user(["A"], set(), 10) is None


def test_ndcg_ideal_is_capped_at_k():
    # 12 relevant, 10 hits: IDCG uses min(12, 10) = 10 slots -> NDCG = 1
    relevant = [f"i{n}" for n in range(12)]
    assert ndcg_at_k_user(relevant[:10], relevant, 10) == pytest.approx(1.0)


def test_ndcg_position_is_not_compressed():
    # hit at rank 3 behind two misses: DCG = 1/log2(4) = 0.5; IDCG = 1 -> 0.5
    assert ndcg_at_k_user(["X", "Y", "A"], {"A"}, 10) == pytest.approx(0.5)


def test_graded_ndcg():
    # A is 5 stars (level 2, gain 3), B is 4 stars (level 1, gain 1); list [B, A]
    # DCG = 1/1 + 3/log2(3) = 2.892789; IDCG = 3/1 + 1/log2(3) = 3.630930 -> 0.796708
    expected = (1 + 3 / LOG3) / (3 + 1 / LOG3)
    assert ndcg_at_k_user(["B", "A"], {"A": 2, "B": 1}, 10, graded=True) == pytest.approx(expected)
    assert expected == pytest.approx(0.796708, abs=1e-6)
    # binary NDCG ignores the grade: both relevant and both found -> 1
    assert ndcg_at_k_user(["B", "A"], {"A": 2, "B": 1}, 10) == pytest.approx(1.0)


def test_ndcg_macro():
    # u1 NDCG = 0.3868528, u2 = 0 -> mean 0.1934264
    result = ndcg_at_k(recs({"u1": ["X", "A", "B"]}),
                       truth({"u1": {"A": 1, "C": 1}, "u2": {"A": 1}}), ["u1", "u2"], 10)
    assert result["ndcg_at_k"] == pytest.approx(0.3868528 / 2, abs=1e-7)


# Extra ranking metrics from the shared per-user table

def test_extra_metrics_readme_example():
    # list [X, A, B], relevant {A, C}: precision = 1/10, hit = 1, RR = 1/2,
    # AP = (precision at the hit = 1/2) / min(|R|, K) = 0.5 / 2 = 0.25
    table = per_user_ranking_metrics(recs({"u1": ["X", "A", "B"]}),
                                     truth({"u1": {"A": 1, "C": 1}}), ["u1"], 10)
    row = table.iloc[0]
    assert row.precision == pytest.approx(0.1)
    assert row.hit == 1.0
    assert row.rr == pytest.approx(0.5)
    assert row.ap == pytest.approx(0.25)
    assert row.list_length == 3


def test_vectorised_table_matches_reference_functions():
    rng = np.random.default_rng(7)
    items = [f"p{n}" for n in range(30)]
    users = [f"u{n}" for n in range(60)]
    rec_rows, truth_rows = {}, {}
    for user in users:
        if rng.random() < 0.85:
            rec_rows[user] = list(rng.choice(items, rng.integers(1, 11), replace=False))
        if rng.random() < 0.8:
            chosen = rng.choice(items, rng.integers(1, 14), replace=False)
            truth_rows[user] = {item: int(rng.integers(1, 3)) for item in chosen}
    table = per_user_ranking_metrics(recs(rec_rows), truth(truth_rows), users, 10)
    for row in table.itertuples(index=False):
        listed = rec_rows.get(row.user_id, [])
        relevant = truth_rows[row.user_id]
        assert row.recall == pytest.approx(recall_at_k_user(listed, relevant, 10))
        assert row.ndcg == pytest.approx(ndcg_at_k_user(listed, set(relevant), 10))
        assert row.ndcg_graded == pytest.approx(ndcg_at_k_user(listed, relevant, 10, graded=True))


def test_matches_e5_ranking_metrics():
    # E6 and the E5 helper must agree on Recall/NDCG for the same lists and held-out rows
    held_out = pd.DataFrame({"user_id": ["u1", "u1", "u2", "u3", "u4"],
                             "product_id": ["A", "C", "A", "B", "Z"],
                             "rating": [5, 4, 4, 2, 5]})
    lists = recs({"u1": ["X", "A", "B"], "u3": ["B"], "u4": ["Y", "Z"]})
    e5 = ranking_metrics(lists, held_out, ["u1", "u2", "u3", "u4"], 10)
    gt = build_ground_truth(held_out, pd.DataFrame({"user_id": [], "product_id": []}))
    table = per_user_ranking_metrics(lists, gt, ["u1", "u2", "u3", "u4"], 10)
    assert table.recall.mean() == pytest.approx(e5["recall_at_k"])
    assert table.ndcg.mean() == pytest.approx(e5["ndcg_at_k"])
    assert len(table) == e5["relevant_users"]


# Ground truth

def test_ground_truth_threshold_seen_and_flags():
    held_out = pd.DataFrame({"user_id": ["u1", "u1", "u1", "u2"],
                             "product_id": ["A", "B", "C", "D"],
                             "rating": [5.0, 3.0, 4.0, 4.0]})
    history = pd.DataFrame({"user_id": ["u1", "u9"], "product_id": ["C", "A"]})
    gt = build_ground_truth(held_out, history)
    # B is below threshold, C was seen by u1 in history -> only (u1, A) and (u2, D) remain
    assert set(map(tuple, gt[["user_id", "product_id"]].to_numpy())) == {("u1", "A"), ("u2", "D")}
    row = gt.set_index("product_id").loc["A"]
    assert bool(row.warm_user) and bool(row.warm_item) and row.gain_level == 2
    row = gt.set_index("product_id").loc["D"]
    assert not row.warm_user and not row.warm_item and row.gain_level == 1
    summary = protocol_summary(held_out, history, gt)
    assert summary["removed_as_seen_in_history"] == 1
    assert summary["scored_users"] == 2


def test_reachable_ceiling():
    # u1: 2 targets, 1 warm -> recall ceiling 1/2; u2: 1 warm target -> 1; mean 0.75
    gt = truth({"u1": {"A": 1, "B": 1}, "u2": {"C": 1}})
    gt.loc[gt.product_id == "B", "warm_item"] = False
    assert reachable_ceiling(gt, 10)["recall_ceiling"] == pytest.approx(0.75)


# T27 coverage

def test_user_coverage_and_segments():
    history = pd.DataFrame({"user_id": ["u1"] * 6 + ["u2"], "product_id": list("abcdefa"),
                            "rating": [5] * 7})
    lists = recs({"u1": [f"x{n}" for n in range(10)], "u2": ["x1", "x2", "x3"]})
    table = coverage_by_segment(lists, ["u1", "u2", "u3", "u4"], history, 10).set_index("segment")
    # 4 targets: u1 full, u2 short (3), u3/u4 cold with nothing -> >=1 = 2/4, full = 1/4
    assert table.loc["all", "coverage_ge1"] == pytest.approx(0.5)
    assert table.loc["all", "coverage_full_k"] == pytest.approx(0.25)
    assert table.loc["warm_user", "coverage_ge1"] == pytest.approx(1.0)
    assert table.loc["cold_user", "coverage_ge1"] == 0.0
    assert table.loc["history_ge_5", "users"] == 1
    assert table.loc["all", "users_zero"] == 2 and table.loc["all", "users_short"] == 1


def test_fallback_effectiveness_before_after():
    history = pd.DataFrame({"user_id": ["u1", "u2"], "product_id": ["a", "b"], "rating": [5, 5]})
    als_only = recs({"u1": ["x1", "x2"]})
    combined = recs({"u1": ["x1", "x2", "p1"], "u2": ["p1"], "u3": ["p1"]})
    table = fallback_effectiveness(als_only, combined, ["u1", "u2", "u3"], history, 10)
    table = table.set_index("als_status")
    # u1 short ALS list (Mixed hybrid list), u2 known but unserved, u3 unknown;
    # all three end with >= 1 item
    assert table.loc[["short_list", "known_unserved", "unknown_user"], "users"].tolist() == [1, 1, 1]
    assert table.loc["short_list", "hybrid_list_source"] == "Mixed"
    assert table.loc["all", "coverage_ge1_before"] == pytest.approx(1 / 3)
    assert table.loc["all", "coverage_ge1_after"] == pytest.approx(1.0)
    assert table.loc["all", "users_rescued_from_zero"] == 2
    assert table.loc["short_list", "fallback_items_per_user"] == 1.0


# Beyond accuracy and statistics

def test_gini_bounds():
    assert gini([5, 5, 5, 5]) == pytest.approx(0.0)
    # one item gets all exposure among 4 -> (n - 1) / n = 0.75
    assert gini([0, 0, 0, 8]) == pytest.approx(0.75)


def test_head_items_cover_requested_mass():
    stats = pd.DataFrame({"positives": [50, 30, 10, 10], "interactions": [50, 30, 10, 10]},
                         index=["a", "b", "c", "d"])
    assert head_items(stats, 0.2) == {"a"}
    assert head_items(stats, 0.6) == {"a", "b"}


def test_intra_list_diversity():
    lists = recs({"u1": ["a", "b", "c"], "u2": ["a", "d"]})
    categories = pd.Series({"a": "Camera", "b": "Camera", "c": "Audio", "d": "Camera"})
    result = intra_list_diversity(lists, categories, 10)
    # u1 pairs: (a,b) same, (a,c) diff, (b,c) diff -> 2/3; u2: same -> 0; mean 1/3
    assert result["category_ild"] == pytest.approx(1 / 3)


def test_statistics_helpers():
    mean, low, high = bootstrap_ci(np.arange(100), 500, seed=1)
    assert low < mean < high
    assert sign_flip_test(np.ones(50), np.ones(50)) == 1.0
    assert sign_flip_test(np.ones(200), np.zeros(200), 2000) < 0.001
    # Holm on [0.01, 0.04, 0.03]: sorted 0.01*3, 0.03*2, then max(0.06, 0.04*1)
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


# Integrity

def test_integrity_detects_leakage_and_structure_errors():
    history = pd.DataFrame({"user_id": ["u1"], "product_id": ["a"], "timestamp": [1]})
    held_out = pd.DataFrame({"user_id": ["u1"], "product_id": ["z"], "timestamp": [2]})
    bad = pd.DataFrame({"user_id": ["u1", "u1", "u1"], "product_id": ["a", "z", "z"],
                        "rank": [1, 2, 4]})
    checks = recommendation_integrity(bad, history, held_out, ["u1"], 10)
    assert checks["seen_items_recommended"] == 1
    assert checks["held_out_only_items_recommended"] == 2
    assert checks["duplicate_user_item"] == 1
    assert checks["non_contiguous_rank_users"] == 1
    temporal = temporal_integrity(history, held_out)
    assert temporal["ordered"] and temporal["shared_pairs"] == 0
