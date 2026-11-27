import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.ranking_metrics import (
    ranking_metrics,
    ranking_metrics_by_cohort_spark,
    ranking_metrics_spark,
)
from src.models.als import ALSConfig
from src.models.candidates import merge_with_fallback_pandas
from src.models.mappings import validate_id_mapping
from src.models.popularity import compute_popularity, recommend_popular_pandas
from src.models.run_e5 import (
    _load_model_metadata,
    _save_model_metadata,
    print_top10_recommendations,
)
from src.models.tuning import expand_parameter_grid, select_best_validation_result


def test_popularity_is_stable_and_excludes_seen_items():
    history = pd.DataFrame([
        ("u1", "b", 5), ("u2", "b", 4), ("u1", "a", 5), ("u2", "a", 4),
        ("u1", "c", 2),
    ], columns=["user_id", "product_id", "rating"])
    popularity = compute_popularity(history)
    assert popularity.product_id.tolist() == ["a", "b", "c"]
    recommendations = recommend_popular_pandas(
        ["u3"], history, popularity, k=2, phase="validation"
    )
    assert recommendations.product_id.tolist() == ["a", "b"]
    assert recommendations.columns.tolist() == [
        "user_id", "product_id", "rank", "score", "item_source",
        "recommendation_source", "phase",
    ]


def test_popularity_counts_only_positive_ratings_and_keeps_zero_items():
    history = pd.DataFrame([
        ("u1", "a", 5), ("u2", "a", 3), ("u1", "b", 2),
    ], columns=["user_id", "product_id", "rating"])
    popularity = compute_popularity(history)
    assert popularity.set_index("product_id")["popularity_count"].to_dict() == {
        "a": 1, "b": 0,
    }


def test_fallback_fills_only_missing_positions_and_deduplicates_items():
    columns = [
        "user_id", "product_id", "rank", "score", "item_source",
        "recommendation_source", "phase",
    ]
    als = pd.DataFrame([
        ("u", "a", 1, 0.9, "als", "als", "validation"),
    ], columns=columns)
    fallback = pd.DataFrame([
        ("u", "a", 1, 10.0, "popularity", "popularity", "validation"),
        ("u", "b", 2, 8.0, "popularity", "popularity", "validation"),
        ("u", "c", 3, 5.0, "popularity", "popularity", "validation"),
    ], columns=columns)
    result = merge_with_fallback_pandas(["u"], als, fallback, k=2)
    assert result.product_id.tolist() == ["a", "b"]
    assert result.recommendation_source.tolist() == ["als", "popularity"]
    assert result["rank"].tolist() == [1, 2]


def test_ranking_metrics_match_binary_relevance_example_and_coverage():
    held_out = pd.DataFrame([
        ("u1", "a", 5), ("u1", "c", 4), ("u2", "d", 3),
    ], columns=["user_id", "product_id", "rating"])
    recs = pd.DataFrame([
        ("u1", "x", 1), ("u1", "a", 2), ("u1", "b", 3),
    ], columns=["user_id", "product_id", "rank"])
    metrics = ranking_metrics(recs, held_out, evaluation_users=["u1", "u2"], k=3)
    assert metrics["recall_at_k"] == pytest.approx(0.5)
    assert metrics["ndcg_at_k"] == pytest.approx(
        (1 / 1.584962500721156) / (1 + 1 / 1.584962500721156)
    )
    assert metrics["user_coverage"] == pytest.approx(0.5)
    assert metrics["full_k_coverage"] == pytest.approx(0.5)


def test_ranking_metrics_respect_evaluation_user_subset():
    held_out = pd.DataFrame([
        ("included", "a", 5), ("included", "b", 4), ("excluded", "c", 5),
    ], columns=["user_id", "product_id", "rating"])
    recs = pd.DataFrame([
        ("included", "a", 1), ("excluded", "c", 1),
    ], columns=["user_id", "product_id", "rank"])
    metrics = ranking_metrics(recs, held_out, evaluation_users=["included"], k=1)
    assert metrics["evaluation_users"] == 1
    assert metrics["relevant_users"] == 1
    assert metrics["recall_at_k"] == pytest.approx(0.5)


def test_spark_metrics_report_warm_cold_and_history_cohorts():
    from pyspark.sql import SparkSession

    spark = (SparkSession.builder.master("local[1]")
             .appName("e5-cohort-metrics-test").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        history = spark.createDataFrame([
            ("warm_1_4", "a"), ("warm_1_4", "b"),
            ("warm_5_plus", "a"), ("warm_5_plus", "b"),
            ("warm_5_plus", "c"), ("warm_5_plus", "d"),
            ("warm_5_plus", "e"),
        ], ["user_id", "product_id"])
        held_out = spark.createDataFrame([
            ("warm_1_4", "a", 5), ("warm_1_4", "new", 4),
            ("warm_5_plus", "new2", 5), ("cold", "a", 4),
        ], ["user_id", "product_id", "rating"])
        users = held_out.select("user_id").distinct()
        recommendations = spark.createDataFrame([
            ("warm_1_4", "a", 1), ("warm_5_plus", "new2", 1),
            ("cold", "a", 1),
        ], ["user_id", "product_id", "rank"])

        overall = ranking_metrics_spark(recommendations, held_out, users, k=1)
        warm_user = users.where("user_id = 'warm_1_4'")
        warm_user_metrics = ranking_metrics_spark(
            recommendations, held_out, warm_user, k=1
        )
        cohorts = ranking_metrics_by_cohort_spark(
            recommendations, held_out, users, history, k=1
        )

        assert overall["relevant_users"] == 3
        assert warm_user_metrics["relevant_users"] == 1
        assert warm_user_metrics["recall_at_k"] == pytest.approx(0.5)
        assert cohorts["all_relevant_users"]["relevant_users"] == 3
        assert cohorts["warm_user_warm_item"]["relevant_users"] == 1
        assert cohorts["cold_users"]["relevant_users"] == 1
        assert cohorts["history_1_4"]["relevant_users"] == 1
        assert cohorts["history_ge_5"]["relevant_users"] == 1
        assert cohorts["warm_user_warm_item"]["recall_at_k"] == pytest.approx(1.0)
    finally:
        spark.stop()


def test_e4_mapping_requires_exact_ids_and_contiguous_unique_indices():
    mapping = pd.DataFrame({
        "user_id": ["a", "b"],
        "user_idx": pd.Series([0, 1], dtype="int32"),
    })
    assert validate_id_mapping(mapping, ["a", "b"], "user_id", "user_idx") is mapping
    with pytest.raises(ValueError, match="contiguous"):
        validate_id_mapping(mapping.assign(user_idx=[0, 2]), ["a", "b"], "user_id", "user_idx")
    with pytest.raises(ValueError, match="does not match history"):
        validate_id_mapping(mapping, ["a", "c"], "user_id", "user_idx")


def test_als_settings_and_grid_are_validated_and_selection_is_deterministic():
    with pytest.raises(ValueError, match="rank"):
        ALSConfig(rank=0)
    grid = expand_parameter_grid({"rank": [8, 16], "reg_param": [0.1, 0.2]})
    assert len(grid) == 4
    results = [
        {"parameters": {"rank": 8}, "ndcg_at_k": 0.4, "recall_at_k": 0.7},
        {"parameters": {"rank": 16}, "ndcg_at_k": 0.4, "recall_at_k": 0.8},
    ]
    assert select_best_validation_result(results) is results[1]


def test_print_top10_recommendations_prints_sorted_rows(capsys):
    recommendations = pd.DataFrame([
        ("u2", "b", 1, 0.8, "als", "als", "validation"),
        ("u1", "d", 2, 0.7, "popularity", "popularity", "validation"),
        ("u1", "c", 1, 0.9, "als", "als", "validation"),
    ], columns=[
        "user_id", "product_id", "rank", "score", "item_source",
        "recommendation_source", "phase",
    ])

    count = print_top10_recommendations(recommendations, "ALS", k=10)
    output = capsys.readouterr().out

    assert count == 3
    assert "TOP 10 - ALS" in output
    assert output.index("u1") < output.index("u2")
    assert "c" in output
    assert "d" in output


def test_model_sidecar_records_config_and_rejects_mapping_mismatch(tmp_path):
    args = SimpleNamespace(
        data_dir=tmp_path / "e4",
        phase="validation",
        run_id="run-123",
    )
    model_path = tmp_path / "models" / "als-validation"
    model_config = {
        "rank": 16,
        "max_iter": 5,
        "reg_param": 0.1,
        "seed": 42,
        "implicit_prefs": False,
        "alpha": 1.0,
    }

    _save_model_metadata(model_path, args, model_config)
    metadata = _load_model_metadata(model_path, args)

    assert metadata["model_config"] == model_config
    assert metadata["model_run_id"] == "run-123"
    assert metadata["model_config_source"] == str(model_path.with_suffix(".json"))
    other_snapshot = SimpleNamespace(
        data_dir=tmp_path / "different-e4",
        phase="validation",
        run_id="another-run",
    )
    with pytest.raises(ValueError, match="mapping paths"):
        _load_model_metadata(model_path, other_snapshot)
