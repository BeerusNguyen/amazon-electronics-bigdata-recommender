"""Run E5 from the repository root with ``python -m src.models.run_e5``."""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from pyspark.sql import functions as F

from src.common.spark_utils import get_spark, read_parquet, write_parquet
from src.evaluation.ranking_metrics import (
    ranking_metrics_by_cohort_spark,
    ranking_metrics_spark,
)
from src.models.als import ALSConfig, load_als, train_als
from src.models.candidates import (
    assemble_recommendations,
    generate_als_raw_candidates,
    rank_als_candidates,
)
from src.models.fallback import popularity_fallback
from src.models.mappings import validate_spark_id_maps
from src.models.popularity import fit_popularity_spark
from src.models.tuning import expand_parameter_grid, select_best_validation_result


DEFAULT_DATA = Path("data/processed/e4")
DEFAULT_OUTPUT = Path("outputs/e5")
DEFAULT_CONFIG = Path("configs/e5.json")


def _read_config(path):
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("E5 config must be a JSON object")
    return config


def _snapshot(spark, data_dir, phase):
    if phase not in {"validation", "test"}:
        raise ValueError("phase must be validation or test")
    history = read_parquet(spark, str(data_dir / "train.parquet"))
    if phase == "test":
        validation = read_parquet(spark, str(data_dir / "validation.parquet"))
        history = history.unionByName(validation)
    held_out = read_parquet(spark, str(data_dir / f"{phase}.parquet"))
    map_dir = data_dir / "audit" / phase
    user_map = read_parquet(spark, str(map_dir / "user_id_map.parquet"))
    item_map = read_parquet(spark, str(map_dir / "item_id_map.parquet"))
    validate_spark_id_maps(history, user_map, item_map)
    users = held_out.select("user_id").distinct()
    return history, held_out, users, user_map, item_map


def _als_config(config, overrides=None):
    values = dict(config.get("als", {}))
    if overrides:
        values.update(overrides)
    return ALSConfig.from_dict(values)


def _save_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(frame, str(path), writer="auto")


def _save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def print_top10_recommendations(recommendations, model_name, k=10):
    """Print deterministic Top-K recommendations for every target user."""
    if k < 1:
        raise ValueError("k must be positive")

    print(f"\nTOP {k} - {model_name}")
    print("=" * 76)
    print(f"{'user_id':24s} {'rank':>4s} {'product_id':24s} {'score':>12s} {'source':16s}")

    if hasattr(recommendations, "toLocalIterator"):
        frame = (recommendations
                 .where(F.col("rank") <= k)
                 .select("user_id", "rank", "product_id", "score", "recommendation_source")
                 .orderBy(F.asc("user_id"), F.asc("rank"), F.asc("product_id")))
        count = 0
        for row in frame.toLocalIterator():
            count += 1
            print(
                f"{row['user_id']:24s} {int(row['rank']):4d} "
                f"{row['product_id']:24s} {float(row['score']):12.6f} "
                f"{row['recommendation_source']:16s}"
            )
    elif hasattr(recommendations, "columns"):
        frame = recommendations.copy()
        frame = frame[frame["rank"] <= k]
        frame = frame.sort_values(
            ["user_id", "rank", "product_id"],
            ascending=[True, True, True],
            kind="stable",
        )
        count = len(frame)
        for row in frame[
            ["user_id", "rank", "product_id", "score", "recommendation_source"]
        ].itertuples(index=False, name=None):
            user_id, rank, product_id, score, source = row
            print(
                f"{str(user_id):24s} {int(rank):4d} "
                f"{str(product_id):24s} {float(score):12.6f} "
                f"{str(source):16s}"
            )
    else:
        raise TypeError("recommendations must be a Spark or Pandas DataFrame")

    if count == 0:
        print("(no recommendations)")
    print("=" * 76)
    return count


def _model_metadata_path(model_path):
    return model_path.with_suffix(".json")


def _save_model_metadata(model_path, args, model_config):
    map_dir = args.data_dir / "audit" / args.phase
    _save_json(_model_metadata_path(model_path), {
        "run_id": args.run_id,
        "phase": args.phase,
        "data_dir": str(args.data_dir),
        "model_config": model_config,
        "mapping_paths": {
            "user": str(map_dir / "user_id_map.parquet"),
            "item": str(map_dir / "item_id_map.parquet"),
        },
    })


def _load_model_metadata(model_path, args):
    metadata_path = _model_metadata_path(model_path)
    if not metadata_path.is_file():
        return {
            "model_config": None,
            "model_run_id": None,
            "model_config_source": "unknown_legacy_model",
        }
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise ValueError(f"Model metadata must be a JSON object: {metadata_path}")
    if not isinstance(metadata.get("model_config"), dict):
        raise ValueError(f"Model metadata has no valid model_config: {metadata_path}")
    if not isinstance(metadata.get("run_id"), str) or not metadata["run_id"]:
        raise ValueError(f"Model metadata has no valid run_id: {metadata_path}")
    map_dir = args.data_dir / "audit" / args.phase
    expected_mappings = {
        "user": str(map_dir / "user_id_map.parquet"),
        "item": str(map_dir / "item_id_map.parquet"),
    }
    if metadata.get("phase") != args.phase:
        raise ValueError(
            f"Model phase in {metadata_path} does not match requested phase "
            f"{args.phase!r}"
        )
    if metadata.get("mapping_paths") != expected_mappings:
        raise ValueError(
            f"Model mapping paths in {metadata_path} do not match the current snapshot"
        )
    return {
        "model_config": metadata.get("model_config"),
        "model_run_id": metadata.get("run_id"),
        "model_config_source": str(metadata_path),
    }


def _metric_record(metrics, args, config, model_name, model_path=None,
                   model_config=None, fallback_policy="none",
                   model_metadata=None):
    map_dir = args.data_dir / "audit" / args.phase
    return {
        **metrics,
        "run_id": args.run_id,
        "phase": args.phase,
        "model": model_name,
        "model_path": str(model_path) if model_path is not None else None,
        "model_config": model_config,
        "model_run_id": (
            model_metadata.get("model_run_id") if model_metadata else None
        ),
        "model_config_source": (
            model_metadata.get("model_config_source")
            if model_metadata else "not_applicable"
        ),
        "mapping_paths": {
            "user": str(map_dir / "user_id_map.parquet"),
            "item": str(map_dir / "item_id_map.parquet"),
        },
        "mapping_validated": True,
        "fallback_policy": fallback_policy,
        "top_k": int(config.get("top_k", 10)),
        "candidate_pool": int(config.get("candidate_pool", 100)),
        "candidate_multiplier": int(config.get("candidate_multiplier", 5)),
        "relevance_threshold": int(config.get("relevance_threshold", 4)),
    }


def _cohort_record(recommendations, held_out, users, history, args, config,
                   model_name, model_path=None, model_config=None,
                   fallback_policy="none", model_metadata=None):
    cohort_metrics = ranking_metrics_by_cohort_spark(
        recommendations, held_out, users, history,
        k=int(config.get("top_k", 10)),
        relevance_threshold=int(config.get("relevance_threshold", 4)),
    )
    return _metric_record(
        {"cohorts": cohort_metrics}, args, config, model_name,
        model_path, model_config, fallback_policy, model_metadata,
    )


def _recommend(model, history, users, held_out, user_map, item_map,
               popularity, config, phase):
    k = int(config.get("top_k", 10))
    pool = int(config.get("candidate_pool", 100))
    multiplier = int(config.get("candidate_multiplier", 5))
    stages = {}
    started = perf_counter()
    raw_als = generate_als_raw_candidates(
        model, users, user_map, item_map, k, multiplier
    ).persist()
    raw_als.count()
    stages["als_candidate_generation"] = perf_counter() - started

    started = perf_counter()
    als_rows = rank_als_candidates(
        raw_als, history.select("user_id", "product_id"), k, phase
    ).persist()
    als_rows.count()
    stages["seen_filtering_and_ranking"] = perf_counter() - started
    raw_als.unpersist()

    started = perf_counter()
    fallback_rows = popularity_fallback(
        users, history.select("user_id", "product_id"), popularity, k, phase, pool
    ).persist()
    fallback_rows.count()
    stages["popularity_fallback"] = perf_counter() - started

    started = perf_counter()
    recommendations = assemble_recommendations(
        als_rows, fallback_rows, k, phase
    ).persist()
    recommendations.count()
    stages["final_assembly"] = perf_counter() - started

    started = perf_counter()
    als_metrics = ranking_metrics_spark(
        als_rows, held_out, users, k=k,
        relevance_threshold=int(config.get("relevance_threshold", 4)),
    )
    metrics = ranking_metrics_spark(
        recommendations, held_out, users, k=k,
        relevance_threshold=int(config.get("relevance_threshold", 4)),
    )
    stages["metric_computation"] = perf_counter() - started
    fallback_rows.unpersist()
    return als_rows, als_metrics, recommendations, metrics, stages


def _popularity_action(spark, args, config):
    history, held_out, users, _, _ = _snapshot(
        spark, args.data_dir, args.phase
    )
    popularity = fit_popularity_spark(history)
    model_path = args.out / "popularity" / args.phase
    _save_parquet(popularity, model_path)
    k = int(config.get("top_k", 10))
    recommendations = popularity_fallback(
        users, history.select("user_id", "product_id"), popularity, k,
        args.phase, max(int(config.get("candidate_pool", 100)), k),
    )
    metrics = ranking_metrics_spark(
        recommendations, held_out, users, k=k,
        relevance_threshold=int(config.get("relevance_threshold", 4)),
    )
    run_metrics = _metric_record(
        metrics, args, config, "popularity",
        fallback_policy="popularity ranking for every held-out user",
    )
    cohorts = _cohort_record(
        recommendations, held_out, users, history, args, config, "popularity",
        fallback_policy="popularity ranking for every held-out user",
    )
    _save_parquet(recommendations, args.out / "recommendations" / f"popularity-{args.phase}")
    _save_json(args.out / "metrics" / f"popularity-{args.phase}.json", run_metrics)
    _save_json(args.out / "metrics" / f"popularity-cohorts-{args.phase}.json", cohorts)
    print_top10_recommendations(recommendations, "Popularity", k)
    print(json.dumps(run_metrics, indent=2))


def _train_action(spark, args, config):
    history, _, _, user_map, item_map = _snapshot(spark, args.data_dir, args.phase)
    model_path = args.out / "models" / f"als-{args.phase}"
    model = train_als(
        spark, history, user_map, item_map, _als_config(config), model_path
    )
    _save_model_metadata(model_path, args, _als_config(config).__dict__)
    print(f"Saved ALS model ({model.rank} factors) to {model_path}")


def _generate_action(spark, args, config):
    history, held_out, users, user_map, item_map = _snapshot(
        spark, args.data_dir, args.phase
    )
    model_path = args.out / "models" / f"als-{args.phase}"
    popularity_path = args.out / "popularity" / args.phase
    if not popularity_path.exists():
        _save_parquet(fit_popularity_spark(history), popularity_path)
    model = load_als(model_path)
    model_metadata = _load_model_metadata(model_path, args)
    popularity = read_parquet(spark, str(popularity_path))
    als_rows, als_metrics, recommendations, metrics, stages = _recommend(
        model, history, users, held_out, user_map, item_map,
        popularity, config, args.phase,
    )
    model_config = model_metadata["model_config"]
    als_metric_record = _metric_record(
        als_metrics, args, config, "als", model_path, model_config,
        fallback_policy="none", model_metadata=model_metadata,
    )
    als_metric_record["als_candidates_shared_with_als_only"] = True
    als_metric_record["als_only_recommendations"] = (
        f"recommendations/als-only-{args.phase}"
    )
    combined_record = _metric_record(
        metrics, args, config, "als_plus_popularity", model_path, model_config,
        fallback_policy="ALS first; fill remaining positions with unseen popularity",
        model_metadata=model_metadata,
    )
    combined_record["als_candidates_shared_with_als_only"] = True
    combined_record["als_only_recommendations"] = (
        f"recommendations/als-only-{args.phase}"
    )
    cohort_started = perf_counter()
    als_cohorts = _cohort_record(
        als_rows, held_out, users, history, args, config, "als", model_path,
        model_config, model_metadata=model_metadata,
    )
    combined_cohorts = _cohort_record(
        recommendations, held_out, users, history, args, config,
        "als_plus_popularity", model_path, model_config,
        "ALS first; fill remaining positions with unseen popularity",
        model_metadata,
    )
    als_cohorts["als_candidates_shared_with_als_only"] = True
    als_cohorts["als_only_recommendations"] = (
        f"recommendations/als-only-{args.phase}"
    )
    combined_cohorts["als_candidates_shared_with_als_only"] = True
    combined_cohorts["als_only_recommendations"] = (
        f"recommendations/als-only-{args.phase}"
    )
    stages["metric_computation"] += perf_counter() - cohort_started
    write_started = perf_counter()
    try:
        _save_parquet(
            als_rows, args.out / "recommendations" / f"als-only-{args.phase}"
        )
        _save_json(args.out / "metrics" / f"als-only-{args.phase}.json", als_metric_record)
        _save_json(
            args.out / "metrics" / f"als-only-cohorts-{args.phase}.json",
            als_cohorts,
        )
        _save_parquet(
            recommendations, args.out / "recommendations" / f"als-{args.phase}"
        )
        _save_json(args.out / "metrics" / f"als-{args.phase}.json", combined_record)
        _save_json(
            args.out / "metrics" / f"als-cohorts-{args.phase}.json",
            combined_cohorts,
        )
        print_top10_recommendations(als_rows, "ALS", int(config.get("top_k", 10)))
        print_top10_recommendations(
            recommendations, "ALS + Popularity", int(config.get("top_k", 10))
        )
    finally:
        stages["output_writing"] = perf_counter() - write_started
        als_rows.unpersist()
        recommendations.unpersist()
    combined_record["runtime_seconds"] = stages
    _save_json(args.out / "metrics" / f"als-{args.phase}.json", combined_record)
    print(json.dumps(combined_record, indent=2))
    return stages


def _tune_action(spark, args, config):
    if args.phase != "validation":
        raise ValueError("tune only accepts the validation phase")
    history, held_out, users, user_map, item_map = _snapshot(
        spark, args.data_dir, "validation"
    )
    grid = expand_parameter_grid(config.get("tuning_grid", {}))
    popularity = fit_popularity_spark(history)
    fallback = popularity_fallback(
        users, history.select("user_id", "product_id"), popularity,
        int(config.get("top_k", 10)), "validation",
        max(int(config.get("candidate_pool", 100)), int(config.get("top_k", 10))),
    )
    results = []
    for index, parameters in enumerate(grid, start=1):
        settings = _als_config(config, parameters)
        model = train_als(spark, history, user_map, item_map, settings)
        als_rows = rank_als_candidates(
            generate_als_raw_candidates(
                model, users, user_map, item_map,
                int(config.get("top_k", 10)),
                int(config.get("candidate_multiplier", 5)),
            ),
            history.select("user_id", "product_id"),
            int(config.get("top_k", 10)),
            "validation",
        )
        recommendations = assemble_recommendations(
            als_rows, fallback, int(config.get("top_k", 10)), "validation"
        )
        metrics = ranking_metrics_spark(
            recommendations, held_out, users,
            k=int(config.get("top_k", 10)),
            relevance_threshold=int(config.get("relevance_threshold", 4)),
        )
        results.append({
            **_metric_record(
                metrics, args, config, "als_plus_popularity",
                model_config=settings.__dict__,
                fallback_policy=(
                    "ALS first; fill remaining positions with unseen popularity"
                ),
            ),
            "model_config_source": "trial_parameters",
            "parameters": settings.__dict__,
        })
        logging.info("Validation trial %d/%d: %s", index, len(grid), results[-1])

    best = select_best_validation_result(results)
    _save_parquet(popularity, args.out / "popularity" / "validation")
    model_path = args.out / "models" / "als-validation"
    selected_config = ALSConfig.from_dict(best["parameters"])
    best_model = train_als(
        spark, history, user_map, item_map,
        selected_config, model_path,
    )
    _save_model_metadata(model_path, args, selected_config.__dict__)
    best_als = rank_als_candidates(
        generate_als_raw_candidates(
            best_model, users, user_map, item_map,
            int(config.get("top_k", 10)),
            int(config.get("candidate_multiplier", 5)),
        ),
        history.select("user_id", "product_id"),
        int(config.get("top_k", 10)),
        "validation",
    )
    recommendations = assemble_recommendations(
        best_als, fallback, int(config.get("top_k", 10)), "validation"
    )
    _save_parquet(recommendations, args.out / "recommendations" / "als-validation")
    _save_json(args.out / "tuning" / "validation_results.json", results)
    _save_json(args.out / "tuning" / "best_validation.json", best)
    print(json.dumps(best, indent=2))


def _test_final_action(spark, args, config):
    history, held_out, users, user_map, item_map = _snapshot(
        spark, args.data_dir, "test"
    )
    best_path = args.out / "tuning" / "best_validation.json"
    if not best_path.is_file():
        raise FileNotFoundError(f"Run tune first; missing selected config: {best_path}")
    with best_path.open(encoding="utf-8") as stream:
        best = json.load(stream)
    settings = ALSConfig.from_dict(best["parameters"])
    model_path = args.out / "models" / "als-test-final"
    model = train_als(spark, history, user_map, item_map, settings, model_path)
    _save_model_metadata(model_path, args, settings.__dict__)
    model_metadata = _load_model_metadata(model_path, args)
    popularity = fit_popularity_spark(history)
    als_rows, als_metrics, recommendations, metrics, stages = _recommend(
        model, history, users, held_out, user_map, item_map,
        popularity, config, "test",
    )
    model_config = settings.__dict__
    als_record = _metric_record(
        als_metrics, args, config, "als", model_path, model_config,
        model_metadata=model_metadata,
    )
    combined_record = _metric_record(
        metrics, args, config, "als_plus_popularity", model_path, model_config,
        "ALS first; fill remaining positions with unseen popularity",
        model_metadata,
    )
    combined_record["selected_validation_parameters"] = model_config
    combined_record["als_candidates_shared_with_als_only"] = True
    als_record["als_candidates_shared_with_als_only"] = True
    als_record["als_only_recommendations"] = "recommendations/als-only-test"
    combined_record["als_only_recommendations"] = "recommendations/als-only-test"
    cohort_started = perf_counter()
    als_cohorts = _cohort_record(
        als_rows, held_out, users, history, args, config, "als", model_path,
        model_config, model_metadata=model_metadata,
    )
    combined_cohorts = _cohort_record(
        recommendations, held_out, users, history, args, config,
        "als_plus_popularity", model_path, model_config,
        "ALS first; fill remaining positions with unseen popularity",
        model_metadata,
    )
    als_cohorts["als_candidates_shared_with_als_only"] = True
    als_cohorts["als_only_recommendations"] = "recommendations/als-only-test"
    combined_cohorts["als_candidates_shared_with_als_only"] = True
    combined_cohorts["als_only_recommendations"] = "recommendations/als-only-test"
    stages["metric_computation"] += perf_counter() - cohort_started
    write_started = perf_counter()
    try:
        _save_parquet(popularity, args.out / "popularity" / "test")
        _save_parquet(als_rows, args.out / "recommendations" / "als-only-test")
        _save_json(args.out / "metrics" / "als-only-test.json", als_record)
        _save_json(args.out / "metrics" / "als-only-cohorts-test.json", als_cohorts)
        _save_parquet(recommendations, args.out / "recommendations" / "als-test")
        _save_json(args.out / "metrics" / "als-test.json", combined_record)
        _save_json(args.out / "metrics" / "als-cohorts-test.json", combined_cohorts)
        print_top10_recommendations(als_rows, "ALS", int(config.get("top_k", 10)))
        print_top10_recommendations(
            recommendations, "ALS + Popularity", int(config.get("top_k", 10))
        )
        popularity_recommendations = popularity_fallback(
            users, history.select("user_id", "product_id"), popularity,
            int(config.get("top_k", 10)), "test",
            max(int(config.get("candidate_pool", 100)), int(config.get("top_k", 10))),
        )
        print_top10_recommendations(
            popularity_recommendations, "Popularity", int(config.get("top_k", 10))
        )
    finally:
        stages["output_writing"] = perf_counter() - write_started
        als_rows.unpersist()
        recommendations.unpersist()
    combined_record["runtime_seconds"] = stages
    _save_json(args.out / "metrics" / "als-test.json", combined_record)
    print(json.dumps(combined_record, indent=2))


def _run_all_validation(spark, args, config):
    if args.phase != "validation":
        raise ValueError("run-all only accepts the validation phase; Test is explicit")

    started = perf_counter()
    _popularity_action(spark, args, config)
    popularity_seconds = perf_counter() - started

    training_started = perf_counter()
    _train_action(spark, args, config)
    training_seconds = perf_counter() - training_started

    generation_started = perf_counter()
    _generate_action(spark, args, config)
    generation_seconds = perf_counter() - generation_started

    metric_paths = {
        "popularity": args.out / "metrics" / "popularity-validation.json",
        "als": args.out / "metrics" / "als-only-validation.json",
        "als_plus_popularity": args.out / "metrics" / "als-validation.json",
    }
    metrics = {}
    for method, path in metric_paths.items():
        with path.open(encoding="utf-8") as stream:
            metrics[method] = json.load(stream)

    cohort_paths = {
        "popularity": args.out / "metrics" / "popularity-cohorts-validation.json",
        "als": args.out / "metrics" / "als-only-cohorts-validation.json",
        "als_plus_popularity": args.out / "metrics" / "als-cohorts-validation.json",
    }
    cohorts = {}
    for method, path in cohort_paths.items():
        with path.open(encoding="utf-8") as stream:
            cohorts[method] = json.load(stream)

    summary = {
        "status": "PASS",
        "run_id": args.run_id,
        "phase": "validation",
        "data_dir": str(args.data_dir),
        "als_config": _als_config(config).__dict__,
        "model_path": str(args.out / "models" / "als-validation"),
        "mapping_paths": {
            "user": str(args.data_dir / "audit" / "validation" / "user_id_map.parquet"),
            "item": str(args.data_dir / "audit" / "validation" / "item_id_map.parquet"),
        },
        "mapping_validated": True,
        "top_k": int(config.get("top_k", 10)),
        "relevance_threshold": int(config.get("relevance_threshold", 4)),
        "als_model_reloaded_for_generation": True,
        "methods": metrics,
        "cohort_metrics": cohorts,
        "runtime_seconds": {
            "popularity_baseline": popularity_seconds,
            "als_training_and_save": training_seconds,
            "als_generation_and_evaluation": generation_seconds,
            "generation_stages": metrics["als_plus_popularity"].get(
                "runtime_seconds", {}
            ),
            "total": perf_counter() - started,
        },
        "outputs": {
            method: str(path.relative_to(args.out))
            for method, path in metric_paths.items()
        },
        "cohort_outputs": {
            method: str(path.relative_to(args.out))
            for method, path in cohort_paths.items()
        },
    }
    summary_path = args.out / "run-summary-validation.json"
    _save_json(summary_path, summary)

    print("\n" + "=" * 76)
    print("E5 VALIDATION - FINAL MODEL COMPARISON")
    print("=" * 76)
    print(
        f"{'Method':24s} {'Recall@K':>12s} {'NDCG@K':>12s} "
        f"{'Coverage >=1':>14s} {'Coverage =K':>14s}"
    )
    for method, label in (
        ("popularity", "Popularity"),
        ("als", "ALS"),
        ("als_plus_popularity", "ALS + Popularity"),
    ):
        result = metrics[method]
        print(
            f"{label:24s} "
            f"{result['recall_at_k']:>12.6f} "
            f"{result['ndcg_at_k']:>12.6f} "
            f"{result['user_coverage']:>14.6f} "
            f"{result['full_k_coverage']:>14.6f}"
        )
    print(
        f"Validation users: {metrics['als_plus_popularity']['evaluation_users']:,} | "
        f"Relevant users: {metrics['als_plus_popularity']['relevant_users']:,}"
    )
    print("ALS model: trained and saved, then reloaded for generation.")
    print(
        "Runtime (seconds): "
        f"Popularity={popularity_seconds:.2f}, "
        f"ALS train/save={training_seconds:.2f}, "
        f"generation/evaluation={generation_seconds:.2f}"
    )
    print(f"Summary: {summary_path}")
    print("=" * 76)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("popularity", "train", "generate", "run-all", "tune", "test-final"),
    )
    parser.add_argument("--phase", choices=("validation", "test"), default="validation")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--shuffle-partitions", type=int, default=24)
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
        help="Identifier shared by all metrics produced by this invocation",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = _read_config(args.config)
    spark = get_spark("E5-recommender", args.shuffle_partitions)
    try:
        actions = {
            "popularity": _popularity_action,
            "train": _train_action,
            "generate": _generate_action,
            "run-all": _run_all_validation,
            "tune": _tune_action,
            "test-final": _test_final_action,
        }
        actions[args.action](spark, args, config)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
