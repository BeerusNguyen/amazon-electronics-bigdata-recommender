"""Why ALS ranks the way it does: diagnostics that need the E5 model or a Spark retrain.

All outputs go to results/e6/diagnostics/. Nothing here reads Test to choose a model:
seed stability and the ablation run on Validation only, and the ablation is diagnostic.

rating_prediction   RMSE/MAE of the saved ALS models on warm held-out pairs vs constant baselines
item_support        history popularity of the items each method ranks
tuning_table        the E5 Validation grid, copied from outputs/e5/tuning
seed_stability      locked config, several seeds: how large is seed noise
ablation            locked config with larger regParam and with implicitPrefs=True
"""

import json
import logging
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.evaluation.per_user_metrics import per_user_ranking_metrics
from src.evaluation.protocol import build_ground_truth, load_snapshot

MODEL_DIRS = {"validation": "als-validation", "test": "als-test-final"}
REC_DIRS = {"popularity": "popularity-{phase}", "als": "als-only-{phase}"}


def read_parquet_dir(path):
    path = Path(path)
    if path.is_file():
        return pd.read_parquet(path)
    files = sorted(p for p in path.rglob("*.parquet") if not p.name.startswith((".", "_")))
    if not files:
        raise FileNotFoundError(f"No parquet files under {path}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def spark_session(name):
    """Spark session for E6 jobs that move frames between Spark and pandas."""
    from src.common.spark_utils import get_spark
    spark = get_spark(name, 8)
    # note: Arrow 12 bundled with Spark 3.5 fails on JDK 21; plain conversion works on 17 and 21
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
    return spark


def rating_prediction(data_dir, e5_dir, phase):
    """ALS rating error on warm user-warm item held-out pairs, next to constant predictors."""
    from src.models.als import load_als

    history, held_out = load_snapshot(data_dir, phase)
    map_dir = Path(data_dir) / "audit" / phase
    warm = (held_out.merge(pd.read_parquet(map_dir / "user_id_map.parquet"), on="user_id")
            .merge(pd.read_parquet(map_dir / "item_id_map.parquet"), on="product_id"))
    spark = spark_session("E6-rating-prediction")
    try:
        model = load_als(Path(e5_dir) / "models" / MODEL_DIRS[phase])
        scored = model.transform(spark.createDataFrame(
            warm[["user_idx", "item_idx", "rating"]].astype(
                {"user_idx": "int32", "item_idx": "int32", "rating": "float32"}))).toPandas()
    finally:
        spark.stop()
    frame = scored.merge(warm[["user_idx", "item_idx", "user_id", "product_id"]],
                         on=["user_idx", "item_idx"])
    predictors = {
        "als_clipped_1_5": frame["prediction"].clip(1, 5),
        "als_raw": frame["prediction"],
        "global_mean": pd.Series(history["rating"].mean(), index=frame.index),
        "item_mean": frame["product_id"].map(history.groupby("product_id")["rating"].mean()),
        "user_mean": frame["user_id"].map(history.groupby("user_id")["rating"].mean()),
    }
    return pd.DataFrame([{"phase": phase, "predictor": name, "pairs": len(frame),
                          "rmse": float(np.sqrt(np.mean((value - frame["rating"]) ** 2))),
                          "mae": float(np.mean(np.abs(value - frame["rating"])))}
                         for name, value in predictors.items()])


def item_support(data_dir, e5_dir, phase, k=10, relevance_threshold=4):
    """How popular, in history, are the items Popularity and ALS put in the Top-K."""
    history, _ = load_snapshot(data_dir, phase)
    interactions = history.groupby("product_id").size()
    positives = history[history["rating"] >= relevance_threshold].groupby("product_id").size()
    rows = []
    for method, folder in REC_DIRS.items():
        recs = read_parquet_dir(Path(e5_dir) / "recommendations" / folder.format(phase=phase))
        recs = recs[recs["rank"] <= k]
        support = recs["product_id"].map(interactions)
        rows.append({
            "phase": phase, "method": method, "slots": len(recs),
            "median_item_history_ratings": float(support.median()),
            "share_items_with_le_3_ratings": float((support <= 3).mean()),
            "mean_item_positive_count": float(recs["product_id"].map(positives).fillna(0).mean()),
            "mean_score": float(recs["score"].mean()),
        })
    return pd.DataFrame(rows)


def tuning_table(e5_dir):
    """The E5 Validation grid with the selected trial flagged."""
    tuning = Path(e5_dir) / "tuning"
    results = json.loads((tuning / "validation_results.json").read_text())
    best = json.loads((tuning / "best_validation.json").read_text())["parameters"]
    return pd.DataFrame([{"trial": i, **{key: r["parameters"][key] for key in
                                         ("rank", "max_iter", "reg_param", "seed", "implicit_prefs")},
                          "recall_at_k": r["recall_at_k"], "ndcg_at_k": r["ndcg_at_k"],
                          "selected": r["parameters"] == best}
                         for i, r in enumerate(results, start=1)])


def locked_parameters(e5_dir):
    return dict(json.loads((Path(e5_dir) / "tuning" / "best_validation.json").read_text())
                ["parameters"])


def evaluate_variants(data_dir, config_path, variants, scratch, k=10, relevance_threshold=4):
    """Retrain ALS variants on Validation only and score ALS and ALS + Popularity."""
    from pyspark.sql import functions as F

    from src.common.spark_utils import read_parquet
    from src.models.als import ALSConfig, train_als
    from src.models.candidates import (
        assemble_recommendations,
        generate_als_raw_candidates,
        rank_als_candidates,
    )
    from src.models.fallback import popularity_fallback
    from src.models.popularity import fit_popularity_spark

    data_dir = Path(data_dir)
    config = json.loads(Path(config_path).read_text())
    history_pd, held_out_pd = load_snapshot(data_dir, "validation")
    users_pd = held_out_pd["user_id"].unique()
    ground_truth = build_ground_truth(held_out_pd, history_pd, relevance_threshold)
    spark = spark_session("E6-als-variants")
    rows = []
    try:
        history = read_parquet(spark, str(data_dir / "train.parquet"))
        held_out = read_parquet(spark, str(data_dir / "validation.parquet"))
        users = held_out.select("user_id").distinct()
        map_dir = data_dir / "audit" / "validation"
        user_map = read_parquet(spark, str(map_dir / "user_id_map.parquet"))
        item_map = read_parquet(spark, str(map_dir / "item_id_map.parquet"))
        seen = history.select("user_id", "product_id")
        fallback = popularity_fallback(users, seen, fit_popularity_spark(history), k, "validation",
                                       max(int(config.get("candidate_pool", 100)), k)).persist()
        warm_pairs = (held_out.join(user_map, "user_id").join(item_map, "product_id")
                      .select("user_idx", "item_idx", "rating")).persist()
        for label, parameters in variants:
            started = perf_counter()
            model = train_als(spark, history, user_map, item_map, ALSConfig.from_dict(parameters))
            train_seconds = perf_counter() - started
            clipped = F.least(F.greatest(F.col("prediction"), F.lit(1.0)), F.lit(5.0))
            error = (model.transform(warm_pairs)
                     .agg(F.sqrt(F.avg((clipped - F.col("rating")) ** 2)).alias("rmse"),
                          F.avg(F.abs(F.col("prediction"))).alias("mean_abs_prediction"))
                     .first())
            als_rows = rank_als_candidates(
                generate_als_raw_candidates(model, users, user_map, item_map, k,
                                            int(config.get("candidate_multiplier", 5))),
                seen, k, "validation").persist()
            combined = assemble_recommendations(als_rows, fallback, k, "validation")
            for method, frame in (("als", als_rows), ("als_plus_popularity", combined)):
                # note: spill through Parquet instead of toPandas to keep driver memory flat
                path = Path(scratch) / method
                frame.select("user_id", "product_id", "rank").write.mode("overwrite").parquet(str(path))
                table = per_user_ranking_metrics(read_parquet_dir(path), ground_truth, users_pd, k)
                rows.append({"variant": label, "method": method, **parameters,
                             "train_seconds": train_seconds,
                             "warm_pair_rmse_clipped": float(error.rmse),
                             "mean_abs_prediction": float(error.mean_abs_prediction),
                             "scored_users": len(table),
                             "recall": float(table["recall"].mean()),
                             "ndcg": float(table["ndcg"].mean()),
                             "hit": float(table["hit"].mean())})
                logging.info("%s %s recall=%.6f ndcg=%.6f", label, method,
                             rows[-1]["recall"], rows[-1]["ndcg"])
            als_rows.unpersist()
    finally:
        spark.stop()
        shutil.rmtree(scratch, ignore_errors=True)
    return pd.DataFrame(rows)


def seed_variants(e5_dir, seeds):
    locked = locked_parameters(e5_dir)
    return [(f"seed={seed}", dict(locked, seed=int(seed))) for seed in seeds]


def ablation_variants(e5_dir):
    """Is weak ALS caused by regularisation or by the explicit-rating objective?"""
    locked = locked_parameters(e5_dir)
    return [("explicit (locked E5 config)", dict(locked)),
            ("explicit regParam=0.5", dict(locked, reg_param=0.5)),
            ("explicit regParam=1.0", dict(locked, reg_param=1.0)),
            ("implicit alpha=1", dict(locked, implicit_prefs=True, alpha=1.0)),
            ("implicit alpha=10", dict(locked, implicit_prefs=True, alpha=10.0))]
