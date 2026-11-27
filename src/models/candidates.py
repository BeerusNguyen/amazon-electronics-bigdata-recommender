"""ALS candidate scoring and final recommendation assembly."""

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.models.popularity import RECOMMENDATION_COLUMNS


def generate_als_raw_candidates(model, users, user_map, item_map, k=10,
                                candidate_multiplier=5):
    """Generate scored ALS candidates and map their IDs back to source values."""
    if k < 1 or candidate_multiplier < 1:
        raise ValueError("k and candidate_multiplier must be positive")
    target_users = users.select("user_id").distinct()
    warm_user_indices = (target_users.join(user_map, "user_id", "inner")
                         .select("user_idx").distinct())
    recommendations = model.recommendForUserSubset(
        warm_user_indices, int(k * candidate_multiplier)
    )
    exploded = (recommendations
                .select("user_idx", F.explode("recommendations").alias("candidate"))
                .select(
                    "user_idx",
                    F.col("candidate.item_idx").alias("item_idx"),
                    F.col("candidate.rating").cast("double").alias("score"),
                )
                .join(user_map.select("user_idx", "user_id"), "user_idx")
                .join(item_map.select("item_idx", "product_id"), "item_idx")
                .select("user_id", "product_id", "score"))
    return exploded


def rank_als_candidates(candidates, seen, k=10, phase="validation"):
    """Remove seen products and rank the remaining ALS candidates."""
    if k < 1:
        raise ValueError("k must be positive")
    candidates = candidates.join(
        seen.select("user_id", "product_id").distinct(),
        ["user_id", "product_id"], "left_anti",
    )
    ranking = Window.partitionBy("user_id").orderBy(
        F.desc("score"), F.asc("product_id")
    )
    return (candidates.withColumn("rank", F.row_number().over(ranking))
            .where(F.col("rank") <= k)
            .withColumn("item_source", F.lit("als"))
            .withColumn("recommendation_source", F.lit("als"))
            .withColumn("phase", F.lit(phase))
            .select(*RECOMMENDATION_COLUMNS))


def generate_als_candidates(model, users, seen, user_map, item_map, k=10,
                            candidate_multiplier=5, phase="validation"):
    """Score warm users using E4 indices and remove products already seen."""
    candidates = generate_als_raw_candidates(
        model, users, user_map, item_map, k, candidate_multiplier
    )
    return rank_als_candidates(candidates, seen, k, phase)


def merge_with_fallback_pandas(users, als_recommendations, fallback_recommendations,
                               k=10, phase="validation"):
    """Prefer ALS rows and fill remaining positions with unseen popularity rows."""
    if k < 1:
        raise ValueError("k must be positive")
    frames = []
    for recommendations in (als_recommendations, fallback_recommendations):
        frame = recommendations.copy()
        if not frame.empty:
            missing = set(RECOMMENDATION_COLUMNS) - set(frame.columns)
            if missing:
                raise ValueError(f"Recommendations are missing columns: {sorted(missing)}")
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)

    candidates = pd.concat(frames, ignore_index=True)
    candidates = candidates[candidates.user_id.isin(set(users))]
    if candidates.empty:
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)
    if candidates.duplicated(["user_id", "product_id"]).any():
        candidates = candidates.drop_duplicates(["user_id", "product_id"], keep="first")
    candidates["_source_priority"] = candidates.recommendation_source.map(
        {"als": 0, "popularity": 1}
    ).fillna(2)
    candidates = candidates.sort_values(
        ["user_id", "_source_priority", "rank", "score", "product_id"],
        ascending=[True, True, True, False, True],
        kind="stable",
    )
    result = candidates.groupby("user_id", sort=False).head(k).copy()
    result["rank"] = result.groupby("user_id", sort=False).cumcount() + 1
    result["phase"] = phase
    result.drop(columns="_source_priority", inplace=True)
    return result[RECOMMENDATION_COLUMNS].reset_index(drop=True)


def assemble_recommendations(als_recommendations, fallback_recommendations,
                             k=10, phase="validation"):
    """Combine Spark candidate frames with ALS precedence and contiguous ranks."""
    if k < 1:
        raise ValueError("k must be positive")
    als = als_recommendations
    fallback = fallback_recommendations
    existing = als.select("user_id", "product_id").distinct()
    fallback = fallback.join(existing, ["user_id", "product_id"], "left_anti")
    candidates = als.unionByName(fallback)
    priority = F.when(F.col("recommendation_source") == "als", 0).otherwise(1)
    ordering = Window.partitionBy("user_id").orderBy(
        priority.asc(), F.col("rank").asc(), F.col("score").desc(),
        F.col("product_id").asc(),
    )
    return (candidates.withColumn("phase", F.lit(phase))
            .withColumn("rank", F.row_number().over(ordering))
            .where(F.col("rank") <= k)
            .select(*RECOMMENDATION_COLUMNS))
