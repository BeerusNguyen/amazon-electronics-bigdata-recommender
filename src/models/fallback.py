"""Popularity fallback for warm users with short ALS lists and cold users."""

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.models.popularity import RECOMMENDATION_COLUMNS


def popularity_fallback(users, seen, popularity, k=10, phase="validation",
                        candidate_pool=1000):
    """Create bounded popularity candidates for all target users, excluding history."""
    if k < 1:
        raise ValueError("k must be positive")
    if candidate_pool < k:
        raise ValueError("candidate_pool must be at least k")
    popular_items = (popularity.orderBy(F.desc("score"), F.asc("product_id"))
                     .limit(candidate_pool)
                     .select("product_id", F.col("score").cast("double").alias("score")))
    candidates = (users.select("user_id").distinct().crossJoin(popular_items)
                  .join(seen.select("user_id", "product_id").distinct(),
                        ["user_id", "product_id"], "left_anti"))
    ranking = Window.partitionBy("user_id").orderBy(
        F.desc("score"), F.asc("product_id")
    )
    return (candidates.withColumn("rank", F.row_number().over(ranking))
            .where(F.col("rank") <= k)
            .withColumn("item_source", F.lit("popularity"))
            .withColumn("recommendation_source", F.lit("popularity"))
            .withColumn("phase", F.lit(phase))
            .select(*RECOMMENDATION_COLUMNS))
