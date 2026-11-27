"""Deterministic popularity baselines for pandas fixtures and Spark jobs."""

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def compute_popularity(history):
    """Return positive counts for every catalog item, then sort deterministically."""
    required = {"product_id", "rating"}
    if not required.issubset(history.columns):
        raise ValueError("History must contain product_id and rating")
    if history["product_id"].isna().any():
        raise ValueError("History contains null product_id values")
    catalog = history[["product_id"]].drop_duplicates()
    positive = history.loc[history["rating"] >= 4, ["product_id"]].copy()
    positive["_positive"] = 1
    counts = positive.groupby("product_id", as_index=False)["_positive"].sum()
    counts.rename(columns={"_positive": "popularity_count"}, inplace=True)
    result = catalog.merge(counts, on="product_id", how="left")
    result["popularity_count"] = result["popularity_count"].fillna(0).astype("int64")
    result["score"] = result["popularity_count"].astype(float)
    return result.sort_values(
        ["score", "product_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def fit_popularity_spark(history):
    """Count positive ratings while retaining every item in the history catalog."""
    if not {"product_id", "rating"}.issubset(history.columns):
        raise ValueError("History must contain product_id and rating")
    catalog = history.select("product_id").distinct()
    counts = (history.where(F.col("rating") >= F.lit(4))
              .groupBy("product_id")
              .agg(F.count("*").alias("popularity_count")))
    return (catalog.join(counts, "product_id", "left")
            .fillna({"popularity_count": 0})
            .withColumn("score", F.col("popularity_count").cast("double"))
            .select("product_id", "popularity_count", "score"))


def recommend_popular_pandas(users, history, popularity, k=10, phase="validation"):
    """Generate an exact, small-data popularity recommendation fixture."""
    if k < 1:
        raise ValueError("k must be positive")
    if not {"user_id", "product_id"}.issubset(history.columns):
        raise ValueError("History must contain user_id and product_id")
    required = {"product_id", "score"}
    if not required.issubset(popularity.columns):
        raise ValueError("Popularity table must contain product_id and score")
    seen = history.groupby("user_id")["product_id"].agg(set).to_dict()
    ordered = popularity.sort_values(
        ["score", "product_id"], ascending=[False, True], kind="stable"
    )
    rows = []
    for user_id in sorted(set(users)):
        rank = 0
        for item, score in ordered[["product_id", "score"]].itertuples(index=False, name=None):
            if item in seen.get(user_id, set()):
                continue
            rank += 1
            rows.append({
                "user_id": user_id,
                "product_id": item,
                "rank": rank,
                "score": float(score),
                "item_source": "popularity",
                "recommendation_source": "popularity",
                "phase": phase,
            })
            if rank == k:
                break
    return pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)


RECOMMENDATION_COLUMNS = [
    "user_id", "product_id", "rank", "score", "item_source",
    "recommendation_source", "phase",
]
