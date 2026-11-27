"""Vectorised per-user Top-K metrics shared by T25 (Recall), T26 (NDCG) and T28.

Every metric is computed per user first and averaged afterwards (macro average), so the
same per-user table feeds bootstrap confidence intervals and paired significance tests.

Denominator rule (T18): only users with at least one relevant target are scored.
A scored user that received no recommendation gets 0 for every quality metric.
"""

import math

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window

QUALITY_COLUMNS = [
    "recall", "precision", "hit", "rr", "ap", "ndcg", "ndcg_graded",
]
PER_USER_COLUMNS = [
    "user_id", "relevant_count", "list_length", "hits", "recall", "precision", "hit",
    "rr", "ap", "dcg", "idcg", "ndcg", "dcg_graded", "idcg_graded", "ndcg_graded",
]


def discount(position):
    """Log2 position discount, positions start at 1."""
    return 1.0 / math.log2(position + 1)


def ideal_binary_dcg(relevant_count, k):
    """IDCG for binary relevance: all min(|R|, K) slots filled with hits."""
    return sum(discount(position) for position in range(1, min(relevant_count, k) + 1))


def graded_gain(gain_level):
    """Exponential gain 2^level - 1: 4 stars -> 1, 5 stars -> 3."""
    return float(2 ** int(gain_level) - 1)


def top_k_recommendations(recommendations, k):
    """Validate a recommendation table and keep ranks 1..K (pandas)."""
    if k < 1:
        raise ValueError("k must be positive")
    missing = {"user_id", "product_id", "rank"} - set(recommendations.columns)
    if missing:
        raise ValueError(f"Recommendations are missing columns: {sorted(missing)}")
    frame = recommendations[["user_id", "product_id", "rank"]]
    if frame.duplicated(["user_id", "product_id"]).any():
        raise ValueError("Recommendations contain duplicate user-item pairs")
    if frame.duplicated(["user_id", "rank"]).any():
        raise ValueError("Recommendations contain duplicate ranks for a user")
    if (frame["rank"] < 1).any():
        raise ValueError("Ranks must start at 1")
    return frame[frame["rank"] <= k]


def per_user_ranking_metrics(recommendations, ground_truth, evaluation_users, k=10):
    """Return one row per scored user with Recall/Precision/Hit/RR/AP/NDCG at K (pandas)."""
    users = pd.Index(pd.unique(pd.Series(list(evaluation_users), dtype="object")))
    if len(users) == 0:
        raise ValueError("At least one evaluation user is required")
    top = top_k_recommendations(recommendations, k)
    top = top[top["user_id"].isin(users)]
    truth = ground_truth[ground_truth["user_id"].isin(users)]
    if truth.duplicated(["user_id", "product_id"]).any():
        raise ValueError("Ground truth contains duplicate user-item targets")

    scored = truth.groupby("user_id").size().rename("relevant_count").to_frame()
    if scored.empty:
        return pd.DataFrame(columns=PER_USER_COLUMNS)

    lengths = top.groupby("user_id").size().rename("list_length")
    hits = top.merge(truth[["user_id", "product_id", "gain_level"]],
                     on=["user_id", "product_id"], how="inner")
    hits = hits.sort_values(["user_id", "rank"], kind="stable")
    hits["disc"] = 1.0 / np.log2(hits["rank"].astype(float) + 1.0)
    hits["graded"] = (np.power(2.0, hits["gain_level"].astype(float)) - 1.0) * hits["disc"]
    # note: precision at the position of each hit, summed for average precision
    hits["prec_at_hit"] = (hits.groupby("user_id").cumcount() + 1) / hits["rank"]
    hit_stats = hits.groupby("user_id").agg(
        hits=("product_id", "size"),
        dcg=("disc", "sum"),
        dcg_graded=("graded", "sum"),
        first_hit=("rank", "min"),
        ap_sum=("prec_at_hit", "sum"),
    )

    ideal = truth[["user_id", "gain_level"]].copy()
    ideal["gain"] = np.power(2.0, ideal["gain_level"].astype(float)) - 1.0
    ideal = ideal.sort_values(["user_id", "gain"], ascending=[True, False], kind="stable")
    ideal["position"] = ideal.groupby("user_id").cumcount() + 1
    ideal = ideal[ideal["position"] <= k]
    ideal["value"] = ideal["gain"] / np.log2(ideal["position"].astype(float) + 1.0)
    idcg_graded = ideal.groupby("user_id")["value"].sum().rename("idcg_graded")

    table = (scored.join(lengths, how="left").join(hit_stats, how="left")
             .join(idcg_graded, how="left"))
    table = table.fillna({"list_length": 0, "hits": 0, "dcg": 0.0, "dcg_graded": 0.0,
                          "ap_sum": 0.0})
    prefix = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(1, k + 1) + 1.0))])
    capped = np.minimum(table["relevant_count"].to_numpy(), k)
    table["idcg"] = prefix[capped]
    table["recall"] = table["hits"] / table["relevant_count"]
    table["precision"] = table["hits"] / k
    table["hit"] = (table["hits"] > 0).astype(float)
    table["rr"] = np.where(table["first_hit"].notna(), 1.0 / table["first_hit"], 0.0)
    table["ap"] = table["ap_sum"] / capped
    table["ndcg"] = table["dcg"] / table["idcg"]
    table["ndcg_graded"] = table["dcg_graded"] / table["idcg_graded"]
    table = table.reset_index().rename(columns={"index": "user_id"})
    table["list_length"] = table["list_length"].astype("int64")
    table["hits"] = table["hits"].astype("int64")
    return table[PER_USER_COLUMNS].sort_values("user_id", kind="stable").reset_index(drop=True)


def macro_average(per_user, columns=QUALITY_COLUMNS):
    """Mean of every quality column over scored users."""
    result = {"scored_users": int(len(per_user))}
    for column in columns:
        result[column] = float(per_user[column].mean()) if len(per_user) else 0.0
    return result


def per_user_ranking_metrics_spark(recommendations, ground_truth, evaluation_users, k=10):
    """Distributed per-user table with the same columns and definitions as the pandas one."""
    if k < 1:
        raise ValueError("k must be positive")
    users = evaluation_users.select("user_id").distinct()
    truth = ground_truth.join(users, "user_id", "inner")
    top = (recommendations.where(F.col("rank") <= k)
           .select("user_id", "product_id", "rank")
           .join(users, "user_id", "inner"))
    lengths = top.groupBy("user_id").agg(F.count("*").alias("list_length"))
    relevant_counts = truth.groupBy("user_id").agg(F.count("*").alias("relevant_count"))

    by_rank = Window.partitionBy("user_id").orderBy("rank")
    disc = 1.0 / F.log2(F.col("rank") + F.lit(1.0))
    hits = (top.join(truth.select("user_id", "product_id", "gain_level"),
                     ["user_id", "product_id"], "inner")
            .withColumn("cum_hits", F.row_number().over(by_rank))
            .groupBy("user_id")
            .agg(
                F.count("*").alias("hits"),
                F.sum(disc).alias("dcg"),
                F.sum((F.pow(F.lit(2.0), F.col("gain_level")) - 1.0) * disc).alias("dcg_graded"),
                F.min("rank").alias("first_hit"),
                F.sum(F.col("cum_hits") / F.col("rank")).alias("ap_sum"),
            ))
    by_gain = Window.partitionBy("user_id").orderBy(F.desc("gain"), F.asc("product_id"))
    idcg_graded = (truth.withColumn("gain", F.pow(F.lit(2.0), F.col("gain_level")) - 1.0)
                   .withColumn("position", F.row_number().over(by_gain))
                   .where(F.col("position") <= k)
                   .groupBy("user_id")
                   .agg(F.sum(F.col("gain") / F.log2(F.col("position") + F.lit(1.0)))
                        .alias("idcg_graded")))
    ideal_sum = (
        f"aggregate(sequence(1, least(relevant_count, {int(k)})), "
        "cast(0.0 as double), (acc, pos) -> acc + 1.0 / log2(pos + 1))"
    )
    capped = F.least(F.col("relevant_count"), F.lit(int(k)))
    return (relevant_counts.join(lengths, "user_id", "left")
            .join(hits, "user_id", "left")
            .join(idcg_graded, "user_id", "left")
            .fillna({"list_length": 0, "hits": 0, "dcg": 0.0, "dcg_graded": 0.0,
                     "ap_sum": 0.0})
            .withColumn("idcg", F.expr(ideal_sum))
            .withColumn("recall", F.col("hits") / F.col("relevant_count"))
            .withColumn("precision", F.col("hits") / F.lit(float(k)))
            .withColumn("hit", (F.col("hits") > 0).cast("double"))
            .withColumn("rr", F.when(F.col("first_hit").isNotNull(), 1.0 / F.col("first_hit"))
                        .otherwise(F.lit(0.0)))
            .withColumn("ap", F.col("ap_sum") / capped)
            .withColumn("ndcg", F.col("dcg") / F.col("idcg"))
            .withColumn("ndcg_graded", F.col("dcg_graded") / F.col("idcg_graded"))
            .select(*PER_USER_COLUMNS))
