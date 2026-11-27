"""Manual, deterministic Top-K ranking metrics for E5 recommendations."""

import math

from pyspark.sql import functions as F


def ranking_metrics(recommendations, held_out, evaluation_users=None, k=10,
                    relevance_threshold=4):
    """Compute macro Recall/NDCG and user coverage at K using binary relevance."""
    if k < 1:
        raise ValueError("k must be positive")
    if not {"user_id", "product_id", "rank"}.issubset(recommendations.columns):
        raise ValueError("Recommendations require user_id, product_id, and rank")
    if not {"user_id", "product_id", "rating"}.issubset(held_out.columns):
        raise ValueError("Held-out ratings require user_id, product_id, and rating")
    ranked = recommendations.copy()
    if ranked.duplicated(["user_id", "product_id"]).any():
        raise ValueError("Recommendations contain duplicate user-item pairs")
    ranked = ranked[ranked["rank"] <= k]
    relevant = held_out.loc[
        held_out["rating"] >= relevance_threshold, ["user_id", "product_id"]
    ].drop_duplicates()
    user_ids = set(evaluation_users) if evaluation_users is not None else set(held_out.user_id)
    if not user_ids:
        raise ValueError("At least one evaluation user is required")
    relevant = relevant[relevant["user_id"].isin(user_ids)]
    relevant_by_user = relevant.groupby("user_id")["product_id"].agg(set).to_dict()
    recommended_by_user = (ranked.sort_values(["user_id", "rank"], kind="stable")
                           .groupby("user_id")["product_id"].agg(list).to_dict())

    recalls, ndcgs = [], []
    for user_id, items in relevant_by_user.items():
        if not items:
            continue
        observed = recommended_by_user.get(user_id, [])[:k]
        hits = [item in items for item in observed]
        recalls.append(sum(hits) / len(items))
        dcg = sum(
            1.0 / math.log2(position + 2)
            for position, hit in enumerate(hits) if hit
        )
        ideal_hits = min(len(items), k)
        idcg = sum(1.0 / math.log2(position + 2) for position in range(ideal_hits))
        ndcgs.append(dcg / idcg)

    recommended_users = set(ranked.user_id)
    total_users = len(user_ids)
    return {
        "k": k,
        "relevance_threshold": relevance_threshold,
        "evaluation_users": total_users,
        "relevant_users": len(recalls),
        "recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
        "ndcg_at_k": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "user_coverage": len(recommended_users & user_ids) / total_users,
        "full_k_coverage": sum(
            len(recommended_by_user.get(user_id, [])) >= k for user_id in user_ids
        ) / total_users,
    }


def ranking_metrics_spark(recommendations, held_out, evaluation_users, k=10,
                          relevance_threshold=4):
    """Distributed equivalent of ranking_metrics for E4-sized snapshots."""
    if k < 1:
        raise ValueError("k must be positive")
    relevant = (held_out.where(F.col("rating") >= relevance_threshold)
                .select("user_id", "product_id").distinct())
    relevant_counts = relevant.groupBy("user_id").agg(
        F.count("*").alias("relevant_count")
    )
    ranked = (recommendations.where(F.col("rank") <= k)
              .select("user_id", "product_id", "rank"))
    users = evaluation_users.select("user_id").distinct()
    hits = (ranked.join(relevant, ["user_id", "product_id"], "inner")
            .groupBy("user_id")
            .agg(
                F.count("*").alias("hit_count"),
                F.sum(1.0 / F.log2(F.col("rank") + F.lit(1.0))).alias("dcg"),
            ))
    ideal_sum = (
        f"aggregate(sequence(1, least(relevant_count, {int(k)})), "
        "cast(0.0 as double), (acc, pos) -> acc + 1.0 / log2(pos + 1))"
    )
    per_user = (relevant_counts.join(hits, "user_id", "left")
                .fillna({"hit_count": 0, "dcg": 0.0})
                .withColumn("recall", F.col("hit_count") / F.col("relevant_count"))
                .withColumn("idcg", F.expr(ideal_sum))
                .withColumn(
                    "ndcg",
                    F.when(F.col("idcg") > 0, F.col("dcg") / F.col("idcg"))
                    .otherwise(F.lit(0.0)),
                ))
    quality = (per_user.join(users, "user_id", "inner").agg(
        F.count("*").alias("relevant_users"),
        F.avg("recall").alias("recall_at_k"),
        F.avg("ndcg").alias("ndcg_at_k"),
    ).first())

    user_count = users.count()
    if user_count == 0:
        raise ValueError("At least one evaluation user is required")
    rec_counts = (ranked.groupBy("user_id").agg(F.count("*").alias("count"))
                  .join(users, "user_id", "inner"))
    coverage = rec_counts.agg(
        F.count(F.when(F.col("count") >= 1, 1)).alias("covered"),
        F.count(F.when(F.col("count") >= k, 1)).alias("full_k"),
    ).first()
    return {
        "k": k,
        "relevance_threshold": relevance_threshold,
        "evaluation_users": user_count,
        "relevant_users": int(quality.relevant_users),
        "recall_at_k": float(quality.recall_at_k or 0.0),
        "ndcg_at_k": float(quality.ndcg_at_k or 0.0),
        "user_coverage": coverage.covered / user_count,
        "full_k_coverage": coverage.full_k / user_count,
    }


def ranking_metrics_by_cohort_spark(recommendations, held_out, evaluation_users,
                                    history, k=10, relevance_threshold=4):
    """Evaluate overall, warm/cold-user, warm-item, and history-size cohorts."""
    if k < 1:
        raise ValueError("k must be positive")

    users = evaluation_users.select("user_id").distinct()
    relevant = (held_out.where(F.col("rating") >= relevance_threshold)
                .select("user_id", "product_id").distinct()
                .join(users, "user_id", "inner"))
    user_counts = history.groupBy("user_id").agg(F.count("*").alias("history_count"))
    warm_items = history.select("product_id").distinct()

    tagged_relevance = [
        relevant.withColumn("cohort", F.lit("all_relevant_users")),
        (relevant.join(user_counts, "user_id", "left_anti")
         .withColumn("cohort", F.lit("cold_users"))),
        (relevant.join(user_counts, "user_id", "inner")
         .where(F.col("history_count").between(1, 4))
         .select("user_id", "product_id")
         .withColumn("cohort", F.lit("history_1_4"))),
        (relevant.join(user_counts, "user_id", "inner")
         .where(F.col("history_count") >= 5)
         .select("user_id", "product_id")
         .withColumn("cohort", F.lit("history_ge_5"))),
        (relevant.join(user_counts, "user_id", "inner")
         .join(warm_items, "product_id", "inner")
         .select("user_id", "product_id")
         .withColumn("cohort", F.lit("warm_user_warm_item"))),
    ]
    relevant_by_cohort = tagged_relevance[0]
    for frame in tagged_relevance[1:]:
        relevant_by_cohort = relevant_by_cohort.unionByName(frame)
    relevant_by_cohort = relevant_by_cohort.distinct()

    relevant_counts = relevant_by_cohort.groupBy("cohort", "user_id").agg(
        F.count("*").alias("relevant_count")
    )
    ranked = (recommendations.where(F.col("rank") <= k)
              .select("user_id", "product_id", "rank"))
    hits = (ranked.join(relevant_by_cohort, ["user_id", "product_id"], "inner")
            .groupBy("cohort", "user_id")
            .agg(
                F.count("*").alias("hit_count"),
                F.sum(1.0 / F.log2(F.col("rank") + F.lit(1.0))).alias("dcg"),
            ))
    ideal_sum = (
        f"aggregate(sequence(1, least(relevant_count, {int(k)})), "
        "cast(0.0 as double), (acc, pos) -> acc + 1.0 / log2(pos + 1))"
    )
    per_user = (relevant_counts.join(hits, ["cohort", "user_id"], "left")
                .fillna({"hit_count": 0, "dcg": 0.0})
                .withColumn("recall", F.col("hit_count") / F.col("relevant_count"))
                .withColumn("idcg", F.expr(ideal_sum))
                .withColumn(
                    "ndcg",
                    F.when(F.col("idcg") > 0, F.col("dcg") / F.col("idcg"))
                    .otherwise(F.lit(0.0)),
                ))
    quality = per_user.groupBy("cohort").agg(
        F.count("*").alias("relevant_users"),
        F.avg("recall").alias("recall_at_k"),
        F.avg("ndcg").alias("ndcg_at_k"),
    )

    cohort_users = relevant_by_cohort.select("cohort", "user_id").distinct()
    rec_counts = (ranked.groupBy("user_id").agg(F.count("*").alias("count"))
                  .join(cohort_users, "user_id", "inner")
                  .groupBy("cohort", "user_id").agg(F.max("count").alias("count")))
    coverage = rec_counts.groupBy("cohort").agg(
        F.count(F.when(F.col("count") >= 1, 1)).alias("covered"),
        F.count(F.when(F.col("count") >= k, 1)).alias("full_k"),
    )
    user_counts_by_cohort = cohort_users.groupBy("cohort").agg(
        F.count("*").alias("evaluation_users")
    )
    rows = (quality.join(coverage, "cohort", "left")
            .join(user_counts_by_cohort, "cohort", "inner")
            .collect())
    result = {
        name: {
            "k": k,
            "relevance_threshold": relevance_threshold,
            "evaluation_users": 0,
            "relevant_users": 0,
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "user_coverage": 0.0,
            "full_k_coverage": 0.0,
        }
        for name in (
            "all_relevant_users", "warm_user_warm_item", "cold_users",
            "history_1_4", "history_ge_5",
        )
    }
    result.update({
        row.cohort: {
            "k": k,
            "relevance_threshold": relevance_threshold,
            "evaluation_users": int(row.evaluation_users),
            "relevant_users": int(row.relevant_users),
            "recall_at_k": float(row.recall_at_k or 0.0),
            "ndcg_at_k": float(row.ndcg_at_k or 0.0),
            "user_coverage": float(row.covered or 0) / int(row.evaluation_users),
            "full_k_coverage": float(row.full_k or 0) / int(row.evaluation_users),
        }
        for row in rows
    })
    return result
