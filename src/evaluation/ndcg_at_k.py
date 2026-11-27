"""T26 NDCG@K: ranking quality of the Top-K list, rewarding relevant items placed early.

DCG@K(u)  = sum over positions i = 1..K of gain(i) / log2(i + 1)
IDCG@K(u) = DCG of the ideal ordering of u's targets, truncated at K
NDCG@K(u) = DCG@K(u) / IDCG@K(u)

Binary (primary, frozen in T18): gain = 1 for a relevant item (rating >= 4), else 0.
Graded (secondary, added in E6): gain = 2^level - 1 with level = rating - 3, so a 5-star
target is worth 3 and a 4-star target is worth 1. It is reported next to binary NDCG and is
never used to select a model.

Same candidate and denominator rules as Recall@K (T25): positions are never compressed
after removing non-relevant items, users without targets are excluded, users with targets
but no recommendations score 0.
"""

import math

from src.evaluation.per_user_metrics import (
    macro_average,
    per_user_ranking_metrics,
    per_user_ranking_metrics_spark,
)


def dcg_at_k(gains, k=10):
    """DCG of an ordered gain vector, positions start at 1."""
    return sum(gain / math.log2(position + 2) for position, gain in enumerate(list(gains)[:k]))


def ndcg_at_k_user(recommended, relevance, k=10, graded=False):
    """Reference implementation for one user.

    relevance: dict item -> gain level (1 for binary use), or an iterable of relevant items.
    Returns None when the user has no relevant item.
    """
    if k < 1:
        raise ValueError("k must be positive")
    if not isinstance(relevance, dict):
        relevance = {item: 1 for item in relevance}
    if not relevance:
        return None
    top = list(recommended)[:k]
    if len(set(top)) != len(top):
        raise ValueError("Recommended list contains duplicates")

    def gain(level):
        return float(2 ** int(level) - 1) if graded else 1.0

    observed = [gain(relevance[item]) if item in relevance else 0.0 for item in top]
    ideal = sorted((gain(level) for level in relevance.values()), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(observed, k) / idcg if idcg > 0 else 0.0


def per_user_ndcg(recommendations, ground_truth, evaluation_users, k=10):
    """Per-user binary and graded NDCG (pandas)."""
    table = per_user_ranking_metrics(recommendations, ground_truth, evaluation_users, k)
    return table[["user_id", "relevant_count", "dcg", "idcg", "ndcg",
                  "dcg_graded", "idcg_graded", "ndcg_graded"]]


def ndcg_at_k(recommendations, ground_truth, evaluation_users, k=10):
    """Macro binary and graded NDCG@K (pandas)."""
    table = per_user_ranking_metrics(recommendations, ground_truth, evaluation_users, k)
    summary = macro_average(table, ["ndcg", "ndcg_graded"])
    return {"k": k, "scored_users": summary["scored_users"], "ndcg_at_k": summary["ndcg"],
            "ndcg_graded_at_k": summary["ndcg_graded"]}


def per_user_ndcg_spark(recommendations, ground_truth, evaluation_users, k=10):
    """Distributed per-user NDCG with the same definition."""
    return per_user_ranking_metrics_spark(
        recommendations, ground_truth, evaluation_users, k
    ).select("user_id", "relevant_count", "dcg", "idcg", "ndcg",
             "dcg_graded", "idcg_graded", "ndcg_graded")
