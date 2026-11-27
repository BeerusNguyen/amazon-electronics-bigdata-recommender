"""T25 Recall@K: share of a user's relevant held-out items retrieved in the Top-K list.

Recall@K(u) = |TopK(u) ∩ R(u)| / |R(u)|,  macro Recall@K = mean over users with |R(u)| > 0.

Rules frozen in T18 and applied here:
  users with no relevant target are excluded from the denominator (returned as None)
  users with relevant targets but no recommendation score 0, they are never dropped
  the full |R(u)| is the denominator even when |R(u)| > K, so Recall@10 can be < 1 for a
  perfect list; NDCG@K (T26) normalises by min(|R(u)|, K) instead
"""

from src.evaluation.per_user_metrics import (
    macro_average,
    per_user_ranking_metrics,
    per_user_ranking_metrics_spark,
)


def recall_at_k_user(recommended, relevant, k=10):
    """Reference implementation for one user; recommended is an ordered list."""
    if k < 1:
        raise ValueError("k must be positive")
    relevant = set(relevant)
    if not relevant:
        return None
    top = list(recommended)[:k]
    if len(set(top)) != len(top):
        raise ValueError("Recommended list contains duplicates")
    return sum(item in relevant for item in top) / len(relevant)


def per_user_recall(recommendations, ground_truth, evaluation_users, k=10):
    """Per-user recall table (pandas): user_id, relevant_count, hits, recall."""
    table = per_user_ranking_metrics(recommendations, ground_truth, evaluation_users, k)
    return table[["user_id", "relevant_count", "hits", "recall"]]


def recall_at_k(recommendations, ground_truth, evaluation_users, k=10):
    """Macro Recall@K and the number of scored users (pandas)."""
    table = per_user_ranking_metrics(recommendations, ground_truth, evaluation_users, k)
    summary = macro_average(table, ["recall"])
    return {"k": k, "scored_users": summary["scored_users"], "recall_at_k": summary["recall"]}


def per_user_recall_spark(recommendations, ground_truth, evaluation_users, k=10):
    """Distributed per-user recall with the same definition."""
    return per_user_ranking_metrics_spark(
        recommendations, ground_truth, evaluation_users, k
    ).select("user_id", "relevant_count", "hits", "recall")
