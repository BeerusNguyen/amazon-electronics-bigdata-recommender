"""T27 user coverage and fallback effectiveness (catalog coverage lives in beyond_accuracy.py).

User coverage uses every target user of the held-out period as denominator, including users
with no relevant target:
  coverage_ge1   share of target users with at least one recommendation
  coverage_full  share of target users with exactly K recommendations
"""

import numpy as np
import pandas as pd

from src.evaluation.per_user_metrics import top_k_recommendations

SEGMENTS = ["all", "warm_user", "cold_user", "history_1_4", "history_ge_5"]
STATUSES = ["full_list", "short_list", "known_unserved", "unknown_user"]
# note: the hybrid list of a user is all-ALS, mixed or all-Popularity depending on ALS status
HYBRID_SOURCE = {"full_list": "ALS", "short_list": "Mixed", "known_unserved": "Popularity",
                 "unknown_user": "Popularity"}


def list_lengths(recommendations, evaluation_users, k=10):
    """Top-K list length per target user, 0 when the user has no list."""
    lengths = top_k_recommendations(recommendations, k).groupby("user_id").size()
    users = pd.Series(pd.unique(pd.Series(list(evaluation_users), dtype="object")))
    return pd.DataFrame({"user_id": users,
                         "list_length": users.map(lengths).fillna(0).astype("int64")})


def _with_history(evaluation_users, history):
    counts = history.groupby("user_id").size()
    frame = pd.DataFrame({"user_id": pd.unique(pd.Series(list(evaluation_users), dtype="object"))})
    frame["history_count"] = frame["user_id"].map(counts).fillna(0).astype("int64")
    return frame


def _summary(lengths, k):
    total = len(lengths)
    covered = int((lengths >= 1).sum())
    full = int((lengths >= k).sum())
    return {"users": total, "coverage_ge1": covered / total if total else 0.0,
            "coverage_full_k": full / total if total else 0.0,
            "users_zero": total - covered, "users_short": covered - full,
            "mean_list_length": float(lengths.mean()) if total else 0.0}


def coverage_by_segment(recommendations, evaluation_users, history, k=10):
    """User coverage for all target users, warm vs cold users and history-length buckets."""
    frame = _with_history(evaluation_users, history).merge(
        list_lengths(recommendations, evaluation_users, k), on="user_id")
    count = frame["history_count"]
    masks = {"all": count >= 0, "warm_user": count > 0, "cold_user": count == 0,
             "history_1_4": count.between(1, 4), "history_ge_5": count >= 5}
    return pd.DataFrame([{"segment": name, **_summary(frame.loc[masks[name], "list_length"], k)}
                         for name in SEGMENTS])


def serving_status(als_only, evaluation_users, history, k=10):
    """ALS status of each target user before fallback."""
    frame = _with_history(evaluation_users, history).merge(
        list_lengths(als_only, evaluation_users, k), on="user_id")
    frame["status"] = np.select(
        [frame["history_count"] == 0, frame["list_length"] == 0, frame["list_length"] < k],
        ["unknown_user", "known_unserved", "short_list"], default="full_list")
    return frame


def fallback_effectiveness(als_only, combined, evaluation_users, history, k=10,
                           combined_per_user=None):
    """Coverage before vs after the Popularity fallback, by ALS status, with hybrid quality."""
    status = serving_status(als_only, evaluation_users, history, k)
    after = list_lengths(combined, evaluation_users, k).set_index("user_id")["list_length"]
    status["after_length"] = status["user_id"].map(after).fillna(0).astype("int64")
    if combined_per_user is not None:
        status = status.merge(combined_per_user[["user_id", "recall", "ndcg"]],
                              on="user_id", how="left")
    rows = []
    for name in STATUSES + ["all"]:
        part = status if name == "all" else status[status["status"] == name]
        total = len(part)
        row = {
            "als_status": name,
            "hybrid_list_source": HYBRID_SOURCE.get(name, "all"),
            "users": total,
            "share_of_targets": total / max(len(status), 1),
            "coverage_ge1_before": float((part["list_length"] >= 1).mean()) if total else 0.0,
            "coverage_ge1_after": float((part["after_length"] >= 1).mean()) if total else 0.0,
            "coverage_full_before": float((part["list_length"] >= k).mean()) if total else 0.0,
            "coverage_full_after": float((part["after_length"] >= k).mean()) if total else 0.0,
            "fallback_items_per_user": float((part["after_length"] - part["list_length"]).mean())
            if total else 0.0,
            "users_rescued_from_zero": int(((part["list_length"] == 0)
                                            & (part["after_length"] >= 1)).sum()),
        }
        if combined_per_user is not None:
            scored = part.dropna(subset=["recall"])
            row.update({"scored_users": len(scored),
                        "hybrid_recall": float(scored["recall"].mean()) if len(scored) else 0.0,
                        "hybrid_ndcg": float(scored["ndcg"].mean()) if len(scored) else 0.0})
        rows.append(row)
    return pd.DataFrame(rows)
