"""E6 evaluation protocol: snapshot loading, ground truth, reachable ceiling and integrity checks.

Frozen rules (T18, docs/E6_README.md section 2):
  history         Validation: Train. Test: Train + Validation
  relevant target held-out rating >= relevance_threshold (default 4), never an item already
                  rated by the user in history
  warm_item       target item exists in the history catalog
  warm_user       user has at least one history interaction
  gain_level      rating - relevance_threshold + 1 (4 stars -> 1, 5 stars -> 2) for graded NDCG
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

GROUND_TRUTH_COLUMNS = [
    "user_id", "product_id", "rating", "gain_level", "warm_user", "warm_item",
]


def _check_columns(frame, required, name):
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def load_snapshot(data_dir, phase):
    """History and held-out exactly as E5 used them."""
    data_dir = Path(data_dir)
    if phase not in {"validation", "test"}:
        raise ValueError("phase must be validation or test")
    history = pd.read_parquet(data_dir / "train.parquet")
    if phase == "test":
        history = pd.concat([history, pd.read_parquet(data_dir / "validation.parquet")],
                            ignore_index=True)
    return history, pd.read_parquet(data_dir / f"{phase}.parquet")


def build_ground_truth(held_out, history, relevance_threshold=4):
    """One row per relevant (user, item) target of the held-out period (pandas)."""
    _check_columns(held_out, {"user_id", "product_id", "rating"}, "held_out")
    _check_columns(history, {"user_id", "product_id"}, "history")
    if held_out[["user_id", "product_id"]].isna().any().any():
        raise ValueError("held_out contains null user_id or product_id")

    relevant = held_out.loc[
        held_out["rating"] >= relevance_threshold, ["user_id", "product_id", "rating"]
    ]
    # note: keep the highest rating if a pair repeats, so a target is never counted twice
    relevant = (relevant.sort_values("rating", ascending=False, kind="stable")
                .drop_duplicates(["user_id", "product_id"], keep="first"))
    seen = history[["user_id", "product_id"]].drop_duplicates().assign(_seen=True)
    relevant = relevant.merge(seen, on=["user_id", "product_id"], how="left")
    relevant = relevant[relevant["_seen"].isna()].drop(columns="_seen")

    relevant = relevant.assign(
        gain_level=(relevant["rating"] - relevance_threshold + 1).astype("int64"),
        warm_user=relevant["user_id"].isin(set(history["user_id"].unique())),
        warm_item=relevant["product_id"].isin(set(history["product_id"].unique())),
    )
    return relevant[GROUND_TRUTH_COLUMNS].sort_values(
        ["user_id", "product_id"], kind="stable").reset_index(drop=True)


def build_ground_truth_spark(held_out, history, relevance_threshold=4):
    """Distributed equivalent of build_ground_truth."""
    _check_columns(held_out, {"user_id", "product_id", "rating"}, "held_out")
    _check_columns(history, {"user_id", "product_id"}, "history")
    relevant = (held_out.where(F.col("rating") >= relevance_threshold)
                .groupBy("user_id", "product_id").agg(F.max("rating").alias("rating"))
                .join(history.select("user_id", "product_id").distinct(),
                      ["user_id", "product_id"], "left_anti"))
    users = history.select("user_id").distinct().withColumn("warm_user", F.lit(True))
    items = history.select("product_id").distinct().withColumn("warm_item", F.lit(True))
    return (relevant.join(users, "user_id", "left").join(items, "product_id", "left")
            .fillna({"warm_user": False, "warm_item": False})
            .withColumn("gain_level",
                        (F.col("rating") - F.lit(relevance_threshold) + 1).cast("long"))
            .select(*GROUND_TRUTH_COLUMNS))


def restrict_ground_truth(ground_truth, scope):
    """'all' keeps every target; 'warm_item' keeps targets that exist in the catalog."""
    if scope == "all":
        return ground_truth
    if scope == "warm_item":
        return ground_truth[ground_truth["warm_item"]]
    raise ValueError(f"Unknown ground-truth scope: {scope}")


def reachable_ceiling(ground_truth, k=10):
    """Best Recall/NDCG any recommender limited to the history catalog could reach."""
    per_user = ground_truth.groupby("user_id").agg(
        relevant=("product_id", "size"), reachable=("warm_item", "sum"))
    prefix = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(1, k + 1) + 1.0))])
    reachable = np.minimum(per_user["reachable"].to_numpy(), k)
    relevant = per_user["relevant"].to_numpy()
    return {
        "recall_ceiling": float(np.mean(reachable / relevant)),
        "ndcg_ceiling": float(np.mean(prefix[reachable] / prefix[np.minimum(relevant, k)])),
        "users_with_any_reachable_target": int((reachable > 0).sum()),
    }


def protocol_summary(held_out, history, ground_truth, relevance_threshold=4, k=10):
    """Every count behind the denominators, plus the fingerprint all methods must share."""
    relevant_pairs = int(held_out.loc[held_out["rating"] >= relevance_threshold,
                                      ["user_id", "product_id"]].drop_duplicates().shape[0])
    return {
        "k": int(k),
        "relevance_threshold": int(relevance_threshold),
        "target_users": int(held_out["user_id"].nunique()),
        "history_pairs": int(len(history)),
        "catalog_items": int(history["product_id"].nunique()),
        "held_out_pairs": int(len(held_out)),
        "relevant_pairs": relevant_pairs,
        "removed_as_seen_in_history": relevant_pairs - int(len(ground_truth)),
        "ground_truth_pairs": int(len(ground_truth)),
        "scored_users": int(ground_truth["user_id"].nunique()),
        "cold_item_targets": int((~ground_truth["warm_item"]).sum()),
        "targets_of_cold_users": int((~ground_truth["warm_user"]).sum()),
        "share_five_star_targets": float((ground_truth["rating"] >= 5).mean()),
        **reachable_ceiling(ground_truth, k),
    }


def recommendation_integrity(recommendations, history, held_out, evaluation_users, k=10):
    """Violation counts for one Top-K table; the table is scored only if all are 0."""
    recs = recommendations
    catalog = set(history["product_id"].unique())
    held_out_only = set(held_out["product_id"].unique()) - catalog
    seen = history[["user_id", "product_id"]].drop_duplicates()
    contiguous = recs.groupby("user_id")["rank"].apply(
        lambda ranks: sorted(ranks.tolist()) == list(range(1, len(ranks) + 1)))
    return {
        "duplicate_user_item": int(recs.duplicated(["user_id", "product_id"]).sum()),
        "duplicate_user_rank": int(recs.duplicated(["user_id", "rank"]).sum()),
        "non_contiguous_rank_users": int((~contiguous).sum()),
        "lists_longer_than_k": int((recs.groupby("user_id").size() > k).sum()),
        "seen_items_recommended": int(len(recs.merge(seen, on=["user_id", "product_id"]))),
        "items_outside_history_catalog": int((~recs["product_id"].isin(catalog)).sum()),
        "held_out_only_items_recommended": int(recs["product_id"].isin(held_out_only).sum()),
        "users_outside_target_set": int((~recs["user_id"].isin(set(evaluation_users))).sum()),
        "null_keys": int(recs[["user_id", "product_id", "rank"]].isna().any(axis=1).sum()),
    }


def temporal_integrity(history, held_out):
    """History must end strictly before the held-out period, with no shared pair."""
    shared = history[["user_id", "product_id"]].drop_duplicates().merge(
        held_out[["user_id", "product_id"]].drop_duplicates(), on=["user_id", "product_id"])
    return {
        "history_end": pd.to_datetime(history["timestamp"].max(), unit="s").date().isoformat(),
        "held_out_start": pd.to_datetime(held_out["timestamp"].min(), unit="s").date().isoformat(),
        "ordered": bool(history["timestamp"].max() < held_out["timestamp"].min()),
        "shared_pairs": int(len(shared)),
    }
