"""Beyond-accuracy evaluation added in E6: what the lists expose, not only whether they hit.

catalog_coverage      distinct recommended items / items in the history catalog
gini_exposure         inequality of item exposure across the whole catalog (0 equal, 1 one item)
arp                   average popularity (history positive count) of recommended items
head_share            share of recommended slots taken by "head" items, the most popular items
                      that together hold 20% of history positive ratings
novelty               mean self-information -log2(item users / history users) of slots, higher
                      means less obvious items
personalization       1 - mean Jaccard overlap between lists of random user pairs; 0 means every
                      user sees the same list
category_ild          intra-list diversity: share of item pairs in a list with different
                      level-2 category, computed only on items that have E3 metadata
"""

import numpy as np
import pandas as pd

from src.evaluation.per_user_metrics import top_k_recommendations


def item_statistics(history, relevance_threshold=4):
    """Per-item history counts used as popularity and novelty references."""
    stats = history.groupby("product_id").agg(
        interactions=("user_id", "size"),
        positives=("rating", lambda values: int((values >= relevance_threshold).sum())),
    )
    return stats


def head_items(item_stats, head_mass=0.2):
    """Most popular items that together account for head_mass of history positives."""
    ordered = item_stats.sort_values(["positives", "interactions"], ascending=False)
    cumulative = ordered["positives"].cumsum() / max(ordered["positives"].sum(), 1)
    cutoff = int(np.searchsorted(cumulative.to_numpy(), head_mass, side="left")) + 1
    return set(ordered.index[:cutoff])


def gini(values):
    """Gini coefficient of a non-negative vector (sorted Lorenz formula)."""
    array = np.sort(np.asarray(values, dtype=float))
    if array.size == 0 or array.sum() == 0:
        return 0.0
    index = np.arange(1, array.size + 1)
    return float((2 * np.sum(index * array) / (array.size * array.sum())) - (array.size + 1) / array.size)


def personalization(recommendations, k=10, sample_pairs=20000, seed=42):
    """1 - mean Jaccard similarity of Top-K lists over random user pairs."""
    top = top_k_recommendations(recommendations, k)
    lists = top.groupby("user_id")["product_id"].agg(frozenset)
    if len(lists) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    left = rng.integers(0, len(lists), sample_pairs)
    right = rng.integers(0, len(lists), sample_pairs)
    keep = left != right
    values = lists.to_numpy()
    overlaps = [len(values[a] & values[b]) / len(values[a] | values[b])
                for a, b in zip(left[keep], right[keep])]
    return float(1.0 - np.mean(overlaps))


def category_level(metadata, level=1):
    """Category at a given depth (0 = 'Electronics', 1 = e.g. 'Computers & Accessories')."""
    def pick(path):
        if path is None or (isinstance(path, float) and np.isnan(path)):
            return None
        parts = [part.strip() for part in str(path).split(">")]
        return parts[level] if len(parts) > level and parts[level] else None
    return metadata.set_index("asin")["category_path"].map(pick).dropna()


def intra_list_diversity(recommendations, categories, k=10):
    """Mean share of dissimilar-category pairs per list; lists with < 2 labelled items skipped."""
    top = top_k_recommendations(recommendations, k)
    labelled = top.assign(category=top["product_id"].map(categories)).dropna(subset=["category"])
    coverage = len(labelled) / max(len(top), 1)
    values = []
    for _, group in labelled.groupby("user_id")["category"]:
        n = len(group)
        if n < 2:
            continue
        counts = group.value_counts().to_numpy()
        same_pairs = float(np.sum(counts * (counts - 1)) / 2)
        values.append(1.0 - same_pairs / (n * (n - 1) / 2))
    return {
        "category_ild": float(np.mean(values)) if values else float("nan"),
        "metadata_slot_coverage": coverage,
        "lists_scored_for_ild": len(values),
    }


def beyond_accuracy(recommendations, history, k=10, categories=None, relevance_threshold=4,
                    seed=42):
    """All beyond-accuracy indicators for one recommendation table."""
    top = top_k_recommendations(recommendations, k)
    stats = item_statistics(history, relevance_threshold)
    catalog = stats.index
    exposure = top["product_id"].value_counts().reindex(catalog, fill_value=0)
    history_users = history["user_id"].nunique()
    slots = top["product_id"]
    known = slots[slots.isin(catalog)]
    self_info = -np.log2(stats.loc[known, "interactions"].to_numpy() / history_users)
    heads = head_items(stats)
    result = {
        "recommended_slots": int(len(top)),
        "distinct_items": int(slots.nunique()),
        "catalog_items": int(len(catalog)),
        "catalog_coverage": float(slots.nunique() / len(catalog)),
        "gini_exposure": gini(exposure.to_numpy()),
        "arp_positive_count": float(stats.loc[known, "positives"].mean()) if len(known) else 0.0,
        "median_item_positive_count": float(stats.loc[known, "positives"].median())
        if len(known) else 0.0,
        "head_items": len(heads),
        "head_share": float(slots.isin(heads).mean()) if len(slots) else 0.0,
        "novelty_bits": float(self_info.mean()) if len(self_info) else 0.0,
        "personalization": personalization(recommendations, k, seed=seed),
        "share_items_with_1_history_rating": float(
            (stats.loc[known, "interactions"] == 1).mean()) if len(known) else 0.0,
    }
    if categories is not None:
        result.update(intra_list_diversity(recommendations, categories, k))
    return result
