"""E6 parity: the distributed Spark metrics must equal the pandas reference on the same data."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.protocol import build_ground_truth, build_ground_truth_spark
from src.evaluation.per_user_metrics import (
    PER_USER_COLUMNS,
    per_user_ranking_metrics,
    per_user_ranking_metrics_spark,
)


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession
    session = (SparkSession.builder.master("local[1]").appName("e6-parity")
               .config("spark.sql.shuffle.partitions", "2").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def random_snapshot(seed=11, users=80, items=40):
    rng = np.random.default_rng(seed)
    user_ids = [f"u{n:03d}" for n in range(users)]
    item_ids = [f"p{n:03d}" for n in range(items)]
    history = pd.DataFrame({
        "user_id": rng.choice(user_ids[: users // 2], 300),
        "product_id": rng.choice(item_ids, 300),
        "rating": rng.integers(1, 6, 300).astype(float),
    }).drop_duplicates(["user_id", "product_id"])
    held_out = pd.DataFrame({
        "user_id": rng.choice(user_ids, 250),
        "product_id": rng.choice(item_ids + ["new1", "new2"], 250),
        "rating": rng.integers(1, 6, 250).astype(float),
    }).drop_duplicates(["user_id", "product_id"])
    rows = []
    for user in held_out.user_id.unique():
        if rng.random() < 0.8:
            chosen = rng.choice(item_ids, rng.integers(1, 11), replace=False)
            rows += [(str(user), str(item), rank) for rank, item in enumerate(chosen, start=1)]
    recommendations = pd.DataFrame(rows, columns=["user_id", "product_id", "rank"])
    return history, held_out, recommendations


def test_spark_ground_truth_and_metrics_match_pandas(spark):
    history, held_out, recommendations = random_snapshot()
    users = held_out.user_id.unique()
    gt_pd = build_ground_truth(held_out, history)
    gt_sp = build_ground_truth_spark(spark.createDataFrame(held_out),
                                     spark.createDataFrame(history)).toPandas()
    assert len(gt_pd) == len(gt_sp)
    merged = gt_pd.merge(gt_sp, on=["user_id", "product_id"], suffixes=("", "_s"))
    assert len(merged) == len(gt_pd)
    assert (merged.gain_level == merged.gain_level_s).all()
    assert (merged.warm_item == merged.warm_item_s).all()

    expected = per_user_ranking_metrics(recommendations, gt_pd, users, 10)
    actual = per_user_ranking_metrics_spark(
        spark.createDataFrame(recommendations), spark.createDataFrame(gt_pd),
        spark.createDataFrame(pd.DataFrame({"user_id": users})), 10,
    ).toPandas()
    joined = expected.merge(actual, on="user_id", suffixes=("", "_s"))
    assert len(joined) == len(expected) == len(actual)
    for column in PER_USER_COLUMNS[1:]:
        assert np.allclose(joined[column].astype(float), joined[f"{column}_s"].astype(float),
                           atol=1e-12), column
