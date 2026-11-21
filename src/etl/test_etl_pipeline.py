"""Tests for the ratings reader, cleaner and temporal split."""

import pytest

from src.common.spark_utils import get_spark
from src.etl.clean_ratings import clean_ratings
from src.etl.read_ratings import read_ratings
from src.evaluation.split_data import compute_cutoffs, split_by_time
from src.splitting.prepare_e4 import first_interactions

DAY = 86400
LINES = [
    f"A1,U1,5.0,{100 * DAY}",
    f"A2,U1,4.0,{101 * DAY}",
    f"A2,U1,4.0,{101 * DAY}",  # Exact duplicate
    f"A1,U2,2.0,{102 * DAY}",
    f"A1,U2,5.0,{103 * DAY}",  # Later rating
    f"A1,U3,9.0,{104 * DAY}",  # Invalid rating
    f"A3,,3.0,{105 * DAY}",    # Missing user
    "A3,U4,3.0,0",             # Invalid timestamp
    "malformed row",
]


@pytest.fixture(scope="module")
def spark():
    s = get_spark("test", shuffle_partitions=2)
    yield s
    s.stop()


@pytest.fixture()
def sample_path(tmp_path):
    p = tmp_path / "sample.csv"
    p.write_text("\n".join(LINES), encoding="utf-8")
    return str(p)


def test_reader_column_order_and_count(spark, sample_path):
    df = read_ratings(spark, sample_path, header=False)
    assert df.columns == ["product_id", "user_id", "rating", "timestamp"]
    assert df.count() == len(LINES)
    row = df.filter("user_id = 'U1' AND product_id = 'A1'").first()
    assert row["rating"] == 5.0 and row["timestamp"] == 100 * DAY


def test_cleaning_rules_and_reconciliation(spark, sample_path):
    df = read_ratings(spark, sample_path, header=False)
    clean, report = clean_ratings(df, max_timestamp=10_000 * DAY)
    rules = {r["rule"]: r["rows_removed"] for r in report}
    assert rules["user_id_null_or_blank"] == 2       # dong hong + user blank
    assert rules["rating_invalid"] == 1
    assert rules["timestamp_invalid"] == 1
    assert rules["exact_duplicates"] == 1
    assert rules["multi_rating_same_user_item"] == 1
    assert clean.count() == 3
    kept = clean.filter("user_id = 'U2' AND product_id = 'A1'").first()
    assert kept["rating"] == 5.0                     # giu ban moi nhat
    assert report[0]["rows_after"] - sum(r["rows_removed"] for r in report) == clean.count()


def test_temporal_split_no_overlap(spark):
    rows = [(f"A{i}", f"U{i}", 5.0, i * DAY) for i in range(1, 101)]
    df = spark.createDataFrame(rows, ["product_id", "user_id", "rating", "timestamp"])
    c_val, c_test = compute_cutoffs(df, 0.8, 0.1)
    train, val, test = split_by_time(df, c_val, c_test)
    assert train.count() + val.count() + test.count() == 100
    assert train.agg({"timestamp": "max"}).first()[0] < c_val <= val.agg({"timestamp": "min"}).first()[0]
    assert val.agg({"timestamp": "max"}).first()[0] < c_test <= test.agg({"timestamp": "min"}).first()[0]


def test_e4_retains_earliest_interaction_and_flags_earliest_conflict(spark):
    rows = [
        ("A1", "U1", 2.0, 100), ("A1", "U1", 2.0, 100), ("A1", "U1", 5.0, 200),
        ("A2", "U2", 3.0, 100), ("A2", "U2", 4.0, 100),
    ]
    df = spark.createDataFrame(rows, ["product_id", "user_id", "rating", "timestamp"])
    by_user = {row.user_id: row for row in first_interactions(df).collect()}
    first = by_user["U1"]
    assert (first.rating, first.timestamp, first.valid_rows, first.exact_unique_rows) == (2.0, 100, 3, 2)
    assert not first.conflict
    assert by_user["U2"].conflict


def test_e3_history_retains_temporal_events_for_e4(spark):
    rows = [("A1", "U1", 2.0, 100), ("A1", "U1", 2.0, 100), ("A1", "U1", 5.0, 200)]
    df = spark.createDataFrame(rows, ["product_id", "user_id", "rating", "timestamp"])
    history, report = clean_ratings(df, max_timestamp=300, pair_policy="history")
    assert sorted(row.timestamp for row in history.collect()) == [100, 200]
    assert report[-1]["rule"] == "preserve_rating_history"
    assert first_interactions(history).first().timestamp == 100
