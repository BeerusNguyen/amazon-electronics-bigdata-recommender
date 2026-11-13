"""Test mau cho T10 (read), T11 (clean), T16 (split). Chay: pytest tests/ -q"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "etl"))
sys.path.insert(0, os.path.join(ROOT, "src", "evaluation"))

from read_ratings import get_spark, read_ratings  # noqa: E402
from clean_ratings import clean_ratings  # noqa: E402
from split_data import compute_cutoffs, split_by_time  # noqa: E402

DAY = 86400
LINES = [
    '{"overall": 5.0, "reviewerID": "U1", "asin": "A1", "unixReviewTime": %d, "reviewText": "x"}' % (100 * DAY),
    '{"overall": 4.0, "reviewerID": "U1", "asin": "A2", "unixReviewTime": %d}' % (101 * DAY),
    '{"overall": 4.0, "reviewerID": "U1", "asin": "A2", "unixReviewTime": %d}' % (101 * DAY),   # exact dup
    '{"overall": 2.0, "reviewerID": "U2", "asin": "A1", "unixReviewTime": %d}' % (102 * DAY),
    '{"overall": 5.0, "reviewerID": "U2", "asin": "A1", "unixReviewTime": %d}' % (103 * DAY),   # cung cap, moi hon
    '{"overall": 9.0, "reviewerID": "U3", "asin": "A1", "unixReviewTime": %d}' % (104 * DAY),   # rating loi
    '{"overall": 3.0, "reviewerID": "", "asin": "A3", "unixReviewTime": %d}' % (105 * DAY),     # user blank
    '{"overall": 3.0, "reviewerID": "U4", "asin": "A3", "unixReviewTime": 0}',                  # ts loi
    'this is not json',                                                                         # dong hong
]


@pytest.fixture(scope="module")
def spark():
    s = get_spark("test", shuffle_partitions=2)
    yield s
    s.stop()


@pytest.fixture()
def sample_path(tmp_path):
    p = tmp_path / "sample.json"
    p.write_text("\n".join(LINES), encoding="utf-8")
    return str(p)


def test_reader_column_order_and_count(spark, sample_path):
    df = read_ratings(spark, sample_path, "json")
    assert df.columns == ["product_id", "user_id", "rating", "timestamp"]
    assert df.count() == len(LINES)          # dong hong van duoc giu lai (toan null)
    row = df.filter("user_id = 'U1' AND product_id = 'A1'").first()
    assert row["rating"] == 5.0 and row["timestamp"] == 100 * DAY


def test_cleaning_rules_and_reconciliation(spark, sample_path):
    df = read_ratings(spark, sample_path, "json")
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
