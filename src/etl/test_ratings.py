"""Test mau cho T10 (read), T11 (clean), T13 (verify) va T14 (join) voi ratings CSV."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from common.spark_utils import get_spark, write_parquet  # noqa: E402
from etl.read_ratings import detect_header, read_ratings  # noqa: E402
from etl.clean_ratings import clean_ratings  # noqa: E402
from etl.join_metadata import build_enriched, join_quality  # noqa: E402
from etl.write_parquet import DATASETS, verify  # noqa: E402

DAY = 86400
LINES = [
    f"A1,U1,5.0,{100 * DAY}",
    f"A2,U1,4.0,{101 * DAY}",
    f"A2,U1,4.0,{101 * DAY}",          # exact duplicate
    f"A1,U2,2.0,{102 * DAY}",
    f"A1,U2,5.0,{103 * DAY}",          # cung cap (user,item), moi hon -> giu
    f"A1,U3,9.0,{104 * DAY}",          # rating ngoai [1,5]
    f"A3,,3.0,{105 * DAY}",            # user blank
    "A3,U4,3.0,0",                     # timestamp loi
    "A9,U9,abc,xyz",                   # sai kieu
]


@pytest.fixture(scope="module")
def spark():
    s = get_spark("test_ratings", shuffle_partitions=2)
    yield s
    s.stop()


@pytest.fixture()
def csv_path(tmp_path):
    p = tmp_path / "ratings.csv"
    p.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return str(p)


def test_header_detection(tmp_path):
    with_header = tmp_path / "h.csv"
    with_header.write_text("asin,reviewerID,rating,timestamp\nA1,U1,5.0,1\n", encoding="utf-8")
    no_header = tmp_path / "n.csv"
    no_header.write_text("A1,U1,5.0,1\n", encoding="utf-8")
    assert detect_header(str(with_header)) is True
    assert detect_header(str(no_header)) is False


def test_reader_order_and_header(spark, csv_path, tmp_path):
    df = read_ratings(spark, csv_path)
    assert df.columns == ["product_id", "user_id", "rating", "timestamp"]
    assert dict(df.dtypes) == {"product_id": "string", "user_id": "string", "rating": "double", "timestamp": "bigint"}
    assert df.count() == len(LINES)
    row = df.filter("user_id = 'U1' AND product_id = 'A1'").first()
    assert row["rating"] == 5.0 and row["timestamp"] == 100 * DAY
    # file co header: khong duoc tinh dong tieu de thanh du lieu
    h = tmp_path / "h.csv"
    h.write_text("asin,reviewerID,rating,timestamp\nA1,U1,5.0,1\nA2,U2,4.0,2\n", encoding="utf-8")
    assert read_ratings(spark, str(h)).count() == 2


def test_clean_rules_and_reconciliation(spark, csv_path):
    clean, report = clean_ratings(read_ratings(spark, csv_path), max_timestamp=10_000 * DAY)
    rules = {r["rule"]: r["rows_removed"] for r in report}
    assert rules["user_id_null_or_blank"] + rules["product_id_null_or_blank"] + rules["rating_invalid"] \
        + rules["timestamp_invalid"] == 4        # dong blank user, rating 9.0, ts=0, dong sai kieu
    assert rules["exact_duplicates"] == 1
    assert rules["multi_rating_same_user_item"] == 1
    assert clean.count() == 3
    assert clean.filter("user_id = 'U2' AND product_id = 'A1'").first()["rating"] == 5.0
    assert report[0]["rows_after"] - sum(r["rows_removed"] for r in report) == clean.count()


def test_parquet_verify_and_join(spark, csv_path, tmp_path):
    clean, report = clean_ratings(read_ratings(spark, csv_path), max_timestamp=10_000 * DAY)
    out = str(tmp_path / "ratings_pq")
    assert write_parquet(clean, out, writer="pandas", num_files=2) == "pandas"
    spec = DATASETS["ratings"]
    v = verify(spark, out, spec["dtypes"], spec["key"], expected_rows=report[-1]["rows_after"])
    assert v["rows"] == 3 and v["schema_match"] and v["key_unique"] and v["rows_match_report"]

    # T14: A1, A2 co metadata; them 1 rating cho san pham A7 khong co metadata
    ratings = clean.unionByName(spark.createDataFrame([("A7", "U5", 4.0, 5 * DAY)], clean.columns))
    meta = spark.createDataFrame(
        [("A1", "Cam", "X", ["Electronics"], "Electronics"), ("A2", None, None, [], None),
         ("A4", "Unused", "Y", ["E"], "E")],
        "asin string, title string, brand string, category array<string>, category_path string")
    enriched = build_enriched(ratings, meta)
    assert enriched.count() == ratings.count() == 4                  # LEFT JOIN: giu du moi rating
    assert enriched.columns == ["product_id", "user_id", "rating", "timestamp", "title", "brand",
                                "category", "category_path", "has_metadata"]
    assert enriched.filter("has_metadata").count() == 3
    assert enriched.filter("product_id = 'A7'").first()["title"] is None     # thieu metadata -> null, khong mat dong
    rep = join_quality(ratings, meta)
    assert rep["join_preserved_row_count"] and rep["joined_rows"] == ratings.count() == 4
    assert rep["matched_ratings"]["count"] == 3 and rep["matched_ratings"]["denominator"] == 4
    assert rep["matched_products"]["count"] == 2 and rep["matched_products"]["denominator"] == 3
    assert rep["metadata_products_with_rating"]["count"] == 2      # A4 khong co rating
    assert rep["matched_ratings_with_title"]["count"] == 2 and rep["matched_ratings_with_title"]["denominator"] == 3


def test_read_parquet_folder_vs_single_file(spark, tmp_path, caplog):
    import logging
    import pandas as pd
    from common.spark_utils import read_parquet
    d = tmp_path / "ratings_clean"
    d.mkdir()
    for i in range(3):   # mo phong 3 file part-0000i.parquet do pandas ghi
        pd.DataFrame({"product_id": [f"A{i}a", f"A{i}b"], "user_id": ["U1", "U2"],
                      "rating": [5.0, 4.0], "timestamp": [1, 2]}).to_parquet(d / f"part-0000{i}.parquet", index=False)
    (d / "_SUCCESS").write_text("")
    assert read_parquet(spark, str(d)).count() == 6               # ca thu muc: du 6 dong
    with caplog.at_level(logging.WARNING):
        assert read_parquet(spark, str(d / "part-00001.parquet")).count() == 2   # 1 file: chi 1 phan
    assert "MOT PHAN" in caplog.text
    with pytest.raises(FileNotFoundError):
        read_parquet(spark, str(tmp_path / "khong_ton_tai"))


def test_enriched_pyarrow_matches_spark(spark, tmp_path):
    """Writer pyarrow (streaming) phai cho cung ket qua voi build_enriched cua Spark."""
    import pandas as pd
    from etl.join_metadata import build_enriched, parquet_row_count, write_enriched_pyarrow
    rd, md = tmp_path / "r", tmp_path / "m"
    rd.mkdir(); md.mkdir()
    for i in range(2):
        pd.DataFrame({"product_id": ["A1", "A2", "A7", "A1"], "user_id": [f"U{i}{j}" for j in range(4)],
                      "rating": [5.0, 4.0, 3.0, 2.0], "timestamp": [1, 2, 3, 4]}).to_parquet(rd / f"part-0000{i}.parquet", index=False)
    pd.DataFrame({"asin": ["A1", "A2", "A4"], "title": ["Cam", None, "Unused"], "brand": ["X", "Y", None],
                  "brand_raw": ["x", "y", None], "category": [["E", "C"], [], ["E"]],
                  "category_path": ["E > C", None, "E"]}).to_parquet(md / "part-00000.parquet", index=False)
    out = tmp_path / "enriched"
    info = write_enriched_pyarrow(str(rd), str(md), str(out), batch_rows=3, rows_per_file=5)
    assert info["rows"] == parquet_row_count(str(out)) == 8 and info["files"] >= 2
    assert "brand_raw" not in info["columns"] and info["columns"][-1] == "has_metadata"

    from common.spark_utils import read_parquet
    got = read_parquet(spark, str(out))
    exp = build_enriched(read_parquet(spark, str(rd)), read_parquet(spark, str(md)))
    assert got.columns == exp.columns
    norm = lambda rows: sorted(tuple(tuple(v) if isinstance(v, list) else v for v in r) for r in rows)  # noqa: E731
    assert norm(got.collect()) == norm(exp.collect())
    assert got.filter("product_id = 'A7'").first()["has_metadata"] is False


def test_enriched_pyarrow_handles_not_null_metadata_schema(tmp_path):
    """Hoi quy: Parquet metadata do Spark ghi co cot NOT NULL; LEFT JOIN van phai ghi duoc rating thieu metadata."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from etl.join_metadata import write_enriched_pyarrow
    rd, md = tmp_path / "r", tmp_path / "m"
    rd.mkdir(); md.mkdir()
    pd.DataFrame({"product_id": ["A1", "A9"], "user_id": ["U1", "U2"], "rating": [5.0, 4.0], "timestamp": [1, 2]}
                 ).to_parquet(rd / "part-00000.parquet", index=False)
    schema = pa.schema([pa.field("asin", pa.string(), nullable=False),
                        pa.field("title", pa.string(), nullable=True),
                        pa.field("category", pa.list_(pa.string()), nullable=False)])     # NOT NULL nhu Spark
    pq.write_table(pa.Table.from_pylist([{"asin": "A1", "title": "Cam", "category": ["E"]}], schema=schema),
                   md / "part-00000.parquet")
    out = tmp_path / "enriched"
    info = write_enriched_pyarrow(str(rd), str(md), str(out))
    t = pq.read_table(next(out.glob("*.parquet"))).to_pylist()
    assert info["rows"] == 2
    by_user = {r["user_id"]: r for r in t}
    assert by_user["U1"]["has_metadata"] is True and by_user["U1"]["category"] == ["E"]
    assert by_user["U2"]["has_metadata"] is False and by_user["U2"]["category"] is None