"""Test cho T06 / T12 / T13 cua metadata, gom xu ly brand va khung FieldSpec. Chay: pytest tests/ -q"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql import types as T  # noqa: E402

from common.spark_utils import get_spark, write_parquet  # noqa: E402
from etl.read_metadata import read_metadata  # noqa: E402
from etl.metadata_fields import FieldSpec, clean_text, output_columns  # noqa: E402
from etl.clean_metadata import clean_metadata  # noqa: E402
from etl.profile_metadata import profile  # noqa: E402
from etl.write_parquet import DATASETS, parquet_dir_stats, verify  # noqa: E402

CAT = ["Electronics", "X"]
RECORDS = [
    {"asin": "A1", "title": "Cam &amp; Lens", "brand": "X", "main_cat": "Camera", "category": ["Electronics", "Camera"],
     "price": "$9.99", "rank": ["1 in X"], "details": {"a": "b"}},
    {"asin": "A1", "title": "", "brand": "", "main_cat": "Camera", "category": [], "price": ""},            # trung, kem day du
    {"asin": "A2", "title": "<b>Phone</b>  case", "brand": "Y", "main_cat": "<img src=\"x.png\"/>",
     "category": ["Electronics", "<i>Phones</i>"], "price": "$5.00 - $9.00", "rank": "5 in Y"},            # rank kieu hon hop
    {"asin": "A3", "title": "No cat", "brand": "", "main_cat": "Misc", "category": [], "price": "garbage"},
    {"asin": "", "title": "no asin", "brand": "Z", "main_cat": "Misc", "category": [], "price": ""},
    # --- brand: bien the, placeholder, truong hop dac biet ---
    {"asin": "S1", "title": "t", "brand": "Sony", "category": CAT},
    {"asin": "S2", "title": "t", "brand": "SONY", "category": CAT},
    {"asin": "S3", "title": "t", "brand": "Sony\u00ae", "category": CAT},
    {"asin": "S4", "title": "t", "brand": " sony ", "category": CAT},
    {"asin": "S5", "title": "t", "brand": "Visit the Sony Store", "category": CAT},
    {"asin": "M1", "title": "t", "brand": "Samsung", "category": CAT},
    {"asin": "M2", "title": "t", "brand": "SAMSUNG", "category": CAT},     # hoa count -> uu tien dang khong viet HOA
    {"asin": "N1", "title": "t", "brand": "N/A", "category": CAT},
    {"asin": "N2", "title": "t", "brand": "-", "category": CAT},
    {"asin": "N3", "title": "t", "brand": "Brand: Acme Corp.", "category": CAT},
    # --- trung ASIN: ban tot nhat thieu brand, ban kia co brand -> phai dien duoc ---
    {"asin": "D1", "title": "Has title", "brand": "", "category": CAT},
    {"asin": "D1", "title": "", "brand": "Acme Corp", "category": []},
]


@pytest.fixture(scope="module")
def spark():
    s = get_spark("test_meta", shuffle_partitions=2)
    yield s
    s.stop()


@pytest.fixture()
def path(tmp_path):
    p = tmp_path / "meta.json"
    p.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\nnot json\n", encoding="utf-8")
    return str(p)


@pytest.fixture()
def cleaned(spark, path):
    clean, report, info = clean_metadata(read_metadata(spark, path))
    return clean, report, info, {r["asin"]: r.asDict() for r in clean.collect()}


def test_profile_counts(spark, path):
    prof = profile(read_metadata(spark, path, with_corrupt=True))
    assert prof["total_rows"] == len(RECORDS) + 1 and prof["corrupt_records"] == 1
    assert prof["distinct_asin"] == 14 and prof["asin_with_duplicates"] == 2
    assert prof["title_has_html"] == 1 and prof["main_cat_has_html"] == 1
    assert prof["price_range"] == 1 and prof["price_malformed"] == 1 and prof["price_single_valid"] == 1
    removed = {x["value"]: x["count"] for x in prof["removed_by_cleaning_top10"]["brand"]}
    assert removed.get("N/A") == 1 and removed.get("-") == 1          # placeholder bi cleaner loai
    assert prof["variant_profile"]["brand"]["variant_groups"] >= 2     # nhom Sony va Samsung


def test_clean_unique_and_normalized(spark, cleaned):
    clean, report, _, rows = cleaned
    assert clean.count() == len(rows) == 14 and "" not in rows
    assert report[0]["rows_after"] - sum(r["rows_removed"] for r in report) == clean.count()
    assert rows["A1"]["title"] == "Cam & Lens"                      # giu ban day du + giai ma entity
    assert rows["A2"]["title"] == "Phone case"                      # bo HTML, gop khoang trang
    assert rows["A2"]["category"] == ["Electronics", "Phones"]
    assert rows["A2"]["main_cat"] is None                           # main_cat chi la the <img>
    assert rows["A1"]["category_path"] == "Electronics > Camera"
    assert clean.columns == [n for n, _ in output_columns()]


def test_brand_null_handling(cleaned):
    _, _, _, rows = cleaned
    assert rows["N1"]["brand"] is None and rows["N2"]["brand"] is None      # N/A, "-" -> null
    assert rows["N1"]["brand_raw"] == "N/A"                                  # gia tri goc van con de doi chieu
    assert rows["A3"]["brand"] is None                                       # rong -> null
    assert rows["N3"]["brand"] == "Acme Corp"                                # bo "Brand:" va dau cham cuoi


def test_brand_duplicates_merged(cleaned):
    _, _, info, rows = cleaned
    for asin in ("S1", "S2", "S3", "S4", "S5"):
        assert rows[asin]["brand"] == "Sony"                                  # SONY/Sony(R)/sony/Visit the Sony Store
    assert rows["S2"]["brand_raw"] == "SONY"                                  # raw giu nguyen
    assert rows["M1"]["brand"] == rows["M2"]["brand"] == "Samsung"           # hoa count -> khong chon dang HOA
    assert info["variants"]["brand"]["variant_groups"] >= 2
    assert info["variants"]["brand"]["rows_changed"] == 3        # SONY, sony, SAMSUNG


def test_brand_filled_from_duplicate_asin(cleaned):
    _, _, info, rows = cleaned
    assert rows["D1"]["title"] == "Has title"                                 # ban tot nhat (co title)
    assert rows["D1"]["brand"] == "Acme Corp"                                 # brand dien tu ban ghi trung
    assert rows["D1"]["category"] == CAT or rows["D1"]["category"] == []     # category khong bi pha
    fr = {f["field"]: f for f in info["fields"]}
    assert fr["brand"]["filled_from_duplicates"] >= 1
    assert fr["brand"]["newly_missing_from_cleaning_pre_dedup"] == 2         # N/A va "-"


def test_adding_a_field_needs_only_a_fieldspec(spark):
    """Them truong moi chi can FieldSpec: khong sua code pipeline."""
    def clean_model(c):
        return F.upper(F.regexp_replace(clean_text(c), r"\s+", ""))
    specs = [FieldSpec("title", completeness=True),
             FieldSpec("model", cleaner=clean_model),
             FieldSpec("feature", dtype=T.ArrayType(T.StringType())),
             FieldSpec("main_cat", canonicalize_variants=True, fill_null="Unknown")]
    df = spark.createDataFrame(
        [("Z1", "t", " ab 12 ", ["<b>fast</b>", ""], "Camera"),
         ("Z2", "t", None, None, "CAMERA"), ("Z3", "t", "x", ["a"], "Camera"), ("Z4", "t", "x", ["a"], None)],
        "asin string, title string, model string, feature array<string>, main_cat string")
    clean, _, info = clean_metadata(df, specs)
    rows = {r["asin"]: r.asDict() for r in clean.collect()}
    assert clean.columns == ["asin", "title", "model", "feature", "main_cat"]    # khong co category_path (khong co category)
    assert rows["Z1"]["model"] == "AB12" and rows["Z1"]["feature"] == ["fast"]
    assert rows["Z2"]["model"] is None and rows["Z2"]["feature"] == []
    assert rows["Z2"]["main_cat"] == "Camera" and rows["Z4"]["main_cat"] == "Unknown"   # gop bien the + dien null
    assert {f["field"] for f in info["fields"]} == {"title", "model", "feature", "main_cat"}


def test_pandas_writer_and_verify(spark, cleaned, tmp_path):
    clean, report, _, _ = cleaned
    out = str(tmp_path / "meta_pq")
    assert write_parquet(clean, out, writer="pandas") == "pandas"
    spec = DATASETS["metadata"]
    v = verify(spark, out, spec["dtypes"], spec["key"], expected_rows=report[-1]["rows_after"])
    assert v["rows"] == 14 and v["schema_match"] and v["key_unique"] and v["rows_match_report"]
    n_files, size = parquet_dir_stats(out)
    assert n_files == 1 and size > 0


def test_detect_writer(tmp_path):
    from etl.write_parquet import detect_writer
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d in (a, b, c):
        d.mkdir()
    (a / "part-00000-1111-2222-c000.snappy.parquet").write_bytes(b"x")     # kieu Spark
    (b / "part-00000.parquet").write_bytes(b"x")                           # kieu pandas/pyarrow
    assert detect_writer(str(a)) == "spark" and detect_writer(str(b)) == "pandas/pyarrow" and detect_writer(str(c)) == "unknown"