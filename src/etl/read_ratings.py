"""T10 - Schema-based ratings reader (Ratings Only CSV).

Doc ratings Amazon Electronics bang Spark voi schema tuong minh, tra ve DataFrame gom 4 cot
theo dung thu tu:  product_id (string), user_id (string), rating (double), timestamp (long)

File "Ratings Only" cua nijianmo KHONG co header; thu tu cot: item(asin), user(reviewerID), rating, timestamp.
Script tu phat hien header (file local) de tranh doc nham dong tieu de; co the ep bang --header true/false.

Dong CSV hong (sai so cot/sai kieu) duoc Spark dua ve null (mode PERMISSIVE) va bi bat boi
quy tac null/invalid trong clean_ratings.py (T11) - khong bi mat im lang.

Chay thu:
    python src/etl/read_ratings.py --input data/raw/Electronics.csv --show 10
    python src/etl/read_ratings.py --input data/raw/Electronics.csv --limit 100000 --output data/interim/ratings_pilot
"""
import argparse
import gzip
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # them thu muc src/

from pyspark.sql import DataFrame, SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql import types as T  # noqa: E402

from common.spark_utils import get_spark, read_parquet, write_parquet  # noqa: E402

logger = logging.getLogger("read_ratings")

CSV_SCHEMA = T.StructType([
    T.StructField("product_id", T.StringType(), True),   # asin
    T.StructField("user_id", T.StringType(), True),      # reviewerID
    T.StructField("rating", T.DoubleType(), True),
    T.StructField("timestamp", T.LongType(), True),      # unix seconds
])
OUTPUT_COLUMNS = ["product_id", "user_id", "rating", "timestamp"]


def detect_header(path: str) -> bool:
    """True neu dong dau cua file CSV local la header (truong thu 3 khong phai so). Khong doan duoc -> False."""
    if "://" in path or not os.path.isfile(path):
        return False
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        parts = f.readline().strip().split(",")
    if len(parts) < 3:
        return False
    try:
        float(parts[2])
        return False
    except ValueError:
        return True


def read_ratings(spark: SparkSession, path: str, fmt: str = "csv", header: bool = None) -> DataFrame:
    """Tra ve DataFrame [product_id, user_id, rating, timestamp]. header=None -> tu phat hien."""
    if fmt == "parquet":
        return read_parquet(spark, path).select(*OUTPUT_COLUMNS)
    if fmt != "csv":
        raise ValueError(f"Dinh dang khong ho tro: {fmt}")
    if header is None:
        header = detect_header(path)
    logger.info("Doc CSV %s (header=%s)", path, header)
    return (spark.read.schema(CSV_SCHEMA)
            .option("header", str(bool(header)).lower())
            .option("mode", "PERMISSIVE")
            .csv(path)
            .select(*OUTPUT_COLUMNS))


def header_arg(value: str):
    return {"auto": None, "true": True, "false": False}[value]


def parse_args():
    p = argparse.ArgumentParser(description="Read Amazon Electronics ratings (CSV) with a fixed schema")
    p.add_argument("--input", required=True, help="File/thu muc CSV ratings (local hoac gs://)")
    p.add_argument("--format", default="csv", choices=["csv", "parquet"])
    p.add_argument("--header", default="auto", choices=["auto", "true", "false"])
    p.add_argument("--limit", type=int, default=None, help="Chi lay N dong dau (pilot/dev, khong dai dien)")
    p.add_argument("--output", default=None, help="Neu co: ghi ket qua ra Parquet")
    p.add_argument("--show", type=int, default=10)
    p.add_argument("--writer", default="auto", choices=["auto", "spark", "pandas"])
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    spark = get_spark("read_ratings")
    df = read_ratings(spark, args.input, args.format, header_arg(args.header))
    if args.limit:
        df = df.limit(args.limit)

    df.printSchema()
    df.show(args.show, truncate=False)
    total = df.count()
    null_rows = df.filter(F.col("user_id").isNull() | F.col("product_id").isNull()
                          | F.col("rating").isNull() | F.col("timestamp").isNull()).count()
    logger.info("rows=%d, rows_with_null_field=%d", total, null_rows)
    if total and null_rows == total:
        logger.warning("TAT CA dong deu co null: kiem tra lai file (co phai ratings khong? thu tu cot? header?).")

    if args.output:
        used = write_parquet(df, args.output, args.writer)
        logger.info("Da ghi Parquet (%s): %s", used, args.output)
    spark.stop()


if __name__ == "__main__":
    main()