"""T11 - Clean ratings va audit tung quy tac.

Ap dung lan luot cac quy tac, ghi so dong truoc/sau moi quy tac vao results/ratings_quality.csv
de moi dong bi loai deu truy vet duoc. Doi chieu: raw - tong dong bi loai == clean (neu lech -> bao loi).

  R1  user_id null/blank
  R2  product_id null/blank
  R3  rating null hoac ngoai [1, 5]
  R4  timestamp null / <= 0 / o tuong lai
  R5  exact duplicates (trung ca 4 cot)           -> giu 1 ban
  R6  nhieu rating khac nhau cho cung (user,item) -> giu ban MOI NHAT
      (timestamp giam dan, tie-break rating giam dan => deterministic)

Chay:
    python src/etl/clean_ratings.py --input data/raw/Electronics.csv \
        --output data/processed/ratings_clean --report results/ratings_quality.csv
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # them thu muc src/

import pandas as pd  # noqa: E402
from pyspark.sql import DataFrame, Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from common.spark_utils import get_spark, storage_level, write_parquet  # noqa: E402
from etl.read_ratings import header_arg, read_ratings  # noqa: E402

logger = logging.getLogger("clean_ratings")


def _blank(col: str):
    return F.col(col).isNull() | (F.trim(F.col(col)) == "")


def clean_ratings(df: DataFrame, max_timestamp: int = None):
    """Tra ve (clean_df, report_rows). report_rows ghi vao ratings_quality.csv."""
    if max_timestamp is None:
        max_timestamp = int(time.time())

    report = []
    current = df.persist(storage_level())
    n_raw = current.count()
    report.append(dict(step=0, rule="raw_input", rows_before=n_raw, rows_after=n_raw,
                       rows_removed=0, description="So dong doc duoc tu nguon"))

    def apply(step, rule, description, next_df):
        nonlocal current
        before = report[-1]["rows_after"]
        next_df = next_df.persist(storage_level())
        after = next_df.count()
        current.unpersist()
        current = next_df
        report.append(dict(step=step, rule=rule, rows_before=before, rows_after=after,
                           rows_removed=before - after, description=description))
        logger.info("%s: %d -> %d (removed %d)", rule, before, after, before - after)

    apply(1, "user_id_null_or_blank", "Loai dong co user_id null/rong", current.filter(~_blank("user_id")))
    apply(2, "product_id_null_or_blank", "Loai dong co product_id null/rong", current.filter(~_blank("product_id")))
    apply(3, "rating_invalid", "Loai dong co rating null hoac ngoai [1,5]",
          current.filter(F.col("rating").isNotNull() & F.col("rating").between(1.0, 5.0)))
    apply(4, "timestamp_invalid", f"Loai dong co timestamp null, <=0 hoac > {max_timestamp}",
          current.filter(F.col("timestamp").isNotNull() & (F.col("timestamp") > 0)
                         & (F.col("timestamp") <= max_timestamp)))
    apply(5, "exact_duplicates", "Loai dong trung hoan toan ca 4 cot (giu 1 ban)",
          current.dropDuplicates(["product_id", "user_id", "rating", "timestamp"]))

    n_multi_pairs = (current.groupBy("user_id", "product_id").count()
                     .filter(F.col("count") > 1).count())
    w = Window.partitionBy("user_id", "product_id").orderBy(F.col("timestamp").desc(), F.col("rating").desc())
    latest = current.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn")
    apply(6, "multi_rating_same_user_item",
          f"Cung (user,item) co nhieu rating khac nhau ({n_multi_pairs} cap): giu ban moi nhat", latest)

    n_clean = report[-1]["rows_after"]
    removed = sum(r["rows_removed"] for r in report)
    if n_raw - removed != n_clean:
        raise RuntimeError(f"Doi chieu that bai: raw={n_raw}, removed={removed}, clean={n_clean}")
    return current, report


def parse_args():
    p = argparse.ArgumentParser(description="Clean ratings and audit each rule")
    p.add_argument("--input", required=True)
    p.add_argument("--input-format", default="csv", choices=["csv", "parquet"])
    p.add_argument("--header", default="auto", choices=["auto", "true", "false"])
    p.add_argument("--limit", type=int, default=None, help="Chi xu ly N dong dau (pilot/dev)")
    p.add_argument("--output", default="data/processed/ratings_clean")
    p.add_argument("--report", default="results/ratings_quality.csv")
    p.add_argument("--num-output-files", type=int, default=8)
    p.add_argument("--writer", default="auto", choices=["auto", "spark", "pandas"])
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    spark = get_spark("clean_ratings")
    df = read_ratings(spark, args.input, args.input_format, header_arg(args.header))
    if args.limit:
        df = df.limit(args.limit)
    clean, report = clean_ratings(df)

    # ghi report truoc: neu buoc ghi Parquet loi, so lieu audit van con
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    pd.DataFrame(report).to_csv(args.report, index=False)
    used = write_parquet(clean, args.output, args.writer, args.num_output_files)
    logger.info("Da ghi %s (writer=%s) va %s | clean rows = %d", args.output, used, args.report,
                report[-1]["rows_after"])
    spark.stop()


if __name__ == "__main__":
    main()