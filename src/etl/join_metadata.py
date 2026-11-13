"""T14 - Join metadata vao ratings va bao cao coverage (enrichment chi de phuc vu hien thi/serving).

Left join: ratings.product_id = metadata.asin (metadata da khu trung ASIN o T12). ratings o ben TRAI.
Cot metadata duoc lay TU DONG tu Parquet clean (tru asin va cac cot *_raw): them truong o
etl/metadata_fields.py thi bang join va bao cao coverage tu co them cot do.
Moi ty le ghi kem tu so va mau so. Kiem tra join KHONG lam tang so dong ratings.

Chay:
    python src/etl/join_metadata.py --ratings data/processed/ratings_clean --metadata data/processed/metadata_clean
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # them thu muc src/

from pyspark.sql import DataFrame  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from common.spark_utils import get_spark, parquet_files, read_parquet, write_parquet  # noqa: E402

logger = logging.getLogger("join_metadata")


def frac(count: int, denominator: int, note: str = "") -> dict:
    return dict(count=int(count), denominator=int(denominator),
                rate=round(count / denominator, 4) if denominator else None, note=note)


def value_columns(meta: DataFrame):
    """Cac cot metadata dua vao bang join: tat ca tru asin va *_raw."""
    return [c for c in meta.columns if c != "asin" and not c.endswith("_raw")]


def join_metadata(ratings: DataFrame, meta: DataFrame) -> DataFrame:
    m = meta.select(F.col("asin").alias("_asin"), *value_columns(meta), F.lit(1).alias("_matched"))
    return ratings.join(F.broadcast(m), ratings["product_id"] == m["_asin"], "left").drop("_asin")


def build_enriched(ratings: DataFrame, meta: DataFrame) -> DataFrame:
    """Bang ratings da gan metadata (LEFT JOIN, giu du moi rating) + cot has_metadata."""
    return (join_metadata(ratings, meta)
            .withColumn("has_metadata", F.col("_matched").isNotNull())
            .drop("_matched"))


def write_enriched_pyarrow(ratings_path: str, meta_path: str, out_dir: str,
                          batch_rows: int = 500_000, rows_per_file: int = 2_000_000) -> dict:
    """Ghi bang ratings LEFT JOIN metadata theo kieu STREAMING, khong dung Spark/JVM.

    Doc ratings tung batch (mac dinh 500k dong), join voi metadata (nho, nap 1 lan vao pandas), ghi ra Parquet
    roi giai phong batch => bo nho chi phu thuoc kich thuoc batch, khong phu thuoc 21M dong. Khong can Hadoop/winutils.
    Ket qua giong build_enriched(): cac cot ratings, cac cot metadata (tru asin, *_raw), has_metadata.
    """
    rfiles, mfiles = parquet_files(ratings_path), parquet_files(meta_path)
    meta_schema = pq.ParquetFile(mfiles[0]).schema_arrow
    value_cols = [n for n in meta_schema.names if n != "asin" and not n.endswith("_raw")]
    meta_pd = pa.concat_tables([pq.read_table(f, columns=["asin"] + value_cols) for f in mfiles]).to_pandas()
    if meta_pd["asin"].duplicated().any():
        raise RuntimeError("Metadata co ASIN trung: chay clean_metadata.py (T12) truoc.")
    meta_pd = meta_pd.set_index("asin")

    r_schema = pq.ParquetFile(rfiles[0]).schema_arrow
    target = pa.schema([r_schema.field(n) for n in r_schema.names]
                       + [meta_schema.field(n).with_nullable(True) for n in value_cols]   # LEFT JOIN: co the null du Parquet nguon la NOT NULL
                       + [pa.field("has_metadata", pa.bool_())])

    if os.path.isdir(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    writer, in_file, n_files, total = None, 0, 0, 0
    try:
        for f in rfiles:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=batch_rows):
                left = batch.to_pandas()
                joined = left.join(meta_pd, on="product_id", how="left")
                if len(joined) != len(left):
                    raise RuntimeError("Join lam thay doi so dong (ASIN metadata trung?)")
                joined["has_metadata"] = left["product_id"].isin(meta_pd.index).to_numpy()
                table = pa.Table.from_pandas(joined[target.names], schema=target, preserve_index=False)
                if writer is None or in_file >= rows_per_file:
                    if writer is not None:
                        writer.close()
                    writer = pq.ParquetWriter(os.path.join(out_dir, f"part-{n_files:05d}.parquet"), target)
                    n_files, in_file = n_files + 1, 0
                writer.write_table(table)
                in_file += len(table)
                total += len(table)
    finally:
        if writer is not None:
            writer.close()
    return dict(rows=total, files=n_files, columns=target.names)


def parquet_row_count(path: str) -> int:
    """Tong so dong tu footer cua cac file Parquet (khong can Spark)."""
    return sum(pq.ParquetFile(f).metadata.num_rows for f in parquet_files(path))


def join_quality(ratings: DataFrame, meta: DataFrame) -> dict:
    """Bao cao coverage cua ratings LEFT JOIN metadata.

    Tiet kiem bo nho: KHONG cache va KHONG mang cot chuoi dai vao phep join. Cac co "truong nay co gia tri"
    duoc tinh tren bang metadata nho; chi cot product_id cua ratings duoc join voi cac co do.
    """
    n_meta = meta.count()
    if meta.select("asin").distinct().count() != n_meta:
        raise RuntimeError("Metadata co ASIN trung: chay clean_metadata.py (T12) truoc.")
    n_ratings = ratings.count()

    cols, dtypes = value_columns(meta), dict(meta.dtypes)
    has = lambda c: (F.size(c) > 0) if dtypes[c].startswith("array") else F.col(c).isNotNull()  # noqa: E731
    flags = meta.select(F.col("asin").alias("_asin"), F.lit(True).alias("_matched"),
                        *[has(c).alias(f"_has_{c}") for c in cols])
    joined = ratings.select("product_id").join(
        F.broadcast(flags), F.col("product_id") == F.col("_asin"), "left")

    def count_if(cond):
        return F.sum(F.when(cond, 1).otherwise(0))

    r = joined.agg(F.count(F.lit(1)).alias("rows"), count_if(F.col("_matched")).alias("matched"),
                   *[count_if(F.col(f"_has_{c}")).alias(f"with_{c}") for c in cols]).first()
    rows, matched = int(r["rows"]), int(r["matched"])
    if rows != n_ratings:
        raise RuntimeError(f"Join lam thay doi so dong ratings: {n_ratings} -> {rows}")

    meta_asins = meta.select("asin").distinct()
    rating_products = ratings.select("product_id").distinct()
    n_products = rating_products.count()
    matched_products = rating_products.join(meta_asins, rating_products["product_id"] == meta_asins["asin"],
                                            "left_semi").count()
    meta_with_rating = meta_asins.join(rating_products, meta_asins["asin"] == rating_products["product_id"],
                                       "left_semi").count()

    report = dict(
        ratings_rows=n_ratings, joined_rows=rows, join_preserved_row_count=(rows == n_ratings),
        metadata_rows=n_meta,
        matched_ratings=frac(matched, n_ratings, "ratings co product_id ton tai trong metadata"),
        unmatched_ratings=frac(n_ratings - matched, n_ratings, "ratings thieu metadata"),
        matched_products=frac(matched_products, n_products, "product_id distinct trong ratings co metadata"),
        unmatched_products=frac(n_products - matched_products, n_products),
        metadata_products_with_rating=frac(meta_with_rating, n_meta, "ASIN metadata co it nhat 1 rating"),
    )
    for c in cols:
        report[f"matched_ratings_with_{c}"] = frac(int(r[f"with_{c}"]), matched, "mau so = ratings da match")
    return report


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Join metadata to ratings: report coverage and write enriched table")
    ap.add_argument("--ratings", default="data/processed/ratings_clean", help="THU MUC chua cac file part-*.parquet")
    ap.add_argument("--metadata", default="data/processed/metadata_clean", help="THU MUC chua cac file part-*.parquet")
    ap.add_argument("--output", default="results/join_quality.json", help="Bao cao coverage (JSON)")
    ap.add_argument("--enriched-output", default="data/processed/ratings_enriched",
                    help="Thu muc Parquet cho bang ratings da join metadata")
    ap.add_argument("--skip-enriched", action="store_true", help="Chi tinh bao cao, khong ghi bang join")
    ap.add_argument("--num-output-files", type=int, default=16,
                    help="So file output; voi writer pandas, so cang lon moi phan cang nho (it ton RAM)")
    ap.add_argument("--writer", default="pyarrow", choices=["pyarrow", "auto", "spark", "pandas"],
                    help="pyarrow (mac dinh): streaming, it RAM, khong can Hadoop. spark/auto/pandas: di qua Spark")
    ap.add_argument("--batch-rows", type=int, default=500_000, help="Chi cho writer pyarrow: so dong moi batch")
    args = ap.parse_args()

    spark = get_spark("join_metadata")
    ratings = read_parquet(spark, args.ratings)
    meta = read_parquet(spark, args.metadata)
    report = join_quality(ratings, meta)
    logger.info("matched_ratings=%s | matched_products=%s", report["matched_ratings"]["rate"],
                report["matched_products"]["rate"])

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    def save_report():
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    save_report()                                            # bao cao coverage luu ngay, truoc buoc ghi bang join
    logger.info("Da ghi %s", args.output)

    if not args.skip_enriched:
        if args.writer == "pyarrow":
            info = write_enriched_pyarrow(args.ratings, args.metadata, args.enriched_output, args.batch_rows)
            used, columns = "pyarrow", info["columns"]
        else:
            enriched = build_enriched(ratings, meta)        # day du cot; khong cache
            used = write_parquet(enriched, args.enriched_output, args.writer, args.num_output_files)
            columns = read_parquet(spark, args.enriched_output).columns
        rows_back = parquet_row_count(args.enriched_output)  # doc nguoc so dong tu footer file Parquet
        report["enriched_output"] = dict(path=args.enriched_output, writer=used, rows=rows_back,
                                         columns=columns, rows_match_ratings=(rows_back == report["ratings_rows"]))
        logger.info("Da ghi bang join (%s): %s | %d dong", used, args.enriched_output, rows_back)
        if rows_back != report["ratings_rows"]:
            report["enriched_output"]["error"] = "So dong bang join khac so dong ratings"

        save_report()                                        # cap nhat them muc enriched_output
    spark.stop()
    if report.get("enriched_output", {}).get("error"):
        raise SystemExit(report["enriched_output"]["error"])


if __name__ == "__main__":
    main()