"""T13 - Chung minh Parquet (ratings + metadata) ghi/doc end-to-end va do dung luong.

Viec GHI Parquet do clean_ratings.py / clean_metadata.py thuc hien (qua common.spark_utils.write_parquet).
Script nay DOC NGUOC cac thu muc Parquet da ghi va doi chieu:
  * so dong  == dong cuoi trong results/<dataset>_quality.csv (neu co)
  * schema   == schema ky vong
  * khoa duy nhat: metadata theo asin; ratings theo (user_id, product_id)
  * so file output, dung luong, so voi file goc (--raw-ratings .csv, --raw-metadata .json.gz)
Ket qua: results/storage_stats.csv. Kiem tra nao that bai -> exit code != 0.
Dataset nao chua co thu muc Parquet thi bi bo qua (co canh bao).

Chay:
    python src/etl/write_parquet.py --raw-ratings data/raw/Electronics.csv \
        --raw-metadata data/raw/meta_Electronics.json.gz
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # them thu muc src/

import pandas as pd  # noqa: E402

from common.spark_utils import get_spark, read_parquet  # noqa: E402
from etl.metadata_fields import output_columns  # noqa: E402

logger = logging.getLogger("write_parquet")

DATASETS = {
    "metadata": dict(
        dtypes=output_columns(),          # sinh tu FIELDS trong etl/metadata_fields.py
        key=["asin"], report="results/metadata_quality.csv"),
    "ratings": dict(
        dtypes=[("product_id", "string"), ("user_id", "string"), ("rating", "double"), ("timestamp", "bigint")],
        key=["user_id", "product_id"], report="results/ratings_quality.csv"),
}


def parquet_dir_stats(path: str):
    """(so file .parquet, tong byte) cua thu muc local."""
    files = [f for f in Path(path).rglob("*.parquet")]
    return len(files), sum(f.stat().st_size for f in files)


def detect_writer(path: str) -> str:
    """Doan writer da ghi thu muc Parquet tu ten file: Spark dat ten '...-c000.snappy.parquet', pandas/pyarrow dat 'part-0000N.parquet'."""
    names = [f.name for f in Path(path).rglob("*.parquet")]
    if not names:
        return "unknown"
    return "spark" if all("-c000" in n for n in names) else "pandas/pyarrow"


def verify(spark, parquet_path: str, dtypes, key_cols, expected_rows: int = None) -> dict:
    df = read_parquet(spark, parquet_path)
    t0 = time.time()
    rows = df.count()
    read_seconds = round(time.time() - t0, 2)
    distinct_keys = df.select(*key_cols).distinct().count()
    return dict(
        rows=rows, distinct_keys=distinct_keys, read_count_seconds=read_seconds,
        schema_match=df.dtypes == dtypes,
        key_unique=distinct_keys == rows,
        rows_match_report=None if expected_rows is None else rows == expected_rows,
        dtypes=df.dtypes,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Verify processed Parquet artifacts and record storage stats")
    ap.add_argument("--ratings-parquet", default="data/processed/ratings_clean")
    ap.add_argument("--metadata-parquet", default="data/processed/metadata_clean")
    ap.add_argument("--raw-ratings", default=None, help="CSV goc de so sanh dung luong")
    ap.add_argument("--raw-metadata", default=None, help="json.gz goc de so sanh dung luong")
    ap.add_argument("--output", default="results/storage_stats.csv")
    args = ap.parse_args()

    plan = {"ratings": (args.ratings_parquet, args.raw_ratings),
            "metadata": (args.metadata_parquet, args.raw_metadata)}
    spark = get_spark("write_parquet_verify")
    rows_out, failures, ran = [], [], 0

    for name, (pq, raw) in plan.items():
        if not os.path.isdir(pq):
            logger.warning("Bo qua %s: khong thay thu muc Parquet %s", name, pq)
            continue
        ran += 1
        spec = DATASETS[name]
        expected = None
        if os.path.exists(spec["report"]):
            expected = int(pd.read_csv(spec["report"]).iloc[-1]["rows_after"])
        v = verify(spark, pq, spec["dtypes"], spec["key"], expected)
        n_files, size = parquet_dir_stats(pq)

        raw_size = os.path.getsize(raw) if raw and os.path.exists(raw) else None
        if raw_size is not None:
            rows_out.append(dict(dataset=name, artifact=f"{name}_raw", path=raw, format=Path(raw).suffix.lstrip("."),
                                 files=1, size_bytes=raw_size, size_mb=round(raw_size / 1e6, 2), rows=None,
                                 ratio_vs_raw=1.0))
        rows_out.append(dict(dataset=name, artifact=f"{name}_clean_parquet", path=pq, format="parquet",
                             files=n_files, size_bytes=size, size_mb=round(size / 1e6, 2), rows=v["rows"],
                             writer_detected=detect_writer(pq),
                             ratio_vs_raw=round(size / raw_size, 4) if raw_size else None))
        checks = {k: v[k] for k in ("schema_match", "key_unique", "rows_match_report")}
        logger.info("[%s] rows=%d files=%d writer=%s size=%.1f MB read_count=%.1fs checks=%s",
                    name, v["rows"], n_files, detect_writer(pq), size / 1e6, v["read_count_seconds"], checks)
        if any(x is False for x in checks.values()):
            failures.append(f"{name}: {checks}; dtypes={v['dtypes']}")

    if ran == 0:
        raise SystemExit("Khong co thu muc Parquet nao de kiem tra.")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pd.DataFrame(rows_out).to_csv(args.output, index=False)
    spark.stop()
    if failures:
        raise SystemExit("Kiem tra Parquet that bai: " + " | ".join(failures))


if __name__ == "__main__":
    main()