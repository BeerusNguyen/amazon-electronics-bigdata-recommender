"""T15 (tuy chon) - So sanh CSV vs Parquet tren CUNG workload Spark.

Cong bang: cung du lieu (ratings tu CSV goc, chua clean, doc voi schema co dinh), cung cau hinh Spark,
cung truy van. Chi phi convert CSV->Parquet duoc do rieng.
Workload: read -> filter(rating >= 4) -> groupBy(product_id) -> count/avg -> tong hop.
Moi dinh dang chay --repeats lan (khong cache DataFrame; luu y OS co the cache file, lan 1 thuong cham nhat).
Hai dinh dang phai cho cung ket qua, neu khong script bao loi.

Chay:
    python src/etl/csv_parquet_benchmark.py --csv data/raw/Electronics.csv --repeats 3
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # them thu muc src/

import pandas as pd  # noqa: E402
from pyspark.sql import DataFrame  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from common.spark_utils import get_spark, read_parquet, write_parquet  # noqa: E402
from etl.read_ratings import header_arg, read_ratings  # noqa: E402

logger = logging.getLogger("csv_parquet_benchmark")


def workload(df: DataFrame) -> dict:
    per_item = (df.filter(F.col("rating") >= 4.0).groupBy("product_id")
                .agg(F.count(F.lit(1)).alias("n"), F.avg("rating").alias("avg_rating")))
    r = per_item.agg(F.count(F.lit(1)).alias("items"), F.sum("n").alias("positive_ratings")).first()
    return dict(result_items=int(r["items"] or 0), result_positive=int(r["positive_ratings"] or 0))


def dir_bytes(path: str) -> int:
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.name.startswith(("_", ".")))


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, round(time.perf_counter() - t0, 3)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="CSV vs Parquet benchmark (same Spark workload)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--header", default="auto", choices=["auto", "true", "false"])
    ap.add_argument("--parquet-dir", default="data/interim/ratings_raw_parquet",
                    help="Noi ghi ban Parquet cua CSV goc (chua clean) de so sanh cong bang")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--num-files", type=int, default=8)
    ap.add_argument("--writer", default="auto", choices=["auto", "spark", "pandas"])
    ap.add_argument("--output", default="results/csv_parquet_benchmark.csv")
    ap.add_argument("--figure", default="results/figures/csv_parquet_benchmark.png")
    args = ap.parse_args()

    spark = get_spark("csv_parquet_benchmark")
    config = f"shuffle_partitions={spark.conf.get('spark.sql.shuffle.partitions')};driver_memory={spark.conf.get('spark.driver.memory')}"
    rows = []

    csv_df = read_ratings(spark, args.csv, "csv", header_arg(args.header))
    _, convert_s = timed(lambda: write_parquet_used(csv_df, args.parquet_dir, args.writer, args.num_files))
    writer_used = write_parquet_used.last
    rows.append(dict(format="convert_csv_to_parquet", run=0, seconds=convert_s, result_items=None,
                     result_positive=None, size_bytes=dir_bytes(args.parquet_dir), config=config,
                     note=f"writer={writer_used}; chi phi mot lan, khong tinh vao workload"))

    sources = {"csv": (lambda: read_ratings(spark, args.csv, "csv", header_arg(args.header)), dir_bytes(args.csv)),
               "parquet": (lambda: read_parquet(spark, args.parquet_dir), dir_bytes(args.parquet_dir))}
    results = {}
    for fmt, (loader, size) in sources.items():
        for run in range(1, args.repeats + 1):
            res, secs = timed(lambda: workload(loader()))
            results.setdefault(fmt, res)
            if res != results[fmt]:
                raise RuntimeError(f"Ket qua {fmt} khong on dinh giua cac lan chay")
            rows.append(dict(format=fmt, run=run, seconds=secs, result_items=res["result_items"],
                             result_positive=res["result_positive"], size_bytes=size, config=config, note=""))
            logger.info("%s run %d: %.2fs", fmt, run, secs)
    if results["csv"] != results["parquet"]:
        raise RuntimeError(f"CSV va Parquet cho ket qua khac nhau: {results}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    summary = df[df["format"].isin(["csv", "parquet"])].groupby("format")["seconds"].agg(["mean", "min", "max"])
    logger.info("Tom tat (giay):\n%s", summary.round(2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(args.figure) or ".", exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(summary.index, summary["mean"], yerr=[summary["mean"] - summary["min"], summary["max"] - summary["mean"]], capsize=4)
        ax.set_ylabel("Elapsed time (s), mean with min-max")
        ax.set_title("Same Spark workload: CSV vs Parquet")
        fig.tight_layout()
        fig.savefig(args.figure, dpi=150)
        logger.info("Da ghi bieu do %s", args.figure)
    except ImportError:
        logger.warning("Chua cai matplotlib nen bo qua bieu do (pip install matplotlib).")
    spark.stop()


def write_parquet_used(df, path, writer, num_files):
    write_parquet_used.last = write_parquet(df, path, writer, num_files)
    return write_parquet_used.last


if __name__ == "__main__":
    main()