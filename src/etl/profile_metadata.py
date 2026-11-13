"""T06 - Profile chat luong Metadata JSONL.

Do: so dong, dong JSON hong, ASIN duy nhat/trung, ty le null/blank cua title/brand/category/main_cat,
HTML con sot trong title/main_cat, dinh dang price, do sau category.
Ket qua: results/metadata_profile.json

Chay:
    python src/etl/profile_metadata.py --input data/raw/meta_Electronics.json.gz \
        --output results/metadata_profile.json
"""
import argparse
import json
import logging
import os
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # them thu muc src/ de import etl.*, common.*

from pyspark.sql import functions as F  # noqa: E402

from common.spark_utils import get_spark  # noqa: E402
from etl.read_metadata import read_metadata  # noqa: E402

logger = logging.getLogger("profile_metadata")

HTML_RE = r"<[^>]+>"
PRICE_SINGLE = r"^\$\d{1,3}(,\d{3})*(\.\d{1,2})?$"
PRICE_RANGE = r"^\$\d{1,3}(,\d{3})*(\.\d{1,2})?\s*-\s*\$\d{1,3}(,\d{3})*(\.\d{1,2})?$"


def _blank(c):
    return F.col(c).isNull() | (F.trim(F.col(c)) == "")


def _cnt(cond):
    return F.sum(F.when(cond, 1).otherwise(0))


def profile(df):
    price_nonblank = ~_blank("price")
    row = df.agg(
        F.count(F.lit(1)).alias("total_rows"),
        _cnt(F.col("_corrupt_record").isNotNull()).alias("corrupt_records"),
        _cnt(_blank("asin")).alias("asin_blank"),
        _cnt(_blank("title")).alias("title_blank"),
        _cnt(_blank("brand")).alias("brand_blank"),
        _cnt(_blank("main_cat")).alias("main_cat_blank"),
        _cnt(F.col("category").isNull() | (F.size("category") == 0)).alias("category_empty"),
        _cnt(F.col("title").rlike(HTML_RE)).alias("title_has_html"),
        _cnt(F.col("brand").rlike(HTML_RE)).alias("brand_has_html"),
        _cnt(F.col("main_cat").rlike(HTML_RE)).alias("main_cat_has_html"),
        _cnt(~price_nonblank).alias("price_blank"),
        _cnt(price_nonblank & F.col("price").rlike(PRICE_SINGLE)).alias("price_single_valid"),
        _cnt(price_nonblank & F.col("price").rlike(PRICE_RANGE)).alias("price_range"),
        _cnt(price_nonblank & ~F.col("price").rlike(PRICE_SINGLE) & ~F.col("price").rlike(PRICE_RANGE)).alias("price_malformed"),
    ).first().asDict()
    out = {k: int(v or 0) for k, v in row.items()}

    keyed = df.filter(~_blank("asin"))
    per_asin = keyed.groupBy("asin").count()
    n_distinct = per_asin.count()
    dup = per_asin.filter(F.col("count") > 1).agg(
        F.count(F.lit(1)).alias("n"), F.sum(F.col("count") - 1).alias("extra")).first()
    out.update(distinct_asin=n_distinct,
               asin_with_duplicates=int(dup["n"] or 0),
               extra_duplicate_rows=int(dup["extra"] or 0))

    total = max(out["total_rows"], 1)
    out["rates"] = {k.replace("_blank", "_blank_rate").replace("_empty", "_empty_rate"): round(v / total, 4)
                    for k, v in out.items()
                    if k in ("title_blank", "brand_blank", "main_cat_blank", "category_empty", "price_blank")}

    out["category_depth"] = {str(r["d"]): r["count"] for r in
                             df.select(F.coalesce(F.size("category"), F.lit(0)).alias("d"))
                               .groupBy("d").count().orderBy("d").collect()}
    out["main_cat_top15"] = [{"value": (r["main_cat"] or "")[:80], "count": r["count"]} for r in
                             df.groupBy("main_cat").count().orderBy(F.desc("count")).limit(15).collect()]
    out["price_malformed_examples"] = [r["price"][:100] for r in
                                       df.filter(price_nonblank & ~F.col("price").rlike(PRICE_SINGLE)
                                                 & ~F.col("price").rlike(PRICE_RANGE))
                                         .select("price").limit(10).collect()]
    out["title_html_examples"] = [r["title"][:100] for r in
                                  df.filter(F.col("title").rlike(HTML_RE)).select("title").limit(5).collect()]
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Profile metadata quality")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="results/metadata_profile.json")
    args = ap.parse_args()

    t0 = time.time()
    spark = get_spark("profile_metadata")
    result = profile(read_metadata(spark, args.input, with_corrupt=True))
    result.update(input=args.input, runtime_seconds=round(time.time() - t0, 1),
                  field_decisions=dict(
                      keep=["asin", "title", "brand", "main_cat", "category"],
                      profile_only=["price"],
                      drop=["rank (kieu hon hop)", "details", "tech1", "tech2", "fit", "similar_item",
                            "date", "feature", "description", "also_buy", "also_view",
                            "imageURL", "imageURLHighRes"]))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Da ghi %s", args.output)
    spark.stop()


if __name__ == "__main__":
    main()
