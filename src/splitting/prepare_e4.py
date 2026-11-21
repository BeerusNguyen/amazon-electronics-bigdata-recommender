"""Build one deterministic cohort and split it on the agreed dates."""

from datetime import datetime, timezone

import pandas as pd
from pyspark.sql import functions as F, types as T

from src.etl.read_ratings import CSV_SCHEMA, OUTPUT_COLUMNS
from src.evaluation.split_data import split_by_time


def unix_date(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())


BOUNDARIES = (unix_date("2017-02-01"), unix_date("2017-10-01"))
SPLIT_NAMES = ("train", "validation", "test")


def first_interactions(valid):
    # Process all timestamps of each sampled pair. The earliest group's min/max
    # distinguishes exact duplicates from conflicting scores at that timestamp.
    daily = valid.groupBy("user_id", "product_id", "timestamp").agg(
        F.count("*").alias("rows"),
        F.countDistinct("rating").alias("distinct_ratings"),
        F.min("rating").alias("low"),
        F.max("rating").alias("high"),
    )
    pairs = daily.groupBy("user_id", "product_id").agg(
        F.min(F.struct("timestamp", "low", "high")).alias("first"),
        F.sum("rows").alias("valid_rows"),
        F.sum("distinct_ratings").alias("exact_unique_rows"),
    )
    return pairs.select(
        "user_id", "product_id", "valid_rows", "exact_unique_rows",
        F.col("first.timestamp").alias("timestamp"),
        F.col("first.low").alias("rating"),
        (F.col("first.low") != F.col("first.high")).alias("conflict"),
    )


def prepare(spark, csv_path, seed, cohort_bps, destination):
    corrupt = "_corrupt_record"
    schema = T.StructType([*CSV_SCHEMA.fields, T.StructField(corrupt, T.StringType())])
    spark.conf.set("spark.sql.csv.parser.columnPruning.enabled", "false")
    raw = (spark.read.schema(schema).option("header", "false")
           .option("columnNameOfCorruptRecord", corrupt).csv(str(csv_path)))
    raw = raw.withColumn("user_id", F.trim("user_id")).withColumn("product_id", F.trim("product_id"))
    valid = F.coalesce(
        F.col(corrupt).isNull() & (F.length("user_id") > 0) & (F.length("product_id") > 0)
        & F.col("rating").isin(1., 2., 3., 4., 5.)
        & (F.col("timestamp") > 0) & (F.col("timestamp") < unix_date("2018-10-06")),
        F.lit(False),
    )
    selected = F.pmod(F.xxhash64("user_id", F.lit(seed)), F.lit(10000)) < cohort_bps
    def count_if(condition):
        return F.count(F.when(condition, 1))

    source_stats = raw.agg(
        F.count("*").alias("source_rows"),
        count_if(valid).alias("source_valid_rows"),
        count_if(valid & selected).alias("cohort_valid_rows"),
    ).first().asDict()
    cohort = raw.where(valid & selected).select(*OUTPUT_COLUMNS)
    pairs = first_interactions(cohort).persist()
    try:
        totals = pairs.agg(
            F.sum("valid_rows").alias("valid_rows"),
            F.sum("exact_unique_rows").alias("exact_unique_rows"),
            F.count("*").alias("pairs_before_conflict"),
            count_if(F.col("conflict")).alias("earliest_conflicts"),
        ).first().asDict()
        if not totals["pairs_before_conflict"]:
            raise ValueError("The selected cohort has no valid pairs")
        audit = {
            **source_stats,
            "source_invalid_rows": source_stats["source_rows"] - source_stats["source_valid_rows"],
            **totals,
            "exact_duplicates_removed": totals["valid_rows"] - totals["exact_unique_rows"],
            "additional_ratings_removed": totals["exact_unique_rows"] - totals["pairs_before_conflict"],
            "clean_pairs": totals["pairs_before_conflict"] - totals["earliest_conflicts"],
        }
        frames = {}
        clean = pairs.where(~F.col("conflict")).select(*OUTPUT_COLUMNS)
        for name, split in zip(SPLIT_NAMES, split_by_time(clean, *BOUNDARIES)):
            frame = split.toPandas()
            if frame.empty:
                raise ValueError(f"{name} is empty")
            if frame.duplicated(["user_id", "product_id"]).any():
                raise ValueError(f"{name} contains repeated user–item keys")
            frame.to_parquet(destination / f"{name}.parquet", index=False)
            frames[name] = frame
    finally:
        pairs.unpersist()

    if not (audit["cohort_valid_rows"] == audit["valid_rows"]
            and audit["clean_pairs"] == sum(map(len, frames.values()))):
        raise RuntimeError("Input and split counts do not reconcile")
    if not (frames["train"].timestamp.max() < BOUNDARIES[0]
            <= frames["validation"].timestamp.min() <= frames["validation"].timestamp.max()
            < BOUNDARIES[1] <= frames["test"].timestamp.min()):
        raise RuntimeError("Split timestamps violate the UTC boundaries")
    if pd.concat(frames.values(), ignore_index=True).duplicated(["user_id", "product_id"]).any():
        raise RuntimeError("A user–item pair appears in more than one split")
    return frames, audit
