"""T12 - Chuan hoa metadata, xu ly null/duplicate, khu trung ASIN. Pipeline CHUNG cho moi truong khai bao
trong etl/metadata_fields.py (them truong o do, khong can sua file nay).

Thu tu:
  R1  asin null/blank                       -> loai dong
  --  lam sach tung truong (cleaner cua FieldSpec; brand: bo "Visit the X Store", (R)/(TM), placeholder N/A... -> null)
  R2  duplicate ASIN                        -> giu ban ghi DAY DU nhat, roi DIEN cac truong thieu tu cac ban ghi trung
  R3+ canonicalize_<truong>                 -> gop bien the viet khac nhau (SONY/Sony/sony -> dang pho bien nhat)
  R*  fill_null_<truong>                    -> dien gia tri mac dinh (neu FieldSpec.fill_null)

Bao cao:
  results/metadata_quality.csv       so dong truoc/sau moi buoc (doi chieu raw - removed = clean)
  results/metadata_field_report.csv  moi truong: null truoc/sau lam sach, duoc dien tu ban ghi trung, null cuoi, distinct
  results/metadata_variants.json     vi du cac nhom bien the da gop (de ban kiem tra bang mat)

Chay:
    python src/etl/clean_metadata.py --input data/raw/meta_Electronics.json.gz \
        --output data/processed/metadata_clean --report results/metadata_quality.csv
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # them thu muc src/

import pandas as pd  # noqa: E402
from pyspark.sql import DataFrame, Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from common.spark_utils import get_spark, storage_level, write_parquet  # noqa: E402
from etl.metadata_fields import (KEY, active_derived, canonicalize_variants, kept_fields,  # noqa: E402
                                 output_columns)
from etl.read_metadata import read_metadata  # noqa: E402

logger = logging.getLogger("clean_metadata")


def _cnt(cond):
    return F.sum(F.when(cond, 1).otherwise(0))


def normalize(df: DataFrame, specs) -> DataFrame:
    """Ap cleaner cho tung truong; giu them cot _raw_<truong> (gia tri goc) de do thong ke."""
    cols = [F.trim(F.col(KEY)).alias(KEY)]
    for s in specs:
        raw = F.col(s.name)
        cols += [s.clean(raw).alias(s.name), raw.alias(f"_raw_{s.name}")]
    out = df.select(*cols)
    for name, _dtype, builder in active_derived(s.name for s in specs):
        out = out.withColumn(name, builder())
    return out


def stage_stats(df: DataFrame, specs) -> dict:
    """Thong ke truoc khu trung: thieu truoc/sau lam sach, so gia tri distinct."""
    aggs = []
    for s in specs:
        raw, cl = F.col(f"_raw_{s.name}"), F.col(s.name)
        aggs += [_cnt(s.is_missing(raw)).alias(f"{s.name}__raw_missing"),
                 _cnt(s.is_missing(cl)).alias(f"{s.name}__clean_missing")]
        if not s.is_array:
            aggs += [F.countDistinct(F.trim(raw)).alias(f"{s.name}__distinct_raw"),
                     F.countDistinct(cl).alias(f"{s.name}__distinct_clean")]
    return {k: int(v or 0) for k, v in df.agg(*aggs).first().asDict().items()}


def prune_raw(df: DataFrame, specs) -> DataFrame:
    """Giu _raw_<truong> thanh <truong>_raw neu keep_raw, con lai bo."""
    for s in specs:
        if s.keep_raw:
            df = df.withColumnRenamed(f"_raw_{s.name}", f"{s.name}_raw")
        else:
            df = df.drop(f"_raw_{s.name}")
    return df


def dedup_asin(df: DataFrame, specs):
    """Khu trung ASIN: giu ban ghi day du nhat; truong con thieu duoc dien tu cac ban ghi trung theo thu tu uu tien.
    Tra ve (df, {truong: so gia tri duoc dien})."""
    comp = [s for s in specs if s.completeness]
    completeness = sum((~s.is_missing(F.col(s.name))).cast("int") for s in comp) if comp else F.lit(0)
    richness = sum((F.coalesce(F.size(F.col(s.name)), F.lit(0)) if s.is_array
                    else F.length(F.coalesce(F.col(s.name), F.lit(""))) for s in comp)) if comp else F.lit(0)
    tie = [F.col(s.name).asc_nulls_last() for s in specs if not s.is_array]
    tie += [F.col(n).asc_nulls_last() for n, dt, _ in active_derived(s.name for s in specs)
            if dt.simpleString() == "string"]

    w = Window.partitionBy(KEY).orderBy(completeness.desc(), richness.desc(), *tie)
    w_all = w.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

    merge = [s for s in specs if s.merge_duplicates]
    d = df.withColumn("_rn", F.row_number().over(w))
    for s in merge:
        c = F.col(s.name)
        m = F.first(F.when(~s.is_missing(c), c), ignorenulls=True).over(w_all)
        if s.is_array:
            m = F.coalesce(m, F.array().cast("array<string>"))
        d = d.withColumn(f"_m_{s.name}", m)
    d = d.where("_rn = 1").drop("_rn").persist(storage_level())

    filled = {}
    if merge:
        row = d.agg(*[_cnt(s.is_missing(F.col(s.name)) & ~s.is_missing(F.col(f"_m_{s.name}"))).alias(s.name)
                      for s in merge]).first().asDict()
        filled = {k: int(v or 0) for k, v in row.items()}
    for s in merge:
        d = d.withColumn(s.name, F.col(f"_m_{s.name}")).drop(f"_m_{s.name}")
    return d, filled


def fill_nulls(df: DataFrame, spec) -> DataFrame:
    c = F.col(spec.name)
    return df.withColumn(spec.name, F.when(spec.is_missing(c), F.lit(spec.fill_null)).otherwise(c))


def clean_metadata(raw: DataFrame, specs=None):
    """Tra ve (clean_df, report_rows, info) voi info = {"fields": [...], "variants": {...}}."""
    specs = specs or kept_fields()
    report = []
    cur = raw.persist(storage_level())
    n = cur.count()
    report.append(dict(step=0, rule="raw_input", rows_before=n, rows_after=n, rows_removed=0,
                       description="So dong doc duoc"))

    def apply(step, rule, desc, nxt):
        nonlocal cur
        before = report[-1]["rows_after"]
        nxt = nxt.persist(storage_level())
        after = nxt.count()
        cur.unpersist()
        cur = nxt
        report.append(dict(step=step, rule=rule, rows_before=before, rows_after=after,
                           rows_removed=before - after, description=desc))
        logger.info("%s: %d -> %d", rule, before, after)

    apply(1, "asin_null_or_blank", "Loai dong khong co asin (gom ca dong JSON hong)",
          normalize(cur, specs).filter(F.col(KEY).isNotNull() & (F.col(KEY) != "")))
    stats_a = stage_stats(cur, specs)

    deduped, filled = dedup_asin(prune_raw(cur, specs), specs)
    apply(2, "duplicate_asin", "Khu trung ASIN: giu ban day du nhat, dien truong thieu tu ban ghi trung", deduped)

    step, variants = 3, {}
    for s in specs:
        if s.canonicalize_variants and not s.is_array:
            new_df, st = canonicalize_variants(cur, s.name)
            variants[s.name] = st
            apply(step, f"canonicalize_{s.name}",
                  f"Gop bien the {s.name}: {st['distinct_before']} -> {st['distinct_after']} gia tri distinct "
                  f"({st['rows_changed']} dong doi, {st['variant_groups']} nhom)", new_df)
            step += 1
    for s in specs:
        if s.fill_null is not None and not s.is_array:
            apply(step, f"fill_null_{s.name}", f"Dien '{s.fill_null}' cho {s.name} con thieu", fill_nulls(cur, s))
            step += 1

    final = cur.select(*[name for name, _ in output_columns(specs)])
    n_final = report[-1]["rows_after"]
    if n - sum(r["rows_removed"] for r in report) != n_final:
        raise RuntimeError("Doi chieu that bai: raw - removed != clean")
    if final.select(KEY).distinct().count() != n_final:
        raise RuntimeError("ASIN khong duy nhat sau khi khu trung")

    aggs = []
    for s in specs:
        aggs.append(_cnt(s.is_missing(F.col(s.name))).alias(f"{s.name}__missing"))
        if not s.is_array:
            aggs.append(F.countDistinct(F.col(s.name)).alias(f"{s.name}__distinct"))
    fin = {k: int(v or 0) for k, v in final.agg(*aggs).first().asDict().items()}

    fields = []
    for s in specs:
        raw_m, clean_m = stats_a[f"{s.name}__raw_missing"], stats_a[f"{s.name}__clean_missing"]
        st = variants.get(s.name)
        fields.append(dict(
            field=s.name, rows_final=n_final,
            raw_missing_pre_dedup=raw_m, missing_after_cleaning_pre_dedup=clean_m,
            newly_missing_from_cleaning_pre_dedup=clean_m - raw_m,
            filled_from_duplicates=filled.get(s.name),
            missing_final=fin[f"{s.name}__missing"],
            missing_final_rate=round(fin[f"{s.name}__missing"] / n_final, 4) if n_final else None,
            distinct_raw_pre_dedup=stats_a.get(f"{s.name}__distinct_raw"),
            distinct_after_cleaning_pre_dedup=stats_a.get(f"{s.name}__distinct_clean"),
            distinct_final=fin.get(f"{s.name}__distinct"),
            variants_rows_changed=st["rows_changed"] if st else None,
            variants_groups_merged=st["variant_groups"] if st else None))
    return final, report, dict(fields=fields, variants=variants)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Normalize metadata, handle null/duplicates, deduplicate ASIN")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="data/processed/metadata_clean")
    ap.add_argument("--report", default="results/metadata_quality.csv")
    ap.add_argument("--field-report", default="results/metadata_field_report.csv")
    ap.add_argument("--variants-report", default="results/metadata_variants.json")
    ap.add_argument("--num-output-files", type=int, default=4)
    ap.add_argument("--writer", default="auto", choices=["auto", "spark", "pandas"],
                    help="auto: thu Spark, loi (Hadoop/Windows) thi chuyen sang pandas")
    args = ap.parse_args()

    spark = get_spark("clean_metadata")
    clean, report, info = clean_metadata(read_metadata(spark, args.input))

    for path in (args.report, args.field_report, args.variants_report):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(report).to_csv(args.report, index=False)                 # ghi bao cao truoc khi ghi Parquet
    pd.DataFrame(info["fields"]).to_csv(args.field_report, index=False)
    with open(args.variants_report, "w", encoding="utf-8") as f:
        json.dump(info["variants"], f, indent=2, ensure_ascii=False)

    used = write_parquet(clean, args.output, args.writer, args.num_output_files)
    logger.info("Writer da dung: %s", used)
    logger.info("Da ghi %s | clean rows (ASIN duy nhat) = %d", args.output, report[-1]["rows_after"])
    spark.stop()


if __name__ == "__main__":
    main()