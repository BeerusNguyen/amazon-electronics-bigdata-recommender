"""Run E4 from the repository root with python -m src.splitting.run_e4."""

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.common.spark_utils import get_spark
from src.splitting.audit_e4 import run_audit
from src.splitting.prepare_e4 import BOUNDARIES, SPLIT_NAMES, prepare


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def iso_date(seconds):
    return datetime.fromtimestamp(int(seconds), timezone.utc).isoformat()


def source_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_run(spark, args):
    if not 1 <= args.cohort_bps <= 10000:
        raise ValueError("Cohort basis points must be in [1, 10000]")
    if not args.csv.is_file():
        raise FileNotFoundError(args.csv)
    args.out.mkdir(parents=True, exist_ok=False)
    frames, audit = prepare(spark, args.csv, args.seed, args.cohort_bps, args.out)
    stats = []
    for name, frame in frames.items():
        stats.append({
            "split": name,
            "pairs": len(frame),
            "users": frame.user_id.nunique(),
            "items": frame.product_id.nunique(),
            "relevant_pairs": int(frame.rating.ge(4).sum()),
            "share": len(frame) / audit["clean_pairs"],
            "min_utc": iso_date(frame.timestamp.min()),
            "max_utc": iso_date(frame.timestamp.max()),
        })
    pd.DataFrame(stats).to_csv(args.out / "split_statistics.csv", index=False)
    write_json(args.out / "split_manifest.json", {
        "ratings_source": args.csv.name,
        "ratings_sha256": source_hash(args.csv),
        "spark_version": spark.version,
        "master": spark.sparkContext.master,
        "seed": args.seed,
        "user_cohort_basis_points": args.cohort_bps,
        "validation_start_utc": iso_date(BOUNDARIES[0]),
        "test_start_utc": iso_date(BOUNDARIES[1]),
        "review_conflict_policy": "drop pairs with conflicting scores at the earliest timestamp",
        "audit": audit,
    })
    print(f"Prepared {audit['clean_pairs']:,} pairs in {args.out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--csv", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--cohort-bps", type=int, required=True)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--shuffle-partitions", type=int, default=24)
    audit = commands.add_parser("audit")
    audit.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "audit":
        frames = {name: pd.read_parquet(args.out / f"{name}.parquet") for name in SPLIT_NAMES}
        run_audit(frames, args.out)
        print(f"Saved cold-start audit and ID maps to {args.out / 'audit'}")
        return

    spark = get_spark("E4-prepare", args.shuffle_partitions)
    try:
        prepare_run(spark, args)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
