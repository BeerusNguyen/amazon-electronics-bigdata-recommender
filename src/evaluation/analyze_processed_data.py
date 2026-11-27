"""Profile processed Parquet datasets and validate the E4 temporal splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


SPLIT_NAMES = ("train", "validation", "test")
VALIDATION_START = 1485907200
TEST_START = 1506816000


def human_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def parquet_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(path for path in root.rglob("*.parquet") if path.is_file())


def key_columns(columns: Iterable[str]) -> tuple[str, ...]:
    names = set(columns)
    if {"user_id", "product_id"} <= names:
        return ("user_id", "product_id")
    if "asin" in names:
        return ("asin",)
    if "user_id" in names:
        return ("user_id",)
    if "product_id" in names:
        return ("product_id",)
    return ()


def profile_file(path: Path) -> dict:
    frame = pd.read_parquet(path)
    result = {
        "file": str(path).replace("\\", "/"),
        "relative_file": path.as_posix(),
        "bytes": path.stat().st_size,
        "rows": len(frame),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "null_counts": {column: int(frame[column].isna().sum()) for column in frame.columns},
    }
    keys = key_columns(frame.columns)
    if keys:
        duplicate_mask = frame.duplicated(list(keys), keep=False)
        result["key_columns"] = list(keys)
        result["key_unique_rows"] = int(len(frame) - duplicate_mask.sum())
        result["key_duplicate_rows"] = int(duplicate_mask.sum())
    for column in frame.select_dtypes(include=["number"]).columns:
        series = frame[column].dropna()
        if len(series):
            result.setdefault("numeric", {})[column] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
            }
    return result


def profile_tree(root: Path) -> list[dict]:
    files = parquet_files(root)
    groups: dict[Path, list[Path]] = {}
    for path in files:
        groups.setdefault(path.parent, []).append(path)

    profiles = []
    for directory in sorted(groups):
        file_profiles = [profile_file(path) for path in groups[directory]]
        columns = sorted(set().union(*(set(profile["columns"]) for profile in file_profiles)))
        row_count = sum(profile["rows"] for profile in file_profiles)
        size = sum(profile["bytes"] for profile in file_profiles)
        null_counts = {column: 0 for column in columns}
        dtypes = {}
        for profile in file_profiles:
            dtypes.update(profile["dtypes"])
            for column, count in profile["null_counts"].items():
                null_counts[column] += count

        key_sets = {tuple(profile["key_columns"]) for profile in file_profiles if "key_columns" in profile}
        profiles.append({
            "dataset": directory.relative_to(root).as_posix() or ".",
            "file_count": len(file_profiles),
            "rows": row_count,
            "bytes": size,
            "human_size": human_bytes(size),
            "columns": columns,
            "dtypes": dtypes,
            "null_counts": null_counts,
            "key_columns": list(sorted(key_sets)[0]) if len(key_sets) == 1 else [],
            "key_duplicate_rows": sum(profile.get("key_duplicate_rows", 0) for profile in file_profiles),
            "files": file_profiles,
        })
    return profiles


def validate_e4(e4_dir: Path) -> dict:
    frames = {}
    checks = []

    for name in SPLIT_NAMES:
        path = e4_dir / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frames[name] = pd.read_parquet(path)

    train = frames["train"]
    validation = frames["validation"]
    test = frames["test"]
    combined = pd.concat(frames.values(), ignore_index=True)

    checks.append({
        "name": "split_files_present",
        "passed": all((e4_dir / f"{name}.parquet").is_file() for name in SPLIT_NAMES),
        "detail": "train.parquet, validation.parquet and test.parquet are present",
    })
    checks.append({
        "name": "pair_unique_in_each_split",
        "passed": all(not frame.duplicated(["user_id", "product_id"]).any() for frame in frames.values()),
        "detail": "No repeated user-item pair inside each split",
    })
    checks.append({
        "name": "pair_unique_across_splits",
        "passed": not combined.duplicated(["user_id", "product_id"]).any(),
        "detail": "No user-item pair appears in more than one split",
    })
    checks.append({
        "name": "temporal_boundaries",
        "passed": bool(
            train.timestamp.max() < VALIDATION_START
            and validation.timestamp.min() >= VALIDATION_START
            and validation.timestamp.max() < TEST_START
            and test.timestamp.min() >= TEST_START
        ),
        "detail": "Train < 2017-02-01 <= Validation < 2017-10-01 <= Test",
    })
    checks.append({
        "name": "ratings_in_valid_range",
        "passed": bool(combined.rating.between(1, 5).all()),
        "detail": "Every rating is an integer-valued number from 1 to 5",
    })

    audit_dir = e4_dir / "audit"
    cold_start = audit_dir / "cold_start.csv"
    if cold_start.is_file():
        cold = pd.read_csv(cold_start)
        cohort_totals = (
            cold[cold.group.str.startswith("warm_") | cold.group.str.startswith("cold_")]
            .groupby("phase").pairs.sum().sort_index()
        )
        held_out_totals = (
            cold[cold.group == "all"].set_index("phase").pairs.sort_index()
        )
        cohort_total_ok = bool(cohort_totals.equals(held_out_totals))
        checks.append({
            "name": "cold_start_cohort_totals",
            "passed": cohort_total_ok,
            "detail": "Four cold-start cohorts reconcile to each held-out phase",
        })

    split_stats = []
    for name, frame in frames.items():
        split_stats.append({
            "split": name,
            "pairs": len(frame),
            "users": int(frame.user_id.nunique()),
            "items": int(frame.product_id.nunique()),
            "relevant_pairs": int(frame.rating.ge(4).sum()),
            "min_timestamp": int(frame.timestamp.min()),
            "max_timestamp": int(frame.timestamp.max()),
        })

    return {
        "e4_directory": str(e4_dir).replace("\\", "/"),
        "checks": checks,
        "all_passed": all(check["passed"] for check in checks),
        "split_statistics": split_stats,
        "total_pairs": len(combined),
    }


def markdown_report(processed: list[dict], e4: dict | None) -> str:
    lines = [
        "# Parquet dataset assessment",
        "",
        "## Processed datasets",
        "",
        "| Dataset | Files | Rows | Size | Key columns | Duplicate key rows |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for dataset in processed:
        keys = ", ".join(dataset["key_columns"]) or "-"
        lines.append(
            f"| {dataset['dataset']} | {dataset['file_count']} | {dataset['rows']:,} | "
            f"{dataset['human_size']} | {keys} | {dataset['key_duplicate_rows']:,} |"
        )

    if e4:
        lines.extend(["", "## E4 integrity checks", "",
                      "| Check | Result | Detail |", "|---|---|---|"])
        for check in e4["checks"]:
            result = "PASS" if check["passed"] else "FAIL"
            lines.append(f"| {check['name']} | {result} | {check['detail']} |")
        lines.extend(["", "## E4 splits", "",
                      "| Split | Pairs | Users | Items | Relevant pairs (rating >= 4) |",
                      "|---|---:|---:|---:|---:|"])
        for split in e4["split_statistics"]:
            lines.append(
                f"| {split['split']} | {split['pairs']:,} | {split['users']:,} | "
                f"{split['items']:,} | {split['relevant_pairs']:,} |"
            )
    return "\n".join(lines) + "\n"


def write_summary_csv(processed: list[dict], out_dir: Path) -> None:
    rows = [{
        "dataset": item["dataset"],
        "file_count": item["file_count"],
        "rows": item["rows"],
        "bytes": item["bytes"],
        "key_columns": "|".join(item["key_columns"]),
        "key_duplicate_rows": item["key_duplicate_rows"],
    } for item in processed]
    pd.DataFrame(rows).to_csv(out_dir / "parquet_dataset_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--e4-dir", type=Path, default=Path("data/processed/e4"))
    parser.add_argument("--out", type=Path, default=Path("outputs/reports/Electronics/parquet_assessment"))
    args = parser.parse_args()

    if not args.data_root.exists():
        parser.error(f"Data directory not found: {args.data_root}")

    processed = profile_tree(args.data_root)
    e4 = validate_e4(args.e4_dir) if args.e4_dir.is_dir() else None
    args.out.mkdir(parents=True, exist_ok=True)

    write_summary_csv(processed, args.out)
    (args.out / "parquet_dataset_profile.json").write_text(
        json.dumps({"datasets": processed}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.out / "parquet_assessment.md").write_text(
        markdown_report(processed, e4), encoding="utf-8"
    )
    if e4:
        (args.out / "e4_integrity.json").write_text(
            json.dumps(e4, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"Analyzed {len(processed)} processed dataset directories")
    print(f"Saved reports to {args.out}")
    if e4:
        print("E4 integrity:", "PASS" if e4["all_passed"] else "FAIL")


if __name__ == "__main__":
    main()
