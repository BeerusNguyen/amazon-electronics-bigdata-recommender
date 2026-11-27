"""Read-only audit of E4 data used as input to E5."""

import argparse
import json
import logging
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import pyspark

from src.splitting.prepare_e4 import BOUNDARIES


UTC_TRAIN_END = BOUNDARIES[0]
UTC_TEST_START = BOUNDARIES[1]
UTC_TEST_END_EXCLUSIVE = int(
    datetime.fromisoformat("2018-10-06").replace(tzinfo=timezone.utc).timestamp()
)


def read_parquet_dataset(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")
    if path.is_dir():
        parts = sorted(path.glob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No .parquet part files in: {path}")
        return pd.read_parquet(path)
    return pd.read_parquet(path)


def blank_count(frame, column):
    values = frame[column]
    if values.dtype == "object":
        return int(values.isna().sum() + values.map(
            lambda value: isinstance(value, str) and not value.strip()
        ).sum())
    return int(values.isna().sum())


def split_quality(name, frame):
    required = {"user_id", "product_id", "rating", "timestamp"}
    missing_columns = sorted(required - set(frame.columns))
    quality = {
        "status": "PASS",
        "rows": len(frame),
        "distinct_users": int(frame["user_id"].nunique()) if "user_id" in frame else 0,
        "distinct_products": int(frame["product_id"].nunique()) if "product_id" in frame else 0,
        "schema": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_columns": missing_columns,
    }
    if missing_columns:
        quality["status"] = "FAIL"
        return quality

    quality["null_blank"] = {
        column: blank_count(frame, column)
        for column in ("user_id", "product_id", "rating", "timestamp")
    }

    rating = pd.to_numeric(frame["rating"], errors="coerce")
    timestamp = pd.to_numeric(frame["timestamp"], errors="coerce")
    rating_finite = np.isfinite(rating.fillna(np.nan).to_numpy(dtype=float))
    quality["rating"] = {
        "invalid_numeric": int(rating.isna().sum()),
        "non_finite": int((~rating_finite).sum()),
        "non_integer": int((rating.dropna() != rating.dropna().round()).sum()),
        "outside_1_5": int((~rating.isin([1, 2, 3, 4, 5])).sum()),
        "distribution": {
            str(int(key)): int(value)
            for key, value in rating[rating.isin([1, 2, 3, 4, 5])].value_counts().sort_index().items()
        },
    }
    quality["timestamp"] = {
        "invalid_numeric": int(timestamp.isna().sum()),
        "non_finite": int((~np.isfinite(timestamp.fillna(np.nan).to_numpy(dtype=float))).sum()),
        "min": int(timestamp.min()) if timestamp.notna().any() else None,
        "max": int(timestamp.max()) if timestamp.notna().any() else None,
    }

    quality["duplicate_user_product_pairs"] = int(
        frame.duplicated(["user_id", "product_id"]).sum()
    )
    quality["positive_interactions"] = int(rating.ge(4).sum())
    quality["users_with_positive"] = int(
        frame.loc[rating.ge(4), "user_id"].nunique()
    )

    if name == "train":
        in_range = timestamp.lt(UTC_TRAIN_END)
    elif name == "validation":
        in_range = timestamp.ge(UTC_TRAIN_END) & timestamp.lt(UTC_TEST_START)
    else:
        in_range = timestamp.ge(UTC_TEST_START) & timestamp.lt(UTC_TEST_END_EXCLUSIVE)
    quality["timestamp"]["rows_out_of_protocol_range"] = int((~in_range).sum())

    bad = (
        any(value > 0 for value in quality["null_blank"].values())
        or quality["rating"]["invalid_numeric"] > 0
        or quality["rating"]["non_finite"] > 0
        or quality["rating"]["non_integer"] > 0
        or quality["rating"]["outside_1_5"] > 0
        or quality["timestamp"]["invalid_numeric"] > 0
        or quality["timestamp"]["non_finite"] > 0
        or quality["timestamp"]["rows_out_of_protocol_range"] > 0
        or quality["duplicate_user_product_pairs"] > 0
    )
    if bad:
        quality["status"] = "FAIL"
    return quality


def mapping_quality(mapping, history, id_column, index_column):
    result = {"status": "PASS", "rows": len(mapping), "columns": list(mapping.columns)}
    missing = {id_column, index_column} - set(mapping.columns)
    if missing:
        result["status"] = "FAIL"
        result["missing_columns"] = sorted(missing)
        return result

    ids = mapping[id_column]
    indices = mapping[index_column]
    result["schema"] = {
        id_column: str(mapping[id_column].dtype),
        index_column: str(mapping[index_column].dtype),
    }
    result["null_blank_ids"] = blank_count(mapping, id_column)
    result["null_indices"] = int(indices.isna().sum())
    result["duplicate_ids"] = int(ids.duplicated().sum())
    result["duplicate_indices"] = int(indices.duplicated().sum())
    numeric = pd.to_numeric(indices, errors="coerce")
    result["negative_indices"] = int(numeric.lt(0).sum())
    result["non_integer_indices"] = int((numeric.dropna() != numeric.dropna().round()).sum())
    result["index_dtype_int32"] = str(indices.dtype) in ("int32", "Int32")

    ordered = indices.dropna().to_numpy()
    result["index_min"] = int(ordered.min()) if len(ordered) else None
    result["index_max"] = int(ordered.max()) if len(ordered) else None
    result["contiguous_from_zero"] = (
        len(ordered) > 0
        and int(ordered.min()) == 0
        and int(ordered.max()) == len(set(ordered.tolist())) - 1
        and len(set(ordered.tolist())) == len(ordered)
    )

    result["reverse_mapping_unique"] = (
        not ids.duplicated().any() and not indices.duplicated().any()
    )
    history_ids = set(history[id_column].dropna())
    mapping_ids = set(ids.dropna())
    result["history_distinct_ids"] = len(history_ids)
    result["missing_history_ids"] = len(history_ids - mapping_ids)
    result["unexpected_mapping_ids"] = len(mapping_ids - history_ids)
    result["exact_history_coverage"] = (
        history_ids == mapping_ids
        and result["missing_history_ids"] == 0
        and result["unexpected_mapping_ids"] == 0
    )

    bad = (
        result["null_blank_ids"] > 0
        or result["null_indices"] > 0
        or result["duplicate_ids"] > 0
        or result["duplicate_indices"] > 0
        or result["negative_indices"] > 0
        or result["non_integer_indices"] > 0
        or not result["index_dtype_int32"]
        or not result["contiguous_from_zero"]
        or not result["reverse_mapping_unique"]
        or not result["exact_history_coverage"]
    )
    if bad:
        result["status"] = "FAIL"
    return result


def joined_mapping_quality(history, user_map, item_map):
    mapped = history.merge(
        user_map[["user_id", "user_idx"]], on="user_id", how="left", validate="many_to_one"
    ).merge(
        item_map[["product_id", "item_idx"]], on="product_id", how="left", validate="many_to_one"
    )
    return {
        "status": "PASS" if (
            len(mapped) == len(history)
            and not mapped["user_idx"].isna().any()
            and not mapped["item_idx"].isna().any()
        ) else "FAIL",
        "history_rows": len(history),
        "joined_rows": len(mapped),
        "rows_lost": len(history) - len(mapped),
        "rows_added": len(mapped) - len(history),
        "null_user_idx": int(mapped["user_idx"].isna().sum()),
        "null_item_idx": int(mapped["item_idx"].isna().sum()),
        "duplicate_joined_pairs": int(
            mapped.duplicated(["user_id", "product_id"]).sum()
        ),
        "rating_values_preserved": bool(
            pd.Series(history["rating"].sort_values(ignore_index=True)).equals(
                pd.Series(mapped["rating"].sort_values(ignore_index=True))
            )
        ),
    }


def cold_start_quality(history, held_out):
    history_users = set(history["user_id"])
    history_items = set(history["product_id"])
    support = history.groupby("user_id").size()
    tagged = held_out.assign(
        warm_user=held_out["user_id"].isin(history_users),
        warm_item=held_out["product_id"].isin(history_items),
        support=held_out["user_id"].map(support).fillna(0).astype(int),
    )
    tagged["cohort"] = np.select(
        [
            tagged["warm_user"] & tagged["warm_item"],
            ~tagged["warm_user"] & tagged["warm_item"],
            tagged["warm_user"] & ~tagged["warm_item"],
        ],
        ["warm_user/warm_item", "cold_user/warm_item", "warm_user/cold_item"],
        default="cold_user/cold_item",
    )
    tagged["support_group"] = np.select(
        [tagged["support"].eq(0), tagged["support"].lt(5)],
        ["0", "1-4"],
        default="5+",
    )
    cohorts = tagged["cohort"].value_counts().to_dict()
    support_groups = tagged["support_group"].value_counts().to_dict()
    unknown_users = int((~tagged["warm_user"]).sum())
    unknown_items = int((~tagged["warm_item"]).sum())
    seen_pairs = set(history[["user_id", "product_id"]].itertuples(index=False, name=None))
    held_pairs = set(held_out[["user_id", "product_id"]].itertuples(index=False, name=None))
    return {
        "status": "PASS",
        "cohorts": cohorts,
        "support_groups": support_groups,
        "unknown_user_pairs": unknown_users,
        "unknown_item_pairs": unknown_items,
        "heldout_pairs_already_in_history": len(seen_pairs & held_pairs),
        "seen_history_interactions_include_low_ratings": int(
            (history["rating"] < 4).sum()
        ),
    }


def distribution_quality(history):
    user_counts = history.groupby("user_id").size()
    item_counts = history.groupby("product_id").size()
    users = int(user_counts.size)
    items = int(item_counts.size)
    pairs = len(history)
    return {
        "history_rows": pairs,
        "distinct_users": users,
        "distinct_items": items,
        "interactions_per_user": {
            "min": int(user_counts.min()),
            "median": float(user_counts.median()),
            "mean": float(user_counts.mean()),
            "max": int(user_counts.max()),
            "users_with_one": int(user_counts.eq(1).sum()),
        },
        "interactions_per_item": {
            "min": int(item_counts.min()),
            "median": float(item_counts.median()),
            "mean": float(item_counts.mean()),
            "max": int(item_counts.max()),
            "items_with_one": int(item_counts.eq(1).sum()),
        },
        "density": pairs / (users * items) if users and items else None,
        "positive_interactions": int(history["rating"].ge(4).sum()),
        "zero_positive_catalog_items": int(
            history.assign(positive=history["rating"].ge(4))
            .groupby("product_id")["positive"].sum().eq(0).sum()
        ),
    }


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def compare_cold_start_csv(path, phases):
    if not Path(path).exists():
        return {"status": "NOT CHECKED", "reason": "File does not exist"}
    frame = pd.read_csv(path)
    result = {"status": "PASS", "rows": len(frame), "mismatches": []}
    required = {"phase", "group", "pairs"}
    if not required.issubset(frame.columns):
        result["status"] = "FAIL"
        result["missing_columns"] = sorted(required - set(frame.columns))
        return result
    cohort_names = (
        "warm_user/warm_item", "cold_user/warm_item",
        "warm_user/cold_item", "cold_user/cold_item",
    )
    support_names = ("0", "1-4", "5+")
    for phase in ("validation", "test"):
        cold = phases[phase]["cold_start"]
        for group_name in cohort_names:
            rows = frame[(frame["phase"] == phase) & (frame["group"] == group_name)]
            reported = int(rows["pairs"].sum()) if not rows.empty else None
            observed = int(cold["cohorts"].get(group_name, 0))
            if reported != observed:
                result["mismatches"].append({
                    "phase": phase, "group": group_name,
                    "observed": observed, "reported": reported,
                })
        for group_name in (f"history_{name}" for name in support_names):
            rows = frame[(frame["phase"] == phase) & (frame["group"] == group_name)]
            reported = int(rows["pairs"].sum()) if not rows.empty else None
            plain_name = group_name.removeprefix("history_")
            observed = int(cold["support_groups"].get(plain_name, 0))
            if reported != observed:
                result["mismatches"].append({
                    "phase": phase, "group": group_name,
                    "observed": observed, "reported": reported,
                })
    if result["mismatches"]:
        result["status"] = "FAIL"
    return result


def compare_snapshot_manifest(path, phases):
    if not Path(path).exists():
        return {"status": "NOT CHECKED", "reason": "File does not exist"}
    manifest = read_json(path)
    mismatches = []
    fields = (
        "history_pairs", "history_users", "history_items",
        "held_out_pairs", "held_out_users", "held_out_items",
    )
    for phase in ("validation", "test"):
        reported = manifest.get(phase, {})
        observed = {
            "history_pairs": phases[phase]["history_rows"],
            "history_users": phases[phase]["history_users"],
            "history_items": phases[phase]["history_items"],
            "held_out_pairs": phases[phase]["heldout_rows"],
            "held_out_users": phases[phase]["heldout_users"],
            "held_out_items": phases[phase]["heldout_items"],
        }
        for field in fields:
            if reported.get(field) != observed[field]:
                mismatches.append({
                    "phase": phase, "field": field,
                    "observed": observed[field],
                    "reported": reported.get(field),
                })
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "mismatches": mismatches,
    }


def markdown_report(summary):
    lines = [
        "# E5 input data audit",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Overall status: **{summary['overall_status']}**",
        f"- Data directory: `{summary['data_dir']}`",
        f"- Python: `{summary['environment']['python']}`",
        f"- PySpark: `{summary['environment']['pyspark']}`",
        f"- Pandas: `{summary['environment']['pandas']}`",
        f"- PyArrow: `{summary['environment']['pyarrow']}`",
        "",
        "## Split integrity",
        "",
        "| Split | Rows | Users | Items | Positive | Duplicate pairs | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in ("train", "validation", "test"):
        item = summary["splits"][name]
        lines.append(
            f"| {name} | {item['rows']} | {item['distinct_users']} | "
            f"{item['distinct_products']} | {item['positive_interactions']} | "
            f"{item['duplicate_user_product_pairs']} | {item['status']} |"
        )
    lines.extend([
        "",
        "### Schema and quality",
        "",
        "| Split | Rating | Timestamp | Null/blank | Out-of-range | Duplicate pairs | Status |",
        "|---|---|---|---|---:|---:|---|",
    ])
    for name in ("train", "validation", "test"):
        item = summary["splits"][name]
        null_count = sum(item["null_blank"].values())
        lines.append(
            f"| {name} | {item['schema']['rating']} | {item['schema']['timestamp']} | "
            f"{null_count} | {item['timestamp']['rows_out_of_protocol_range']} | "
            f"{item['duplicate_user_product_pairs']} | {item['status']} |"
        )
    split_stats_items = summary["manifest_comparison"].get("split_statistics_rows", {}).values()
    split_stats_status = "PASS" if all(
        item["observed"] == item["reported"] for item in split_stats_items
    ) else "FAIL"
    clean_pairs = summary["manifest_comparison"].get("split_manifest_clean_pairs", {})
    split_manifest_status = (
        "PASS"
        if clean_pairs.get("observed") == clean_pairs.get("reported")
        else "FAIL"
    )
    lines.extend([
        "",
        "### Manifest reconciliation",
        "",
        "- `split_statistics.csv`: "
        f"**{split_stats_status}**",
        "- `split_manifest.json`: "
        f"**{split_manifest_status}**",
        f"- `snapshot_manifest.json`: **{summary['snapshot_manifest_comparison']['status']}**",
        f"- `cold_start.csv`: **{summary['cold_start_csv']['status']}**",
        "",
        "## Phase history and mappings",
        "",
    ])
    for name in ("validation", "test"):
        phase = summary["phases"][name]
        lines.append(f"### {name}")
        lines.append(f"- History rows: {phase['history_rows']}")
        lines.append(f"- History users: {phase['history_users']}")
        lines.append(f"- History items: {phase['history_items']}")
        lines.append(f"- User mapping: {phase['user_mapping']['status']}")
        lines.append(f"- Item mapping: {phase['item_mapping']['status']}")
        lines.append(f"- Joined mapping: {phase['joined_mapping']['status']}")
        lines.append(f"- Joined rows lost/added: "
                     f"{phase['joined_mapping']['rows_lost']}/"
                     f"{phase['joined_mapping']['rows_added']}")
        lines.append(f"- Rating values preserved: "
                     f"{phase['joined_mapping']['rating_values_preserved']}")
        lines.append("")
    lines.extend([
        "## Cold-start",
        "",
        "| Phase | warm/warm | cold/warm | warm/cold | cold/cold | 0 | 1-4 | 5+ | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for name in ("validation", "test"):
        cold = summary["phases"][name]["cold_start"]
        cohorts = cold["cohorts"]
        support = cold["support_groups"]
        lines.append(
            f"| {name} | {cohorts.get('warm_user/warm_item', 0)} | "
            f"{cohorts.get('cold_user/warm_item', 0)} | "
            f"{cohorts.get('warm_user/cold_item', 0)} | "
            f"{cohorts.get('cold_user/cold_item', 0)} | "
            f"{support.get('0', 0)} | {support.get('1-4', 0)} | "
            f"{support.get('5+', 0)} | {cold['status']} |"
        )
    lines.extend([
        "",
        "## ALS input distribution",
        "",
        "| Phase | Rows | Users | Items | Density | Users with 1 | Items with 1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name in ("validation", "test"):
        dist = summary["phases"][name]["als_distribution"]
        lines.append(
            f"| {name} | {dist['history_rows']} | {dist['distinct_users']} | "
            f"{dist['distinct_items']} | {dist['density']:.3e} | "
            f"{dist['interactions_per_user']['users_with_one']} | "
            f"{dist['interactions_per_item']['items_with_one']} |"
        )
    lines.extend([
        "",
        "## Readiness",
        "",
        f"- Popularity: **{summary['readiness']['popularity']}**",
        f"- ALS: **{summary['readiness']['als']}**",
        f"- Cold-start fallback: **{summary['readiness']['fallback']}**",
        "",
        "## Required handling before training",
        "",
    ])
    for issue in summary["issues"]:
        lines.append(f"- {issue}")
    if not summary["issues"]:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def print_final_conclusion(summary):
    print("\n" + "=" * 72)
    print("E5 INPUT DATA AUDIT - FINAL CONCLUSION")
    print("=" * 72)
    print(f"Overall status: {summary['overall_status']}")
    for name in ("train", "validation", "test"):
        split = summary["splits"][name]
        print(
            f"{name.capitalize():10s} "
            f"{split['rows']:>7,d} rows | "
            f"{split['distinct_users']:>7,d} users | "
            f"{split['distinct_products']:>7,d} products | "
            f"{split['positive_interactions']:>7,d} positive | "
            f"{split['status']}"
        )
    print(
        "Phase mapping: "
        f"Validation={summary['phases']['validation']['joined_mapping']['status']}, "
        f"Test={summary['phases']['test']['joined_mapping']['status']}"
    )
    print(
        "Readiness: "
        f"Popularity={summary['readiness']['popularity']}, "
        f"ALS={summary['readiness']['als']}, "
        f"Fallback={summary['readiness']['fallback']}"
    )
    if summary["issues"]:
        print("Required handling before training:")
        for issue in summary["issues"]:
            print(f"- {issue}")
    else:
        print("Required handling before training: None")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/e4"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or Path("outputs/reports/Electronics/e5_input_audit") / run_id
    out.mkdir(parents=True, exist_ok=False)

    splits = {}
    frames = {}
    for name in ("train", "validation", "test"):
        path = args.data_dir / f"{name}.parquet"
        logging.info("Reading split %s from %s", name, path)
        frame = read_parquet_dataset(path)
        frames[name] = frame
        splits[name] = split_quality(name, frame)

    cross_duplicate_pairs = int(
        pd.concat(
            [frames[name][["user_id", "product_id"]] for name in ("train", "validation", "test")],
            ignore_index=True,
        ).duplicated(["user_id", "product_id"]).sum()
    )

    statistics_path = args.data_dir / "split_statistics.csv"
    manifest_path = args.data_dir / "split_manifest.json"
    snapshot_path = args.data_dir / "audit" / "snapshot_manifest.json"
    cold_start_path = args.data_dir / "audit" / "cold_start.csv"

    statistics = pd.read_csv(statistics_path) if statistics_path.exists() else None
    split_manifest = read_json(manifest_path) if manifest_path.exists() else None
    snapshot_manifest = read_json(snapshot_path) if snapshot_path.exists() else None

    manifest_mismatches = {}
    if statistics is not None:
        observed = {
            row["split"]: int(row["pairs"])
            for row in splits.values()
        } if False else {
            name: splits[name]["rows"] for name in ("train", "validation", "test")
        }
        reported = statistics.set_index("split")["pairs"].to_dict()
        manifest_mismatches["split_statistics_rows"] = {
            name: {"observed": observed[name], "reported": int(reported.get(name, -1))}
            for name in ("train", "validation", "test")
        }

    if split_manifest is not None:
        audit = split_manifest.get("audit", {})
        manifest_mismatches["split_manifest_clean_pairs"] = {
            "observed": sum(splits[name]["rows"] for name in ("train", "validation", "test")),
            "reported": int(audit.get("clean_pairs", -1)),
        }

    phases = {}
    for phase in ("validation", "test"):
        history = frames["train"].copy()
        held_out = frames[phase].copy()
        if phase == "test":
            history = pd.concat([history, frames["validation"]], ignore_index=True)

        logging.info("Checking %s history and mappings", phase)
        map_dir = args.data_dir / "audit" / phase
        user_map = read_parquet_dataset(map_dir / "user_id_map.parquet")
        item_map = read_parquet_dataset(map_dir / "item_id_map.parquet")
        user_mapping = mapping_quality(
            user_map, history, "user_id", "user_idx"
        )
        item_mapping = mapping_quality(
            item_map, history, "product_id", "item_idx"
        )
        joined_mapping = joined_mapping_quality(history, user_map, item_map)
        cold_start = cold_start_quality(history, held_out)
        distribution = distribution_quality(history)
        phases[phase] = {
            "history_rows": len(history),
            "history_users": int(history["user_id"].nunique()),
            "history_items": int(history["product_id"].nunique()),
            "heldout_rows": len(held_out),
            "heldout_users": int(held_out["user_id"].nunique()),
            "heldout_items": int(held_out["product_id"].nunique()),
            "user_mapping": user_mapping,
            "item_mapping": item_mapping,
            "joined_mapping": joined_mapping,
            "cold_start": cold_start,
            "als_distribution": distribution,
        }

    issues = []
    for name, split in splits.items():
        if split["status"] != "PASS":
            issues.append(f"{name} split failed quality checks")
    for mismatch_name, mismatch in manifest_mismatches.items():
        if mismatch_name == "split_manifest_clean_pairs":
            if mismatch["observed"] != mismatch["reported"]:
                issues.append("split_manifest clean_pairs mismatch")
        else:
            if any(item["observed"] != item["reported"] for item in mismatch.values()):
                issues.append("split_statistics row-count mismatch")
    for phase, phase_result in phases.items():
        if phase_result["user_mapping"]["status"] != "PASS":
            issues.append(f"{phase} user mapping failed")
        if phase_result["item_mapping"]["status"] != "PASS":
            issues.append(f"{phase} item mapping failed")
        if phase_result["joined_mapping"]["status"] != "PASS":
            issues.append(f"{phase} joined mapping failed")
    if cross_duplicate_pairs:
        issues.append("user-product pairs occur in more than one split")

    snapshot_comparison = compare_snapshot_manifest(snapshot_path, phases)
    cold_start_comparison = compare_cold_start_csv(cold_start_path, phases)
    if snapshot_comparison["status"] == "FAIL":
        issues.append("snapshot_manifest.json differs from recomputed snapshot counts")
    if cold_start_comparison["status"] == "FAIL":
        issues.append("cold_start.csv differs from recomputed cold-start groups")

    mapping_ok = all(
        phases[name][key]["status"] == "PASS"
        for name in ("validation", "test")
        for key in ("user_mapping", "item_mapping", "joined_mapping")
    )
    split_ok = all(splits[name]["status"] == "PASS" for name in ("train", "validation", "test"))
    no_manifest_mismatch = not any(
        (
            item["observed"] != item["reported"]
            for item in mismatch.values()
        )
        for mismatch in manifest_mismatches.values()
        if isinstance(mismatch, dict) and all(isinstance(value, dict) for value in mismatch.values())
    )
    readiness = {
        "popularity": "PASS" if split_ok and mapping_ok else "FAIL",
        "als": "PASS" if split_ok and mapping_ok else "FAIL",
        "fallback": "PASS" if split_ok and mapping_ok else "FAIL",
    }
    overall_status = "PASS" if not issues else "FAIL"

    summary = {
        "run_id": run_id,
        "data_dir": str(args.data_dir),
        "overall_status": overall_status,
        "environment": {
            "python": sys.version.split()[0],
            "pyspark": pyspark.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "platform": platform.platform(),
        },
        "splits": splits,
        "cross_split_duplicate_user_product_pairs": cross_duplicate_pairs,
        "manifest_comparison": manifest_mismatches,
        "snapshot_manifest": snapshot_manifest,
        "snapshot_manifest_comparison": snapshot_comparison,
        "cold_start_csv": cold_start_comparison,
        "phases": phases,
        "readiness": readiness,
        "issues": issues,
    }

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (out / "summary.md").write_text(markdown_report(summary), encoding="utf-8")
    logging.info("Audit status: %s", overall_status)
    logging.info("Report: %s", out)
    print_final_conclusion(summary)
    print(json.dumps({
        "run_id": run_id,
        "overall_status": overall_status,
        "report_dir": str(out),
    }, indent=2))

    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
