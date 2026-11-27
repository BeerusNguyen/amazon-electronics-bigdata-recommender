"""Validation helpers for the reversible E4 integer ID maps."""

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType


def validate_id_mapping(mapping, source_ids, id_column, index_column):
    """Validate an E4 mapping and verify exact coverage of a small ID collection."""
    missing = {id_column, index_column} - set(mapping.columns)
    if missing:
        raise ValueError(f"{id_column} mapping is missing columns: {sorted(missing)}")
    values = mapping[id_column]
    indices = mapping[index_column]
    if values.isna().any() or not values.map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError(f"{id_column} values must be non-empty strings")
    if indices.isna().any() or not pd.api.types.is_integer_dtype(indices.dtype):
        raise ValueError(f"{index_column} values must be non-null integers")
    if values.duplicated().any():
        raise ValueError(f"{id_column} values must be unique")
    if indices.duplicated().any():
        raise ValueError(f"{index_column} values must be unique")
    if len(indices) == 0:
        raise ValueError(f"{index_column} mapping must not be empty")
    if indices.min() < 0 or indices.max() > np.iinfo(np.int32).max:
        raise ValueError(f"{index_column} values must fit non-negative int32")
    if indices.min() != 0 or indices.max() != len(indices) - 1:
        raise ValueError(f"{index_column} values must be contiguous from zero")

    expected = set(source_ids)
    mapped = set(values)
    if expected != mapped:
        missing_ids = len(expected - mapped)
        unexpected_ids = len(mapped - expected)
        raise ValueError(
            f"{id_column} map does not match history "
            f"({missing_ids} missing, {unexpected_ids} unexpected)"
        )
    return mapping


def validate_spark_id_maps(history, user_map, item_map):
    """Validate E4 map schemas, contiguous indices, uniqueness, and history coverage."""
    _validate_spark_mapping(history, user_map, "user_id", "user_idx")
    _validate_spark_mapping(history, item_map, "product_id", "item_idx")


def _validate_spark_mapping(history, mapping, id_column, index_column):
    missing = {id_column, index_column} - set(mapping.columns)
    if missing:
        raise ValueError(f"{id_column} mapping is missing columns: {sorted(missing)}")
    if not isinstance(mapping.schema[id_column].dataType, StringType):
        raise ValueError(f"{id_column} mapping IDs must use Spark string type")
    if not isinstance(mapping.schema[index_column].dataType, IntegerType):
        raise ValueError(f"{index_column} mapping indices must use Spark int32 type")

    stats = mapping.agg(
        F.count("*").alias("rows"),
        F.count(F.when(F.col(id_column).isNull(), 1)).alias("null_ids"),
        F.count(F.when(F.col(index_column).isNull(), 1)).alias("null_indices"),
        F.countDistinct(id_column).alias("distinct_ids"),
        F.countDistinct(index_column).alias("distinct_indices"),
        F.min(index_column).alias("min_index"),
        F.max(index_column).alias("max_index"),
        F.count(F.when(F.length(F.trim(id_column)) == 0, 1)).alias("blank_ids"),
    ).first()
    if stats.rows == 0:
        raise ValueError(f"{id_column} mapping must not be empty")
    if stats.null_ids or stats.null_indices or stats.blank_ids:
        raise ValueError(f"{id_column} mapping contains null or blank values")
    if stats.rows != stats.distinct_ids or stats.rows != stats.distinct_indices:
        raise ValueError(f"{id_column} mapping IDs and indices must be unique")
    if stats.min_index != 0 or stats.max_index != stats.rows - 1:
        raise ValueError(f"{index_column} values must be contiguous from zero")

    if id_column not in history.columns:
        raise ValueError(f"History is missing {id_column}")
    source = history.select(id_column).distinct()
    if source.where(
        F.col(id_column).isNull() | (F.length(F.trim(id_column)) == 0)
    ).limit(1).count():
        raise ValueError(f"History contains null or blank {id_column} values")
    mapping_ids = mapping.select(id_column).distinct()
    missing = source.join(mapping_ids, id_column, "left_anti").limit(1).count()
    unexpected = mapping_ids.join(source, id_column, "left_anti").limit(1).count()
    source_count = source.count()
    if missing or unexpected or source_count != stats.rows:
        raise ValueError(
            f"{id_column} map must exactly match history "
            f"({source_count} history IDs, {stats.rows} mapped IDs)"
        )
