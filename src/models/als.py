"""Spark MLlib ALS training, persistence, and E4 mapping checks."""

import math
from dataclasses import dataclass
from pathlib import Path

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.models.mappings import validate_spark_id_maps


@dataclass(frozen=True)
class ALSConfig:
    rank: int = 32
    max_iter: int = 10
    reg_param: float = 0.1
    seed: int = 42
    implicit_prefs: bool = False
    alpha: float = 1.0

    def __post_init__(self):
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("ALS rank must be positive")
        if (isinstance(self.max_iter, bool) or not isinstance(self.max_iter, int)
                or self.max_iter < 1):
            raise ValueError("ALS max_iter must be positive")
        if not math.isfinite(self.reg_param) or self.reg_param < 0:
            raise ValueError("ALS reg_param must be non-negative")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("ALS alpha must be positive")
        if not isinstance(self.implicit_prefs, bool):
            raise ValueError("ALS implicit_prefs must be a boolean")

    @classmethod
    def from_dict(cls, values):
        implicit_prefs = values.get("implicit_prefs", False)
        if not isinstance(implicit_prefs, bool):
            raise ValueError("ALS implicit_prefs must be a boolean")
        return cls(
            rank=int(values.get("rank", 32)),
            max_iter=int(values.get("max_iter", 10)),
            reg_param=float(values.get("reg_param", 0.1)),
            seed=int(values.get("seed", 42)),
            implicit_prefs=implicit_prefs,
            alpha=float(values.get("alpha", 1.0)),
        )


def map_history_to_indices(history, user_map, item_map):
    """Validate the E4 maps and join ratings to their stable integer indices."""
    required = {"user_id", "product_id", "rating"}
    if not required.issubset(history.columns):
        raise ValueError(f"History must contain {sorted(required)}")
    validate_spark_id_maps(history, user_map, item_map)
    mapped = (history.join(user_map.select("user_id", "user_idx"), "user_id")
              .join(item_map.select("product_id", "item_idx"), "product_id")
              .select(
                  F.col("user_idx").cast("int"),
                  F.col("item_idx").cast("int"),
                  F.col("rating").cast("float").alias("rating"),
              ))
    return mapped


def train_als(spark, history, user_map, item_map, config=None, model_path=None):
    """Fit an explicit/implicit ALS model on rows mapped with E4 IDs."""
    settings = config or ALSConfig()
    if not isinstance(settings, ALSConfig):
        settings = ALSConfig.from_dict(settings)
    indexed = map_history_to_indices(history, user_map, item_map)
    estimator = ALS(
        userCol="user_idx",
        itemCol="item_idx",
        ratingCol="rating",
        rank=settings.rank,
        maxIter=settings.max_iter,
        regParam=settings.reg_param,
        seed=settings.seed,
        implicitPrefs=settings.implicit_prefs,
        alpha=settings.alpha,
        coldStartStrategy="drop",
    )
    model = estimator.fit(indexed)
    if model_path is not None:
        model.write().overwrite().save(str(Path(model_path)))
    return model


def load_als(model_path):
    """Reload a previously persisted Spark ALSModel."""
    return ALSModel.load(str(Path(model_path)))
