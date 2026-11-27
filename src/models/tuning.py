"""Pure hyperparameter-grid expansion and validation-score selection."""

from itertools import product


def expand_parameter_grid(grid):
    """Expand a mapping of parameter names to non-empty candidate lists."""
    if not grid:
        raise ValueError("ALS parameter grid must not be empty")
    allowed = {"rank", "max_iter", "reg_param", "seed", "implicit_prefs", "alpha"}
    unknown = set(grid) - allowed
    if unknown:
        raise ValueError(f"Unknown ALS grid parameters: {sorted(unknown)}")
    keys = sorted(grid)
    values = [grid[key] for key in keys]
    if any(not isinstance(options, list) or not options for options in values):
        raise ValueError("Every ALS grid entry must be a non-empty list")
    return [dict(zip(keys, combination)) for combination in product(*values)]


def select_best_validation_result(results):
    """Select by NDCG@K, then Recall@K, with stable input-order tie breaking."""
    if not results:
        raise ValueError("At least one validation result is required")
    for result in results:
        if "ndcg_at_k" not in result or "recall_at_k" not in result:
            raise ValueError("Validation results require NDCG@K and Recall@K")
    return max(
        enumerate(results),
        key=lambda pair: (
            float(pair[1]["ndcg_at_k"]),
            float(pair[1]["recall_at_k"]),
            -pair[0],
        ),
    )[1]
