"""Uncertainty and paired significance tests for per-user ranking metrics.

Why: a macro Recall@10 is a mean over tens of thousands of users with very skewed values
(most users score 0). A difference between two methods is only claimed when the paired
test across the same users rejects "no difference" after Holm correction.

bootstrap_ci        percentile CI of the mean by resampling users with replacement
paired_bootstrap    CI of mean(A - B) on the same users
sign_flip_test      exact-in-the-limit randomisation test of mean(A - B) = 0 (paired)
holm                Holm-Bonferroni adjusted p-values for a family of comparisons
"""

import numpy as np


def bootstrap_ci(values, n_resamples=2000, alpha=0.05, seed=42, chunk=200):
    """Mean and percentile bootstrap CI."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, n_resamples, chunk):
        size = min(chunk, n_resamples - start)
        idx = rng.integers(0, values.size, (size, values.size))
        means.append(values[idx].mean(axis=1))
    means = np.concatenate(means)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(low), float(high)


def paired_bootstrap(a, b, n_resamples=2000, alpha=0.05, seed=42):
    """Mean difference A - B with a paired bootstrap CI."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired arrays must have the same length")
    return bootstrap_ci(a - b, n_resamples, alpha, seed)


def sign_flip_test(a, b, n_permutations=10000, seed=42, chunk=500):
    """Two-sided paired randomisation test on mean difference."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[diff != 0]
    if diff.size == 0:
        return 1.0
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, n_permutations, chunk):
        size = min(chunk, n_permutations - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, diff.size))
        extreme += int((np.abs((signs * diff).mean(axis=1)) >= observed - 1e-15).sum())
    return float((extreme + 1) / (n_permutations + 1))


def holm(p_values):
    """Holm-Bonferroni step-down adjustment, returned in input order."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = p_values.size
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted
