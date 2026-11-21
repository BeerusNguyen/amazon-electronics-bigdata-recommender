"""UTC boundaries shared by all E4 experiments."""

from pyspark.sql import functions as F


def compute_cutoffs(df, train_fraction, validation_fraction):
    """Estimate two boundaries for exploratory or toy data."""
    if not 0 < train_fraction < train_fraction + validation_fraction < 1:
        raise ValueError("Train and validation fractions must be positive and sum to less than one")
    cutoffs = df.approxQuantile("timestamp", [train_fraction, train_fraction + validation_fraction], 0)
    if len(cutoffs) != 2 or cutoffs[0] >= cutoffs[1]:
        raise ValueError("Cannot obtain two distinct temporal boundaries")
    return tuple(int(value) for value in cutoffs)


def split_by_time(df, validation_start, test_start):
    if validation_start >= test_start:
        raise ValueError("Validation must start before test")
    timestamp = F.col("timestamp")
    return (
        df.where(timestamp < validation_start),
        df.where((timestamp >= validation_start) & (timestamp < test_start)),
        df.where(timestamp >= test_start),
    )
