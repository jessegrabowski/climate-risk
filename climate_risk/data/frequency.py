from typing import Literal

import polars as pl

# How coarsely a monthly record is aggregated. The values are the interval polars wants, which both `date_range`
# and `Expr.dt.truncate` read, so one string builds the grid and lands the events on it.
AggregationFrequency = Literal["annual", "quarterly", "monthly"]
AGGREGATION_INTERVALS: dict[AggregationFrequency, str] = {"annual": "1y", "quarterly": "1q", "monthly": "1mo"}
MONTHS_PER_PERIOD: dict[AggregationFrequency, int] = {"annual": 12, "quarterly": 3, "monthly": 1}


def mean_by_period(series: pl.DataFrame, frequency: AggregationFrequency, *, date: str, value: str) -> pl.DataFrame:
    """
    Average a monthly series over each period, dropping any period with a month missing.

    Parameters
    ----------
    series : DataFrame
        One row per month.
    frequency : {'annual', 'quarterly', 'monthly'}
        How long one period runs.
    date : str
        The column holding each row's date.
    value : str
        The column to average.

    Returns
    -------
    averaged : DataFrame
        One row per complete period, dated to its first day, sorted.
    """
    period = pl.col(date).dt.truncate(AGGREGATION_INTERVALS[frequency]).alias(date)

    return (
        series.group_by(period)
        .agg(pl.col(value).mean(), pl.col(date).dt.month().n_unique().alias("months"))
        .filter(pl.col("months") == MONTHS_PER_PERIOD[frequency])
        .drop("months")
        .sort(date)
    )
