from typing import Literal

import polars as pl

# How coarsely a monthly record is aggregated. The values are the interval polars wants, which both `date_range`
# and `Expr.dt.truncate` read, so one string builds the grid and lands the events on it.
AggregationFrequency = Literal["annual", "quarterly", "monthly"]
AGGREGATION_INTERVALS: dict[AggregationFrequency, str] = {"annual": "1y", "quarterly": "1q", "monthly": "1mo"}
MONTHS_PER_PERIOD: dict[AggregationFrequency, int] = {"annual": 12, "quarterly": 3, "monthly": 1}


def mean_by_period(
    series: pl.DataFrame,
    frequency: AggregationFrequency,
    *,
    date: str,
    value: str,
    resolution: AggregationFrequency = "monthly",
) -> pl.DataFrame:
    """
    Average a series over each period, dropping any period the series does not cover in full.

    Parameters
    ----------
    series : DataFrame
        One row per period of ``resolution``.
    frequency : {'annual', 'quarterly', 'monthly'}
        How long one period runs.
    date : str
        The column holding each row's date.
    value : str
        The column to average.
    resolution : {'annual', 'quarterly', 'monthly'}, optional
        How long one row of ``series`` runs. A request finer than this returns the rows as they are.
        Default ``'monthly'``.

    Returns
    -------
    averaged : DataFrame
        One row per complete period, dated to its first day, sorted.
    """
    rows_per_period = MONTHS_PER_PERIOD[frequency] // MONTHS_PER_PERIOD[resolution]
    if rows_per_period == 0:
        return series.sort(date)

    period = pl.col(date).dt.truncate(AGGREGATION_INTERVALS[frequency]).alias(date)

    return (
        series.group_by(period)
        .agg(pl.col(value).mean(), pl.len().alias("rows"))
        .filter(pl.col("rows") == rows_per_period)
        .drop("rows")
        .sort(date)
    )
