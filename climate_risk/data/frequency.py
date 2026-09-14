from typing import Literal

# How coarsely a monthly record is aggregated. The values are the interval polars wants, which both `date_range`
# and `Expr.dt.truncate` read, so one string builds the grid and lands the events on it.
AggregationFrequency = Literal["annual", "quarterly", "monthly"]
AGGREGATION_INTERVALS: dict[AggregationFrequency, str] = {"annual": "1y", "quarterly": "1q", "monthly": "1mo"}
MONTHS_PER_PERIOD: dict[AggregationFrequency, int] = {"annual": 12, "quarterly": 3, "monthly": 1}
