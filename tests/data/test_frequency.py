from datetime import date

import polars as pl
import pytest

from climate_risk.data.frequency import mean_by_period


def monthly(values_by_month: dict[tuple[int, int], float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Date": [date(year, month, 1) for year, month in values_by_month],
            "value": list(values_by_month.values()),
        }
    )


@pytest.mark.parametrize(
    ("frequency", "expected"),
    [
        ("monthly", {date(1990, month, 1): float(month) for month in range(1, 13)}),
        ("quarterly", {date(1990, 1, 1): 2.0, date(1990, 4, 1): 5.0, date(1990, 7, 1): 8.0, date(1990, 10, 1): 11.0}),
        ("annual", {date(1990, 1, 1): 6.5}),
    ],
)
def test_each_period_averages_the_months_inside_it(frequency, expected):
    series = monthly({(1990, month): float(month) for month in range(1, 13)})

    averaged = mean_by_period(series, frequency, date="Date", value="value")

    assert dict(zip(averaged["Date"], averaged["value"], strict=True)) == pytest.approx(expected)


def test_a_period_missing_a_month_is_dropped():
    """A year averaged from eleven months would be published under the name of twelve."""
    series = monthly({(1990, month): 1.0 for month in range(1, 13)} | {(1991, month): 2.0 for month in range(1, 12)})

    averaged = mean_by_period(series, "annual", date="Date", value="value")

    assert averaged["Date"].to_list() == [date(1990, 1, 1)]
