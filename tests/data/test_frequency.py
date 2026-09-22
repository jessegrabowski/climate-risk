from datetime import date

import polars as pl
import pytest

from climate_risk.data.frequency import mean_by_period


def dated(values_by_month: dict[tuple[int, int], float]) -> pl.DataFrame:
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
    series = dated({(1990, month): float(month) for month in range(1, 13)})

    averaged = mean_by_period(series, frequency, date="Date", value="value")

    assert dict(zip(averaged["Date"], averaged["value"], strict=True)) == pytest.approx(expected)


def test_a_period_missing_a_month_is_dropped():
    """A year averaged from eleven months would be published under the name of twelve."""
    series = dated({(1990, month): 1.0 for month in range(1, 13)} | {(1991, month): 2.0 for month in range(1, 12)})

    averaged = mean_by_period(series, "annual", date="Date", value="value")

    assert averaged["Date"].to_list() == [date(1990, 1, 1)]


def test_a_quarterly_record_averages_its_four_rows_to_a_year():
    series = dated({(1990, 1): 1.0, (1990, 4): 2.0, (1990, 7): 3.0, (1990, 10): 4.0, (1991, 1): 9.0})

    averaged = mean_by_period(series, "annual", date="Date", value="value", resolution="quarterly")

    assert dict(zip(averaged["Date"], averaged["value"], strict=True)) == pytest.approx({date(1990, 1, 1): 2.5})


def test_a_request_finer_than_the_record_returns_the_rows_as_published():
    series = dated({(1990, 4): 2.0, (1990, 1): 1.0})

    monthly_rows = mean_by_period(series, "monthly", date="Date", value="value", resolution="quarterly")

    assert monthly_rows["Date"].to_list() == [date(1990, 1, 1), date(1990, 4, 1)]
    assert monthly_rows["value"].to_list() == [1.0, 2.0]
