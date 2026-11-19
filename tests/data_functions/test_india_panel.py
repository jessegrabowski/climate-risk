import numpy as np
import pandas as pd
import pytest

from statsmodels.tools.sm_exceptions import X13NotFoundError

from climate_risk.data_functions import india_panel
from climate_risk.data_functions.india_panel import (
    ANNUAL_SERIES,
    FRED_SERIES,
    NATIONAL_ACCOUNTS,
    QUARTERLY_FRED_SERIES,
    QUARTERLY_SERIES,
    assemble_panel,
    complete_quarter_mean,
    foreign_quarterly,
    india_panel_metadata,
    place_annual,
    seasonally_adjust,
    splice_on_growth,
    world_bank_annual_series,
)


def quarters(start, periods):
    return pd.period_range(start, periods=periods, freq="Q")


def test_splicing_keeps_the_anchor_and_carries_the_other_bases_growth():
    """Bases break in level, so the anchor's levels are kept and only the other bases' growth rates reach past them."""
    levels = pd.DataFrame(
        {"old": [10.0, 20.0, 40.0, np.nan], "anchor": [np.nan, np.nan, 100.0, 110.0]},
        index=quarters("2000Q1", 4),
    )

    spliced = splice_on_growth(levels, anchor="anchor")

    assert spliced.tolist() == [25.0, 50.0, 100.0, 110.0]


def test_splicing_backward_prefers_the_newest_other_base():
    """Going back from the anchor, the base nearest it in time is the closest in method, so it supplies the growth."""
    levels = pd.DataFrame(
        {"oldest": [1.0, 1.0, np.nan], "middle": [1.0, 2.0, np.nan], "anchor": [np.nan, 100.0, 120.0]},
        index=quarters("2000Q1", 3),
    )

    spliced = splice_on_growth(levels, anchor="anchor")

    assert spliced.iloc[0] == 50.0


def test_splicing_forward_prefers_the_oldest_other_base():
    """Going forward from the anchor, the earlier of the other bases supplies the growth."""
    levels = pd.DataFrame(
        {"anchor": [100.0, 120.0, np.nan], "middle": [np.nan, 1.0, 2.0], "newest": [np.nan, 1.0, 3.0]},
        index=quarters("2000Q1", 3),
    )

    spliced = splice_on_growth(levels, anchor="anchor")

    assert spliced.iloc[-1] == 240.0


def test_an_anchor_with_no_observations_names_the_bases_there_are():
    """A renamed sheet is read as another base, which leaves the anchor's column empty."""
    levels = pd.DataFrame({"2012": [1.0, 2.0], "2024": [np.nan, np.nan]}, index=quarters("2000Q1", 2))

    with pytest.raises(ValueError, match=r"'2024' has no observations.*\['2012'\]"):
        splice_on_growth(levels, anchor="2024")


def test_splicing_stops_at_the_first_gap_no_base_bridges():
    levels = pd.DataFrame(
        {"old": [5.0, np.nan, 40.0, np.nan], "anchor": [np.nan, np.nan, 100.0, 110.0]},
        index=quarters("2000Q1", 4),
    )

    spliced = splice_on_growth(levels, anchor="anchor")

    assert list(spliced.index) == list(quarters("2000Q3", 2))


def test_a_quarter_missing_a_month_is_dropped():
    """A two-month mean would read a seasonal swing as a level change."""
    monthly = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=pd.period_range("2000-01", periods=5, freq="M"))

    quarterly = complete_quarter_mean(monthly)

    assert quarterly.to_dict() == {pd.Period("2000Q1"): 2.0}


def test_fiscal_years_close_in_the_first_quarter_of_the_next_calendar_year():
    """India's fiscal year 2015, April 2015 to March 2016, is known only once 2016Q1 ends."""
    values = pd.Series([0.5, 0.7], index=[2015, 2016])

    fiscal = place_annual(values, "fiscal")
    calendar = place_annual(values, "calendar")

    assert list(fiscal.index) == [pd.Period("2016Q1"), pd.Period("2017Q1")]
    assert list(calendar.index) == [pd.Period("2015Q4"), pd.Period("2016Q4")]


def test_an_annual_value_enters_only_when_its_whole_year_is_in_the_sample():
    """A year straddling the sample's start would be observed as if the sample covered it."""
    sample = quarters("2000Q3", 8)
    quarterly = {component: pd.Series(1.0, index=sample) for component in NATIONAL_ACCOUNTS}
    annual = {"exchange_rate": pd.Series([40.0, 41.0], index=[2000, 2001])}

    panel = assemble_panel(quarterly, annual)

    assert panel["exchange_rate"].dropna().to_dict() == {pd.Timestamp("2001-10-01"): 41.0}


def test_the_panel_is_indexed_by_quarter_start_dates():
    """The estimation notebooks read the panel by date, so each quarter is the timestamp of its first day."""
    sample = quarters("2000Q1", 4)
    quarterly = {component: pd.Series(1.0, index=sample) for component in NATIONAL_ACCOUNTS}

    panel = assemble_panel(quarterly, {})

    assert panel.index.name == "quarter_start"
    assert list(panel.index) == list(pd.to_datetime(["2000-01-01", "2000-04-01", "2000-07-01", "2000-10-01"]))


def fred_panel(extra_series=()):
    """FRED's tidy frame for 2000Q1: one observation of each quarterly series, three months of every other one."""
    months = pd.to_datetime(["2000-01-01", "2000-02-01", "2000-03-01"])
    rows = [
        (name, date, float(month))
        for name in [*FRED_SERIES, *extra_series]
        for month, date in enumerate(months[:1] if name in QUARTERLY_FRED_SERIES else months, start=1)
    ]
    return pd.DataFrame(rows, columns=["series", "date", "value"])


def test_fred_world_output_keeps_its_published_quarters():
    """World output is published quarterly, and the monthly series are averaged over each quarter's three months."""
    foreign = foreign_quarterly(fred_panel())

    assert foreign["world_output"].to_dict() == {pd.Period("2000Q1"): 1.0}
    assert foreign["world_rate_3m"].to_dict() == {pd.Period("2000Q1"): 2.0}


def test_a_fred_series_the_panel_does_not_use_stays_out_of_it():
    """The FRED loader serves other notebooks too, and a series added for one must not become a panel column."""
    foreign = foreign_quarterly(fred_panel(extra_series=["unrelated_series"]))

    assert sorted(foreign) == sorted(FRED_SERIES)


def test_the_metadata_describes_every_column_of_the_panel():
    sample = quarters("2000Q1", 8)
    quarterly = {name: pd.Series(1.0, index=sample) for name in QUARTERLY_SERIES}
    annual = {name: pd.Series([1.0], index=[2001]) for name in ANNUAL_SERIES}

    panel = assemble_panel(quarterly, annual)
    metadata = india_panel_metadata(panel)

    assert list(metadata.index) == list(panel.columns)
    assert metadata.loc["exchange_rate", "observations"] == 1


def test_world_bank_shares_are_nominal_and_annual_inflation_stops_where_the_quarterly_cpi_starts():
    """Annual CPI inflation fills only the years before the RBI's quarterly CPI, so no year is counted twice."""
    world_bank = pd.DataFrame(
        {
            "real_gdp_lcu": [100.0, 100.0, 100.0],
            "gdp_deflator": [200.0, 200.0, 200.0],
            "nominal_consumption": [120.0, 120.0, 120.0],
            "nominal_investment": [60.0, 60.0, 60.0],
            "tax_revenue_gdp": [10.0, 10.0, 10.0],
            "government_expense_gdp": [20.0, 20.0, 20.0],
            "cpi": [100.0, 110.0, 121.0],
            "exchange_rate": [40.0, 41.0, 42.0],
            "lending_rate": [8.0, 8.0, 8.0],
            "external_debt": [1.0, 1.0, 1.0],
            "employment_rate": [50.0, 50.0, 50.0],
        },
        index=[2010, 2011, 2012],
    )

    series = world_bank_annual_series(world_bank, cpi_first_year=2012)

    assert series["nominal_consumption_share"].tolist() == [0.6, 0.6, 0.6]
    assert series["cpi_inflation_annual"].dropna().to_dict() == {2011: pytest.approx(0.1)}


def test_a_gap_inside_a_series_is_named_before_x13_sees_it(monkeypatch):
    """X-13 rejects a missing quarter with a parse error that names neither the series nor the quarter."""
    monkeypatch.setattr(india_panel, "x13_arima_analysis", lambda *args, **kwargs: pytest.fail("X-13 was called"))
    series = pd.Series(1.0, index=quarters("2000Q1", 12), name="cpi").drop([pd.Period("2001Q2"), pd.Period("2001Q3")])

    with pytest.raises(ValueError, match=r"'cpi' is missing \['2001Q2', '2001Q3'\]"):
        seasonally_adjust(series, log=True)


def test_a_missing_x13_binary_says_where_to_get_it(monkeypatch):
    def missing(*args, **kwargs):
        raise X13NotFoundError("x13as not found on path")

    monkeypatch.setattr(india_panel, "x13_arima_analysis", missing)

    with pytest.raises(X13NotFoundError, match=r"census\.gov"):
        seasonally_adjust(pd.Series(1.0, index=quarters("2000Q1", 12)), log=True)
