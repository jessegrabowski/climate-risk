from datetime import date

import polars as pl
import pytest

from climate_risk.data.deflate import (
    US_PRICE_LEVEL,
    annual_price_level,
    deflate,
    deflation_factors,
    rebase,
)
from climate_risk.data.fred import SERIES_NAMES


def monthly(levels: dict[int, float], *, series: str = US_PRICE_LEVEL, months: int = 12) -> pl.DataFrame:
    """The FRED panel carrying one index, flat within each year so its average is the level given."""
    rows = [(series, date(year, month, 1), level) for year, level in levels.items() for month in range(1, months + 1)]

    return pl.DataFrame(rows, schema=["series", "date", "value"], orient="row")


def priced(levels: dict[int, float]) -> pl.DataFrame:
    """Annual index levels, as :func:`annual_price_level` returns them."""
    return pl.DataFrame({"year": list(levels), "price_level": list(levels.values())})


def damages(rows) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["year", "damage"], orient="row")


def test_the_months_of_a_year_average_onto_it():
    observations = monthly({2015: 100.0}).with_columns(
        pl.when(pl.col("date").dt.month() <= 6).then(90.0).otherwise(110.0).alias("value")
    )

    assert annual_price_level(observations).rows() == [(2015, 100.0)]


def test_the_year_the_index_is_still_publishing_is_dropped():
    """A year averaged over the months that have arrived so far is not that year's price level, and it
    is the newest year, so it is the one a caller is most likely to base on.
    """
    observations = pl.concat([monthly({2015: 100.0}), monthly({2016: 200.0}, months=11)])

    assert annual_price_level(observations).rows() == [(2015, 100.0)]


def test_a_closed_year_the_index_skipped_a_month_of_is_kept():
    """The BLS published no October 2025 consumer price index and never went back for it, so a year is
    missing a month permanently rather than pending, and dropping it loses that base year for good.
    """
    observations = pl.concat([monthly({2015: 100.0}), monthly({2016: 100.0})]).filter(
        ~((pl.col("date").dt.year() == 2015) & (pl.col("date").dt.month() == 10))
    )

    assert annual_price_level(observations)["year"].to_list() == [2015, 2016]


def test_a_series_the_panel_does_not_carry_is_rejected():
    with pytest.raises(ValueError, match="no series named"):
        annual_price_level(monthly({2015: 100.0}), series="world_rate_3m")


def test_a_series_reported_less_often_than_monthly_is_rejected():
    """Averaging a quarterly series over its four observations reads as an annual level and is not one."""
    with pytest.raises(ValueError, match="not a monthly index"):
        annual_price_level(monthly({2015: 100.0, 2016: 110.0}, months=4))


def test_the_base_year_carries_a_factor_of_one():
    factors = deflation_factors(priced({2010: 80.0, 2015: 100.0, 2020: 125.0}), base_year=2015)

    assert dict(factors.rows()) == {2010: pytest.approx(1.25), 2015: 1.0, 2020: pytest.approx(0.8)}


def test_a_base_year_the_index_does_not_cover_is_rejected():
    with pytest.raises(ValueError, match="cannot be based on 1970"):
        deflation_factors(priced({2010: 80.0, 2015: 100.0}), base_year=1970)


def test_each_amount_moves_by_the_price_level_of_its_own_year():
    """The whole point of deflating rather than rebasing: two equal nominal amounts a decade apart are
    not equal in constant dollars.
    """
    frame = deflate(
        damages([(2010, 100.0), (2020, 100.0)]),
        priced({2010: 80.0, 2015: 100.0, 2020: 125.0}),
        ["damage"],
        base_year=2015,
    )

    assert frame["damage"].to_list() == [pytest.approx(125.0), pytest.approx(80.0)]


def test_a_money_column_the_caller_did_not_name_passes_through():
    """Only the named columns are money. A count or a rate beside them would be nonsense deflated."""
    frame = deflate(
        damages([(2010, 100.0)]).with_columns(pl.lit(3).alias("events")),
        priced({2010: 80.0, 2015: 100.0}),
        ["damage"],
        base_year=2015,
    )

    assert frame.rows() == [(2010, pytest.approx(125.0), 3)]


def test_a_year_the_index_does_not_cover_is_rejected():
    """A left join would hand those rows a null deflator and null the money out, which reads downstream
    as an event that did no damage.
    """
    with pytest.raises(ValueError, match=r"does not cover \[1970\]"):
        deflate(damages([(1970, 100.0), (2015, 100.0)]), priced({2015: 100.0}), ["damage"], base_year=2015)


def test_an_amount_carrying_no_year_is_rejected():
    undated = damages([(2015, 100.0)]).vstack(pl.DataFrame({"year": [None], "damage": [100.0]}))

    with pytest.raises(ValueError, match="carry no year"):
        deflate(undated, priced({2015: 100.0}), ["damage"], base_year=2015)


def test_rebasing_moves_every_amount_by_one_factor():
    """EM-DAT states its adjusted damages in one year's dollars whatever year the event fell in, so the
    year beside an amount is the event's date and not the date of the money.
    """
    frame = rebase(
        damages([(1970, 100.0), (2020, 100.0)]),
        priced({2015: 100.0, 2020: 125.0}),
        ["damage"],
        source_year=2020,
        base_year=2015,
    )

    assert frame["damage"].to_list() == [pytest.approx(80.0), pytest.approx(80.0)]


def test_a_source_year_the_index_does_not_cover_is_rejected():
    with pytest.raises(ValueError, match="does not reach 2050"):
        rebase(
            damages([(2020, 100.0)]), priced({2015: 100.0, 2020: 125.0}), ["damage"], source_year=2050, base_year=2015
        )


def test_the_price_level_reads_the_consumer_price_index_the_fred_panel_downloads():
    """The two modules are joined by a name rather than by an import, so renaming it in either one
    leaves this module raising at runtime for a series that is right there.
    """
    assert SERIES_NAMES["CPIAUCSL"] == US_PRICE_LEVEL
