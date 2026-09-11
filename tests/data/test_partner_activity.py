import logging

from datetime import datetime

import polars as pl
import pytest

from climate_risk.data import partner_activity
from climate_risk.data.partner_activity import load_partner_activity, transform_partner_activity

BASE_YEAR = 2015
FIRST_YEAR = 2015
LAST_YEAR = 2016


def activity(exports, gdp, **overrides):
    """Run the transform over a two-year window, which every fixture below supplies output for."""
    window = {"base_year": BASE_YEAR, "first_year": FIRST_YEAR, "last_year": LAST_YEAR} | overrides

    return transform_partner_activity(exports, gdp, **window)


def exported(rows) -> pl.DataFrame:
    """Bilateral exports shaped as kuznets returns them from IMTS."""
    return pl.DataFrame(rows, schema=["country", "counterpart", "period", "value"], orient="row")


def outputs(rows) -> pl.DataFrame:
    """The macro panel, carrying only the columns the transform reads."""
    return pl.DataFrame(rows, schema=["country_code", "year", "real_gdp_lcu"], orient="row")


def one_year(country: str, counterpart: str, value: float):
    return (country, counterpart, datetime(2015, 1, 1), value)


def test_partner_output_is_indexed_to_the_base_year():
    """Every partner sits at its own base year, so the index is 1 there whatever the weights are."""
    exports = exported([one_year("LAO", "THA", 70.0), one_year("LAO", "CHN", 30.0)])
    gdp = outputs([("THA", 2015, 100.0), ("THA", 2016, 110.0), ("CHN", 2015, 500.0), ("CHN", 2016, 550.0)])

    frame = activity(exports, gdp)

    assert frame.filter(pl.col("year") == 2015)["partner_activity"].to_list() == [1.0]


def test_partners_are_weighted_by_their_share_of_exports():
    """A partner taking 70% of exports carries 70% of the index. Weighting the log output makes that a
    geometric mean, so a partner that doubles while the other stands still gives 2 raised to its share.
    """
    exports = exported([one_year("LAO", "THA", 70.0), one_year("LAO", "CHN", 30.0)])
    gdp = outputs([("THA", 2015, 100.0), ("THA", 2016, 200.0), ("CHN", 2015, 100.0), ("CHN", 2016, 100.0)])

    frame = activity(exports, gdp)

    assert frame.filter(pl.col("year") == 2016)["partner_activity"].item() == pytest.approx(2.0**0.7)


def test_a_partner_without_output_does_not_read_as_lost_demand():
    """Weights are renormalized over the retained set. Without that, a partner absent from the output
    panel would drag the index toward zero and look like a foreign slump.
    """
    exports = exported([one_year("LAO", "THA", 50.0), one_year("LAO", "XXX", 50.0)])
    gdp = outputs([("THA", 2015, 100.0), ("THA", 2016, 200.0)])

    frame = activity(exports, gdp)

    row = frame.filter(pl.col("year") == 2016)
    assert row["partner_activity"].item() == pytest.approx(2.0)
    assert row["partner_coverage"].item() == pytest.approx(0.5)


def test_the_fund_s_regional_aggregates_are_not_treated_as_partners():
    """IMTS mixes aggregates like `G001` into the counterpart dimension. Counting them alongside the
    countries they contain double-counts the export base and halves every real partner's weight.
    """
    exports = exported([one_year("LAO", "THA", 50.0), one_year("LAO", "G001", 50.0)])
    gdp = outputs([("THA", 2015, 100.0), ("THA", 2016, 200.0)])

    frame = activity(exports, gdp)

    assert frame.filter(pl.col("year") == 2016)["partner_coverage"].item() == pytest.approx(1.0)


def test_a_country_is_not_its_own_trading_partner():
    """Some reporters carry a row against themselves, which would put domestic output into the
    foreign activity measure.
    """
    exports = exported([one_year("LAO", "THA", 50.0), one_year("LAO", "LAO", 50.0)])
    gdp = outputs([("THA", 2015, 100.0), ("THA", 2016, 110.0), ("LAO", 2015, 10.0), ("LAO", 2016, 11.0)])

    frame = activity(exports, gdp)

    assert frame["partner_coverage"].unique().to_list() == [1.0]


def test_a_base_year_no_country_reports_is_rejected():
    """Indexing silently produces an empty panel, which reads downstream as a country with no
    trading partners rather than as a bad argument.
    """
    exports = exported([one_year("LAO", "THA", 70.0)])
    gdp = outputs([("THA", 2015, 100.0)])

    with pytest.raises(ValueError, match="1900"):
        activity(exports, gdp, base_year=1900)


def test_each_country_is_weighted_against_its_own_exports():
    """Shares are taken within a country. Dividing by the panel's total exports instead would make
    every weight depend on which other countries happened to be requested.
    """
    exports = exported([one_year("LAO", "THA", 50.0), one_year("ZMB", "CHN", 900.0), one_year("ZMB", "THA", 100.0)])
    gdp = outputs([("THA", 2015, 100.0), ("THA", 2016, 110.0), ("CHN", 2015, 100.0), ("CHN", 2016, 110.0)])

    frame = activity(exports, gdp)

    assert dict(frame.select("country_code", "partner_coverage").iter_rows()) == pytest.approx({"LAO": 1.0, "ZMB": 1.0})


def test_a_partner_with_a_gap_in_its_output_is_excluded_from_every_year():
    """A partner entering or leaving mid-sample shifts the weighted mean by a discontinuous constant,
    which is indistinguishable in the series from a movement in foreign demand. The partner set is
    therefore fixed across the window, and coverage with it.
    """
    exports = exported([one_year("LAO", "THA", 50.0), one_year("LAO", "CHN", 50.0)])
    gdp = outputs(
        [("THA", 2015, 100.0), ("THA", 2016, 200.0), ("CHN", 2015, 100.0)]  # CHN has no 2016
    )

    frame = activity(exports, gdp)

    assert frame["partner_coverage"].unique().to_list() == [pytest.approx(0.5)]
    assert frame.sort("year")["partner_activity"].to_list() == [pytest.approx(1.0), pytest.approx(2.0)]


def test_a_window_that_ends_before_it_starts_is_rejected():
    exports = exported([one_year("LAO", "THA", 70.0)])
    gdp = outputs([("THA", 2015, 100.0), ("THA", 2016, 110.0)])

    with pytest.raises(ValueError, match="2016-2015"):
        activity(exports, gdp, first_year=2016, last_year=2015)


def test_thin_partner_coverage_is_reported(caplog):
    """An index built from a minority of a country's export markets is not obviously wrong from its
    values alone, so the share it rests on is logged.
    """
    exports = exported([one_year("LAO", "THA", 10.0), one_year("LAO", "CHN", 90.0)])
    gdp = outputs([("THA", 2015, 100.0), ("THA", 2016, 200.0), ("CHN", 2015, 100.0)])

    with caplog.at_level(logging.WARNING, logger="climate_risk.data.partner_activity"):
        activity(exports, gdp)

    assert "LAO" in caplog.text
    assert "10.0%" in caplog.text


def test_a_warm_cache_reaches_neither_upstream(tmp_path, monkeypatch):
    """Both upstreams are read inside the builder, so a second call must reach neither."""
    reached = []

    def fake_load_wb_data(*args, **kwargs):
        reached.append("world_bank")
        return outputs([("THA", 2015, 100.0), ("THA", 2016, 200.0)])

    class FakeReader:
        def __init__(self, *args, **kwargs):
            reached.append("imts")

        def read(self):
            return exported([one_year("LAO", "THA", 50.0)])

    monkeypatch.setattr(partner_activity, "load_wb_data", fake_load_wb_data)
    monkeypatch.setattr(partner_activity.imf, "IMTSReader", FakeReader)
    cold = load_partner_activity(tmp_path, ["LAO"], base_year=2015, first_year=2015, last_year=2016)
    reached.clear()

    warm = load_partner_activity(tmp_path, ["LAO"], base_year=2015, first_year=2015, last_year=2016)

    assert reached == []
    assert warm.equals(cold)


def test_the_countries_asked_for_key_the_cache_in_any_order(tmp_path, monkeypatch):
    """The codes are sorted before they key the entry. Without that, the same pair asked for the other
    way round builds a second panel and downloads again.
    """
    downloads = []

    class FakeReader:
        def __init__(self, symbols, **kwargs):
            downloads.append(symbols)

        def read(self):
            return exported([one_year("LAO", "THA", 50.0), one_year("ZMB", "THA", 50.0)])

    monkeypatch.setattr(
        partner_activity,
        "load_wb_data",
        lambda *a, **k: outputs([("THA", 2015, 100.0), ("THA", 2016, 200.0)]),
    )
    monkeypatch.setattr(partner_activity.imf, "IMTSReader", FakeReader)

    load_partner_activity(tmp_path, ["ZMB", "LAO"], base_year=2015, first_year=2015, last_year=2016)
    load_partner_activity(tmp_path, ["LAO", "ZMB"], base_year=2015, first_year=2015, last_year=2016)

    assert len(downloads) == 1
    assert len(list(tmp_path.glob("partner_activity__*.parquet"))) == 1
