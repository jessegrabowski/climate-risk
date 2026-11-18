import logging

from datetime import datetime

import polars as pl
import pytest

from climate_risk.data import weo
from climate_risk.data.weo import GROSS_DEBT_INDICATOR, load_government_debt, transform_government_debt


def served(rows) -> pl.DataFrame:
    """The indicator as kuznets returns it, under the WEO's own dimension names."""
    frame = pl.DataFrame(rows, schema=["COUNTRY", "period", "value"], orient="row")

    return frame.with_columns(pl.col("period").cast(pl.Datetime))


@pytest.fixture
def serves(monkeypatch):
    """Answer IMFReader from a prepared frame, recording the request it was given."""

    def serve(frame: pl.DataFrame):
        asked = []

        class FakeReader:
            def __init__(self, dataflow, selections=None, **kwargs):
                asked.append((dataflow, selections, kwargs))

            def read(self):
                return frame

        monkeypatch.setattr(weo, "IMFReader", FakeReader)
        return asked

    return serve


def test_debt_is_stored_as_a_fraction_of_gdp():
    """The WEO reports a percentage, and the great ratios it is compared against are fractions."""
    frame = transform_government_debt(served([("IND", datetime(2015, 1, 1), 69.7)]))

    assert frame.rows() == [("IND", 2015, pytest.approx(0.697))]


def test_debt_above_gdp_is_kept():
    """Debt is a stock against a flow, so a ratio above one is ordinary rather than a bad scale."""
    frame = transform_government_debt(served([("JPN", datetime(2015, 1, 1), 231.3)]))

    assert frame["government_debt"].item() == pytest.approx(2.313)


def test_a_year_the_weo_reports_nothing_for_is_dropped():
    frame = transform_government_debt(
        served([("IND", datetime(2015, 1, 1), 69.7), ("IND", datetime(2016, 1, 1), None)])
    )

    assert frame["year"].to_list() == [2015]


def test_a_renamed_dimension_is_rejected():
    """kuznets names the columns after the WEO's dimensions. A renamed one would otherwise surface as
    a missing attribute somewhere downstream.
    """
    renamed = served([("IND", datetime(2015, 1, 1), 69.7)]).rename({"COUNTRY": "REF_AREA"})

    with pytest.raises(ValueError, match="COUNTRY"):
        transform_government_debt(renamed)


def test_the_whole_gross_debt_series_is_asked_for(tmp_path, serves):
    """Left to itself the service answers with recent years only, which reads as a short series."""
    asked = serves(served([("IND", datetime(2015, 1, 1), 69.7)]))

    load_government_debt(tmp_path, ["IND"])

    dataflow, selections, kwargs = asked[0]
    assert (dataflow, selections["INDICATOR"]) == ("WEO", GROSS_DEBT_INDICATOR)
    assert kwargs["start"] <= 1980


def test_a_warm_cache_is_read_back_without_asking_again(tmp_path, serves):
    asked = serves(served([("IND", datetime(2015, 1, 1), 69.7)]))

    cold = load_government_debt(tmp_path, ["IND"])
    warm = load_government_debt(tmp_path, ["IND"])

    assert len(asked) == 1
    assert cold.equals(warm)


def test_the_countries_asked_for_key_the_cache_in_any_order(tmp_path, serves):
    serves(served([("IND", datetime(2015, 1, 1), 69.7), ("LAO", datetime(2015, 1, 1), 60.0)]))

    load_government_debt(tmp_path, ["LAO", "IND"])
    load_government_debt(tmp_path, ["IND", "LAO"])

    assert len(list(tmp_path.glob("weo_government_debt__*.parquet"))) == 1


def test_a_country_the_weo_answers_nothing_for_is_named(tmp_path, serves, caplog):
    serves(served([("IND", datetime(2015, 1, 1), 69.7)]))

    with caplog.at_level(logging.WARNING, logger="climate_risk.data.weo"):
        load_government_debt(tmp_path, ["IND", "LAO"])

    assert "LAO" in caplog.text


def test_an_answer_that_is_not_a_frame_is_rejected(tmp_path, monkeypatch):
    """kuznets honors output_type on its own, so a non-frame answer means the contract moved."""

    class FakeReader:
        def __init__(self, dataflow, selections=None, **kwargs):
            pass

        def read(self):
            return object()

    monkeypatch.setattr(weo, "IMFReader", FakeReader)

    with pytest.raises(TypeError, match="output_type='polars'"):
        load_government_debt(tmp_path, ["IND"])
