import logging

from datetime import datetime

import polars as pl
import pytest

from climate_risk.data import ilo
from climate_risk.data.ilo import (
    UNCOVERED_AREAS,
    load_labour_income_share,
    transform_labour_share,
)
from climate_risk.data.world_bank import REQUESTED_COUNTRY_CODES

FIRST_ILO_YEAR = 2004


def served(rows) -> pl.DataFrame:
    """The dataflow as kuznets returns it, under the ILO's own dimension names."""
    frame = pl.DataFrame(rows, schema=["REF_AREA", "period", "value"], orient="row")

    return frame.with_columns(pl.col("period").cast(pl.Datetime))


@pytest.fixture
def serves(monkeypatch):
    """Answer ILOSTATReader from a prepared frame, recording the selections it was asked for."""

    def serve(frame: pl.DataFrame):
        asked = []

        class FakeReader:
            def __init__(self, dataflow, selections=None, **kwargs):
                asked.append((dataflow, selections, kwargs))

            def read(self):
                return frame

        monkeypatch.setattr(ilo, "ILOSTATReader", FakeReader)
        return asked

    return serve


def test_a_share_of_gdp_is_stored_as_a_fraction():
    """The dataflow reports a percentage and Penn World Table reports a fraction, so a model reading
    both would scale one of them by a hundred.
    """
    frame = transform_labour_share(served([("NPL", datetime(2010, 1, 1), 52.5)]))

    assert frame.rows() == [("NPL", 2010, pytest.approx(0.525))]


def test_a_series_already_expressed_as_a_fraction_is_rejected():
    """Dividing it again leaves a hundredth of itself, still inside the nought-to-one band the column
    is documented to hold, so nothing downstream could tell.
    """
    fractions = served([("NPL", datetime(2010, 1, 1), 0.525), ("LAO", datetime(2010, 1, 1), 0.485)])

    with pytest.raises(ValueError, match="not a percentage"):
        transform_labour_share(fractions)


@pytest.mark.parametrize(
    "shares",
    [[140.0], [-1.0, 50.0]],
    ids=["above a hundred", "below zero beside a credible share"],
)
def test_a_share_outside_nought_to_a_hundred_is_rejected(shares):
    """The negative case carries a credible share alongside it. On its own its maximum is under one,
    which the fraction check would catch first and leave this branch unexercised.
    """
    rows = [("NPL", datetime(2010 + offset, 1, 1), share) for offset, share in enumerate(shares)]

    with pytest.raises(ValueError, match="not a percentage"):
        transform_labour_share(served(rows))


def test_a_dataflow_carrying_no_observations_is_not_read_as_a_bad_scale():
    """An empty answer has no range to judge, and raising on it would turn a quiet source into a crash."""
    empty = served([]).clear()

    assert transform_labour_share(empty).is_empty()


def test_the_year_is_an_integer_the_other_panels_join_on():
    frame = transform_labour_share(served([("LAO", datetime(2010, 1, 1), 48.5)]))

    assert frame.schema["year"] == pl.Int64


def test_a_renamed_dimension_is_rejected():
    """kuznets names the columns after the ILO's dimensions. A renamed one would otherwise surface as
    a missing attribute somewhere downstream.
    """
    renamed = served([("NPL", datetime(2010, 1, 1), 52.5)]).rename({"REF_AREA": "AREA"})

    with pytest.raises(ValueError, match="REF_AREA"):
        transform_labour_share(renamed)


def test_a_year_the_ilo_reports_nothing_for_is_dropped():
    """A null share is the absence of an estimate, and carrying it as a row would put a hole in the
    middle of a series that joins cleanly without it.
    """
    frame = transform_labour_share(served([("NPL", datetime(2010, 1, 1), 52.5), ("NPL", datetime(2011, 1, 1), None)]))

    assert frame["year"].to_list() == [2010]


def test_the_areas_the_ilo_cannot_name_are_not_asked_for(tmp_path, serves):
    """One unknown code fails the whole request, so asking for every World Bank country would never
    return anything at all.
    """
    asked = serves(served([("LAO", datetime(2010, 1, 1), 48.5)]))

    load_labour_income_share(tmp_path, ["LAO", *UNCOVERED_AREAS])

    _, selections, _ = asked[0]
    assert selections == {"REF_AREA": ["LAO"]}


def test_a_country_the_ilo_answers_nothing_for_is_named(tmp_path, serves, caplog):
    """Losing a country to a source that simply has no estimate for it is the failure this project
    keeps hitting, so it is reported rather than left to be noticed in a row count.
    """
    serves(served([("LAO", datetime(2010, 1, 1), 48.5)]))

    with caplog.at_level(logging.WARNING, logger="climate_risk.data.ilo"):
        load_labour_income_share(tmp_path, ["LAO", "NPL"])

    assert "NPL" in caplog.text


def test_a_warm_cache_is_read_back_without_asking_again(tmp_path, serves):
    asked = serves(served([("LAO", datetime(2010, 1, 1), 48.5)]))

    cold = load_labour_income_share(tmp_path, ["LAO"])
    warm = load_labour_income_share(tmp_path, ["LAO"])

    assert len(asked) == 1
    assert cold.equals(warm)


def test_a_changed_country_set_is_cached_apart(tmp_path, serves):
    """The frame depends on who was asked for, so one country set must not read back another's."""
    serves(served([("LAO", datetime(2010, 1, 1), 48.5)]))

    load_labour_income_share(tmp_path, ["LAO"])
    load_labour_income_share(tmp_path, ["LAO", "NPL"])

    assert len(list(tmp_path.glob("ilo_labour_share__*.parquet"))) == 2


def test_the_whole_series_is_asked_for(tmp_path, serves):
    """Left to itself the service answers with a handful of recent years rather than everything it
    holds, and a series that short is indistinguishable from one the ILO only recently began.
    """
    asked = serves(served([("LAO", datetime(2010, 1, 1), 48.5)]))

    load_labour_income_share(tmp_path, ["LAO"])

    _, _, kwargs = asked[0]
    assert kwargs["start"] <= FIRST_ILO_YEAR


def test_the_default_country_set_is_the_panel_s_own(tmp_path, serves):
    """Every caller takes the default, so the countries it resolves to are the ones this source is
    actually fetched for.
    """
    asked = serves(served([("LAO", datetime(2010, 1, 1), 48.5)]))

    load_labour_income_share(tmp_path)

    _, selections, _ = asked[0]
    assert set(selections["REF_AREA"]) == set(REQUESTED_COUNTRY_CODES) - UNCOVERED_AREAS


def test_an_answer_that_is_not_a_frame_is_rejected(tmp_path, monkeypatch):
    """kuznets honors output_type on its own, so a non-frame answer means the contract moved."""

    class FakeReader:
        def __init__(self, dataflow, selections=None, **kwargs):
            pass

        def read(self):
            return object()

    monkeypatch.setattr(ilo, "ILOSTATReader", FakeReader)

    with pytest.raises(TypeError, match="output_type='polars'"):
        load_labour_income_share(tmp_path, ["LAO"])
