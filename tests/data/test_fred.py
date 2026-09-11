from datetime import date

import polars as pl
import pytest

from polars.testing import assert_frame_equal

from climate_risk.data import fred
from climate_risk.data.fred import SERIES_NAMES, _load_series, load_fred_data, transform_fred

ONE_SERIES = {"DTB3": "world_rate_3m"}


def downloaded(series_id: str, rows) -> pl.DataFrame:
    """One series shaped as kuznets returns it, with the date under FRED's own column name."""
    frame = pl.DataFrame(rows, schema=["DATE", series_id], orient="row")

    return frame.with_columns(pl.col("DATE").str.to_datetime("%Y-%m-%d"))


@pytest.fixture
def serves(monkeypatch):
    """Answer FredReader from prepared frames instead of the network, recording what was asked for."""

    def serve(frames: dict[str, pl.DataFrame]):
        requested = []

        class FakeReader:
            def __init__(self, symbols, **kwargs):
                requested.append((symbols, kwargs))
                self._symbols = symbols

            def read(self):
                return frames[self._symbols]

        monkeypatch.setattr(fred.fred, "FredReader", FakeReader)
        return requested

    return serve


def test_series_are_stacked_under_their_readable_names():
    frames = {"DTB3": downloaded("DTB3", [("2020-01-01", 1.5)])}

    frame = transform_fred(frames, SERIES_NAMES)

    assert frame.columns == ["series", "date", "value"]
    assert frame.rows() == [("world_rate_3m", date(2020, 1, 1), 1.5)]


def test_series_at_different_frequencies_each_keep_every_observation():
    """A daily rate and a monthly index share almost no dates, so neither may lose an observation to
    the other's calendar.
    """
    frames = {
        "DTB3": downloaded("DTB3", [("2020-01-01", 1.5), ("2020-01-02", 1.6)]),
        "CPIAUCSL": downloaded("CPIAUCSL", [("2020-01-01", 258.0)]),
    }

    frame = transform_fred(frames, SERIES_NAMES)

    assert frame.filter(pl.col("series") == "world_rate_3m")["value"].to_list() == [1.5, 1.6]
    assert frame.filter(pl.col("series") == "world_price_level")["value"].to_list() == [258.0]


def test_observations_fred_reports_as_missing_are_dropped():
    """FRED sends a null on non-trading days for the daily series; they are not observations."""
    frames = {"DTB3": downloaded("DTB3", [("2020-01-01", 1.5), ("2020-01-02", None)])}

    frame = transform_fred(frames, SERIES_NAMES)

    assert frame["date"].to_list() == [date(2020, 1, 1)]


def test_the_result_is_sorted_by_series_and_date():
    frames = {
        "DTB3": downloaded("DTB3", [("2020-01-02", 1.6), ("2020-01-01", 1.5)]),
        "CPIAUCSL": downloaded("CPIAUCSL", [("2020-01-01", 258.0)]),
    }

    frame = transform_fred(frames, SERIES_NAMES)

    assert frame["series"].to_list() == ["world_price_level", "world_rate_3m", "world_rate_3m"]
    assert frame["date"].to_list() == [date(2020, 1, 1), date(2020, 1, 1), date(2020, 1, 2)]


def test_a_series_fred_does_not_recognize_is_rejected(tmp_path, serves):
    """An unknown ID comes back empty rather than raising, so an unchecked typo would reach the panel
    as a series that is simply absent — and a model would then be estimated without it.
    """
    empty = pl.DataFrame(schema={"DATE": pl.Datetime, "DTB3": pl.Float64})
    serves({"DTB3": empty})

    with pytest.raises(ValueError, match="DTB3"):
        _load_series(tmp_path, "fred", ONE_SERIES, force_reload=False)


def test_a_backend_other_than_polars_is_rejected(tmp_path, monkeypatch):
    """The transform is polars-only, so a frame from another backend must fail here, not in a select."""

    class FakeReader:
        def __init__(self, symbols, **kwargs):
            pass

        def read(self):
            return object()

    monkeypatch.setattr(fred.fred, "FredReader", FakeReader)

    with pytest.raises(TypeError, match="output_type='polars'"):
        load_fred_data(tmp_path)


def test_a_warm_cache_does_not_download(tmp_path, serves):
    """The download is one request per series; a present cache must not trigger it again."""
    requested = serves({"DTB3": downloaded("DTB3", [("2020-01-01", 1.5)])})

    _load_series(tmp_path, "fred", ONE_SERIES, force_reload=False)
    _load_series(tmp_path, "fred", ONE_SERIES, force_reload=False)

    assert [symbols for symbols, _ in requested] == ["DTB3"]


def test_the_cold_run_writes_the_cache_it_will_read(tmp_path, serves):
    """A key spelled one way on write and another on read leaves a file nothing ever reads back."""
    serves({"DTB3": downloaded("DTB3", [("2020-01-01", 1.5)])})

    _load_series(tmp_path, "fred", ONE_SERIES, force_reload=False)

    assert len(list(tmp_path.glob("fred__*.parquet"))) == 1


def test_the_cold_and_warm_frames_agree(tmp_path, serves):
    serves({"DTB3": downloaded("DTB3", [("2020-01-01", 1.5)])})

    cold = _load_series(tmp_path, "fred", ONE_SERIES, force_reload=False)
    warm = _load_series(tmp_path, "fred", ONE_SERIES, force_reload=False)

    assert_frame_equal(cold, warm)


def test_forcing_a_reload_downloads_again(tmp_path, serves):
    requested = serves({"DTB3": downloaded("DTB3", [("2020-01-01", 1.5)])})

    _load_series(tmp_path, "fred", ONE_SERIES, force_reload=False)
    _load_series(tmp_path, "fred", ONE_SERIES, force_reload=True)

    assert len(requested) == 2


def test_the_download_reaches_back_before_any_series_starts(tmp_path, serves):
    """A later start would silently shorten whichever series has the longest history."""
    requested = serves({"DTB3": downloaded("DTB3", [("2020-01-01", 1.5)])})

    _load_series(tmp_path, "fred", ONE_SERIES, force_reload=False)

    _, arguments = requested[0]
    assert arguments["start"] == "1900-01-01"


def test_no_two_series_share_a_readable_name():
    """The names key the tidy panel, so a repeated one would interleave two series into one."""
    names = list(SERIES_NAMES.values())

    assert len(names) == len(set(names))


def test_every_configured_series_is_downloaded(tmp_path, serves):
    """The other loader tests serve a single series, so none of them would notice the public entry
    point asking for a narrowed set.
    """
    requested = serves({sid: downloaded(sid, [("2020-01-01", 1.0)]) for sid in SERIES_NAMES})

    load_fred_data(tmp_path)

    assert [symbols for symbols, _ in requested] == list(SERIES_NAMES)


def test_a_series_returned_without_its_own_column_is_named():
    """The value column is named for the series ID. Without this the select raises a polars error
    naming the column, which reads as an internal fault rather than a withdrawn series.
    """
    headerless = downloaded("DTB3", [("2020-01-01", 1.5)]).drop("DTB3")

    with pytest.raises(ValueError, match="DTB3"):
        transform_fred({"DTB3": headerless}, SERIES_NAMES)


def test_changing_the_series_set_turns_the_cache_over(tmp_path, serves):
    """Cache parameters record what was asked for, not how it was built, so without a fingerprint a
    panel built under an older series list reads back as a hit and silently omits the new series.
    """
    both = {sid: downloaded(sid, [("2020-01-01", 1.0)]) for sid in ("DTB3", "DGS10")}
    serves(both)

    one = _load_series(tmp_path, "fred", {"DTB3": "world_rate_3m"}, force_reload=False)
    two = _load_series(tmp_path, "fred", {"DTB3": "world_rate_3m", "DGS10": "world_rate_10y"}, force_reload=False)

    assert one["series"].unique().to_list() == ["world_rate_3m"]
    assert set(two["series"].unique().to_list()) == {"world_rate_3m", "world_rate_10y"}
