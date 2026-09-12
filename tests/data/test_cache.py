import logging

from dataclasses import replace
from datetime import date

import geopandas as gpd
import pandas as pd
import polars as pl
import pytest

from geopandas.testing import assert_geodataframe_equal
from polars.testing import assert_frame_equal
from shapely.geometry import Point

from climate_risk.data.cache import (
    builder_fingerprint,
    cache_key,
    cached,
    geo_parquet,
    pandas_parquet,
    polars_parquet,
)


@pytest.fixture
def frame():
    return pd.DataFrame({"Date": ["1990-01-01", "1991-01-01"], "co2": [354.4, 355.6]}).set_index("Date")


class CountingBuilder:
    """A builder that records how often it ran, so a cache miss is observable."""

    def __init__(self, frame):
        self.frame = frame
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.frame


@pytest.fixture
def counting_builder(frame):
    return CountingBuilder(frame)


def test_the_builder_runs_once_across_repeated_calls(tmp_path, counting_builder):
    """The bug this replaces recomputed silently every call, for forty minutes at a time."""
    cached(tmp_path, "co2", counting_builder, pandas_parquet())
    result = cached(tmp_path, "co2", counting_builder, pandas_parquet())

    assert counting_builder.calls == 1
    assert result.index.tolist() == ["1990-01-01", "1991-01-01"]


def test_force_rebuilds(tmp_path, counting_builder):
    cached(tmp_path, "co2", counting_builder, pandas_parquet())
    cached(tmp_path, "co2", counting_builder, pandas_parquet(), force=True)

    assert counting_builder.calls == 2


def test_different_parameters_do_not_share_an_entry(tmp_path, frame):
    """Two hardcoded filenames for one parameterised cache is what this replaces."""
    cached(tmp_path, "rivers", lambda: frame, pandas_parquet(), params={"ord_flow_lt": 5})
    cached(tmp_path, "rivers", lambda: frame.head(1), pandas_parquet(), params={"ord_flow_lt": 6})

    assert len(cached(tmp_path, "rivers", lambda: frame, pandas_parquet(), params={"ord_flow_lt": 5})) == 2
    assert len(cached(tmp_path, "rivers", lambda: frame, pandas_parquet(), params={"ord_flow_lt": 6})) == 1


def test_the_written_and_read_frames_agree(tmp_path, frame):
    """A reader that does not restore the index is the cold/warm divergence this pairing prevents."""
    written = cached(tmp_path, "co2", lambda: frame, pandas_parquet())
    reloaded = cached(tmp_path, "co2", lambda: frame, pandas_parquet())

    pd.testing.assert_frame_equal(written, reloaded)


def test_a_geodataframe_round_trips_without_losing_anything(tmp_path):
    """A shapefile truncates field names to ten characters, which diverges cold from warm."""
    points = gpd.GeoDataFrame({"distance_to_river_km": [4.5], "geometry": [Point(102.5, 18.5)]}, crs="EPSG:4326")

    written = cached(tmp_path, "points", lambda: points, geo_parquet())
    reloaded = cached(tmp_path, "points", lambda: points, geo_parquet())

    assert_geodataframe_equal(reloaded, written)


def test_parameters_are_ordered_so_the_key_is_stable():
    """Callers write kwargs in whatever order reads well; the same set must find the same file."""
    assert cache_key("points", {"region": "laos", "grid_size": 400}) == cache_key(
        "points", {"grid_size": 400, "region": "laos"}
    )


def test_the_key_is_readable():
    assert cache_key("points", {"region": "laos", "grid_size": 400}) == "points__grid_size=400__region=laos"


def test_a_bare_name_needs_no_parameters():
    assert cache_key("co2") == "co2"


@pytest.mark.parametrize(
    "params",
    [
        {"region": "la/os"},
        {"region": "la__os"},
        {"region": "la=os"},
        {"gr/id": 4},
        {"version": "4.1"},
    ],
    ids=["slash", "parameter-separator", "value-separator", "in-the-name", "dot"],
)
def test_a_value_that_would_corrupt_the_filename_is_rejected(params):
    """A separator in a value makes two different parameter sets collide on one cache entry. So does
    a dot: the artifact's suffix is taken from the last one, so `4.1` and `4.2` name the same file."""
    with pytest.raises(ValueError, match="must not contain"):
        cache_key("points", params)


@pytest.mark.parametrize("value", [1.5, None, ["a"], object()], ids=["float", "none", "list", "object"])
def test_a_value_that_does_not_format_stably_is_rejected(value):
    """`str()` of a float or an object is not a stable identity to key a cache on."""
    with pytest.raises(ValueError, match="string, integer or boolean"):
        cache_key("points", {"size": value})


def test_a_hyphenated_value_is_allowed():
    """Region and indicator names carry hyphens; only the separators are off limits."""
    assert cache_key("points", {"region": "south-east-asia"}) == "points__region=south-east-asia"


def test_a_write_killed_partway_does_not_poison_the_cache(tmp_path, frame):
    """A truncated file left at the destination would read back as a hit forever after."""

    def write_half_then_die(artifact, path):
        path.write_bytes(b"PAR1 truncated")
        raise OSError("no space left on device")

    dying = replace(pandas_parquet(), write=write_half_then_die)

    with pytest.raises(OSError, match="no space left"):
        cached(tmp_path, "co2", lambda: frame, dying)

    assert not (tmp_path / "co2.parquet").exists()
    assert len(cached(tmp_path, "co2", lambda: frame, pandas_parquet())) == 2


def test_an_empty_name_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        cache_key("")


def test_a_polars_frame_round_trips_without_an_index(tmp_path):
    """A tidy frame carries its columns in the file, so nothing has to be restored on read."""
    frame = pl.DataFrame({"year": [1990, 1991], "co2": [354.4, 355.6]})

    cached(tmp_path, "co2", lambda: frame, polars_parquet())
    reloaded = cached(tmp_path, "co2", lambda: frame, polars_parquet())

    assert_frame_equal(reloaded, frame)


def test_a_polars_date_column_keeps_its_type(tmp_path):
    """A date returning as text is how the cold and warm paths drift apart; parquet carries the dtype."""
    frame = pl.DataFrame({"Date": [date(1990, 1, 1)], "co2": [354.4]})

    cached(tmp_path, "co2", lambda: frame, polars_parquet())

    assert cached(tmp_path, "co2", lambda: frame, polars_parquet()).schema["Date"] == pl.Date


def test_two_builders_named_alike_but_reading_differently_fingerprint_differently():
    """Every builder in this package is called `build`, so a fingerprint taken from the name would
    be one value shared by every cached artifact. What has to reach the key is the rule that
    changed, which lives only in the body."""

    def builder(keep_every_row: bool):
        if keep_every_row:

            def build() -> str:
                return "every row"
        else:

            def build() -> str:
                return "only the rows that are named"

        return build

    assert builder_fingerprint(builder(True)) != builder_fingerprint(builder(False))


def test_the_same_builder_fingerprints_the_same_every_time():
    """A digest that moved between runs would rebuild every artifact on every process start."""

    def build() -> str:
        return "unchanged"

    assert builder_fingerprint(build) == builder_fingerprint(build)


def test_a_builder_closing_over_different_values_fingerprints_the_same():
    """The closure's values are what `params` is for. Folding them in here would key one entry per
    country twice over, and every cached artifact would miss."""

    def builder_for(iso: str):
        def build() -> str:
            return iso

        return build

    assert builder_fingerprint(builder_for("LAO")) == builder_fingerprint(builder_for("ZMB"))


def test_two_copies_of_one_function_fingerprint_the_same():
    """The loaders pass their transform as a rule, and a function's repr carries its address, so a
    rule read for its repr keys an entry the next process cannot find. Building the function twice
    is how a fresh process differs from this one: same source, different address.
    """

    def build() -> str:
        return "unchanged"

    def make_transform():
        def transform(frame):
            return frame

        return transform

    first, second = make_transform(), make_transform()

    assert builder_fingerprint(build, first) == builder_fingerprint(build, second)


def test_editing_a_function_rule_fingerprints_differently():
    """Reading the function for its source is only worth doing if the source is what is read."""

    def build() -> str:
        return "unchanged"

    def transform(frame):
        return frame

    def transform_differently(frame):
        return frame.head()

    assert builder_fingerprint(build, transform) != builder_fingerprint(build, transform_differently)


def test_a_rule_whose_repr_carries_an_address_is_refused():
    """Python's default repr carries the object's address, so an entry keyed on one is written and
    never read back. Refusing is the only way that surfaces, since a miss just rebuilds.
    """

    def build() -> str:
        return "unchanged"

    class Rule:
        pass

    with pytest.raises(ValueError, match="object address"):
        builder_fingerprint(build, Rule())


def test_a_builder_reading_a_different_table_fingerprints_differently():
    """A builder's source does not show the module tables it consults, so changing one changes the
    artifact with nothing in the key to say so."""

    def build() -> str:
        return "unchanged"

    assert builder_fingerprint(build, {"ETH": ("ERI",)}) != builder_fingerprint(build, {})


def test_a_cached_read_says_nothing_unless_the_caller_asked(caplog, tmp_path, frame):
    """One call can touch a dozen artefacts, and a dozen lines of cache chatter buries whatever the
    caller meant to show."""
    with caplog.at_level(logging.INFO, logger="climate_risk"):
        cached(tmp_path, "co2", lambda: frame, pandas_parquet())
        cached(tmp_path, "co2", lambda: frame, pandas_parquet())

    assert caplog.records == []


@pytest.mark.parametrize(("warm", "expected"), [(False, "Building co2"), (True, "Loading cached co2")], ids=str)
def test_a_verbose_read_says_whether_it_built_or_loaded(caplog, tmp_path, frame, warm, expected):
    """Which branch a loader took is the one thing the log is for; a message that cannot tell a
    build from a cache hit would not be worth emitting."""
    if warm:
        cached(tmp_path, "co2", lambda: frame, pandas_parquet())

    with caplog.at_level(logging.INFO, logger="climate_risk"):
        cached(tmp_path, "co2", lambda: frame, pandas_parquet(), verbose=True)

    (record,) = caplog.records
    assert record.message.startswith(expected)
    assert record.filename == "cache.py", "the record points at the loader, not the logging helper"
