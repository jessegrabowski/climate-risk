import logging

from collections.abc import Mapping
from pathlib import Path

import polars as pl

from kuznets import fred

from climate_risk.data.cache import builder_fingerprint, cached, polars_parquet
from climate_risk.data.source import ApiSource

_log = logging.getLogger(__name__)

FRED = ApiSource(
    url="https://api.stlouisfed.org/fred/series/observations",
    license=(
        "Series carry the terms of whoever published them. The US federal series read here are "
        "public domain; other series on FRED are redistributed under restrictions, some serving only "
        "a rolling window of history."
    ),
    citation="Federal Reserve Bank of St. Louis, FRED (Federal Reserve Economic Data), https://fred.stlouisfed.org/.",
    retrieved="2026-09-05",
)

# The foreign block the small open economy model treats as exogenous: a world interest rate, a world
# price level, and world prices for the goods it imports. Every one is a world aggregate, so this
# list is the same whichever country is being estimated. Country-specific series come from the World
# Bank panel, which is keyed by ISO code.
SERIES_NAMES = {
    "DTB3": "world_rate_3m",
    "DGS10": "world_rate_10y",
    "CPIAUCSL": "world_price_level",
    "PALLFNFINDEXM": "commodity_all",
    "PNRGINDEXM": "commodity_energy",
    "PFOODINDEXM": "commodity_food",
}

# Earlier than any series' coverage, so each one starts wherever its own history does.
FIRST_DATE = "1900-01-01"


def transform_fred(frames: Mapping[str, pl.DataFrame], series_names: Mapping[str, str]) -> pl.DataFrame:
    """
    Stack one frame per series into a tidy panel keyed by series name and date.

    The panel is long because the series carry mixed frequencies: daily interest rates alongside
    monthly price indices.

    Parameters
    ----------
    frames : mapping of str to DataFrame
        One frame per FRED series ID, each with a ``DATE`` column and a column named for the ID.
    series_names : mapping of str to str
        The series IDs to keep, each mapped to the name it is stored under.

    Returns
    -------
    observations : DataFrame
        Columns ``series``, ``date`` and ``value``, sorted, with observations FRED reports as
        missing dropped.
    """
    missing = sorted(series_id for series_id, frame in frames.items() if series_id not in frame.columns)
    if missing:
        raise ValueError(f"FRED returned no column for {missing}; the series may have been withdrawn.")

    stacked = []
    for series_id, frame in frames.items():
        name = series_names[series_id]
        stacked.append(
            frame.select(
                pl.lit(name).alias("series"),
                pl.col("DATE").cast(pl.Date).alias("date"),
                pl.col(series_id).cast(pl.Float64).alias("value"),
            ).drop_nulls("value")
        )

    return pl.concat(stacked).sort("series", "date")


def _load_series(
    cache_dir: Path,
    name: str,
    series_names: Mapping[str, str],
    *,
    force_reload: bool,
) -> pl.DataFrame:
    """Download each series in ``series_names`` and cache the stacked panel under ``name``."""

    def build() -> pl.DataFrame:
        frames = {}
        for series_id in series_names:
            _log.info(f"Downloading FRED series {series_id}")
            downloaded = fred.FredReader(series_id, start=FIRST_DATE, end=None, output_type="polars").read()
            if not isinstance(downloaded, pl.DataFrame):
                raise TypeError(f"kuznets returned a {type(downloaded).__name__} for output_type='polars'")

            # A series FRED no longer publishes comes back as an empty frame rather than an error, so
            # an unchecked one would reach the panel as a series that is simply absent.
            if downloaded.is_empty():
                raise ValueError(f"FRED returned no observations for {series_id}")

            frames[series_id] = downloaded

        return transform_fred(frames, series_names)

    return cached(
        cache_dir,
        name,
        build,
        polars_parquet(),
        params={"reading": builder_fingerprint(build, series_names)},
        force=force_reload,
    )


def load_fred_data(cache_dir: Path, *, force_reload: bool = False) -> pl.DataFrame:
    """
    Return the foreign block as a tidy panel of series, date and value, at native frequency.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.fred import load_fred_data

        series = load_fred_data(Path("data"))
    """
    return _load_series(cache_dir, "fred", SERIES_NAMES, force_reload=force_reload)
