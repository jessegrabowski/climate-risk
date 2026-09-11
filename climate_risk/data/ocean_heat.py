from pathlib import Path

import polars as pl

from climate_risk.data.cache import builder_fingerprint, cached, polars_parquet
from climate_risk.data.fetch import fetch
from climate_risk.data.source import DataSource

OCEAN_HEAT = DataSource(
    url=(
        "https://www.ncei.noaa.gov/data/oceans/woa/DATA_ANALYSIS/3M_HEAT_CONTENT/DATA"
        "/basin/3month/ohc_levitus_climdash_seasonal.csv"
    ),
    filename="ohc_levitus_climdash_seasonal.csv",
    license="public domain (U.S. Government work, 17 U.S.C. 105)",
    citation=(
        "NOAA National Centers for Environmental Information, global ocean heat content. "
        "https://doi.org/10.7289/V53F4MVP"
    ),
    retrieved="2026-08-05",
)

# Shifts the NCEI anomalies onto the baseline the published results were estimated against. The
# derivation is unrecorded; changing it invalidates every downstream number.
OCEAN_HEAT_BASELINE_OFFSET = 152


def transform_ocean_heat(seasonal: pl.DataFrame) -> pl.DataFrame:
    """
    Average seasonal ocean-heat anomalies to calendar years and shift them onto the project baseline.

    Parameters
    ----------
    seasonal : DataFrame
        Anomalies with a ``Date`` column of ``YYYY-M`` text, whose month upstream leaves unpadded
        below October, and a ``Temp`` column.

    Returns
    -------
    ocean_heat : DataFrame
        Annual means dated to each year's first day, offset by ``OCEAN_HEAT_BASELINE_OFFSET``.
    """
    # Only the year survives the average, so the year is all that is read.
    year = pl.col("Date").str.split("-").list.first().cast(pl.Int32)

    return (
        seasonal.group_by(year.alias("year"))
        .agg(pl.col("Temp").mean())
        .select(pl.date(pl.col("year"), 1, 1).alias("Date"), pl.col("Temp") + OCEAN_HEAT_BASELINE_OFFSET)
        .sort("Date")
    )


def load_ocean_heat_data(cache_dir: Path, *, force_reload: bool = False) -> pl.DataFrame:
    """
    Load the NOAA/NCEI global ocean heat content record, averaged to one value a year.

    Parameters
    ----------
    cache_dir : Path
        Directory the source caches live under.
    force_reload : bool, optional
        Download again and rebuild the cache rather than reading it. Default False.

    Returns
    -------
    ocean_heat : DataFrame
        One row per year, with a ``Date`` column and a ``Temp`` column.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk import load_ocean_heat_data

        ocean_heat = load_ocean_heat_data(Path("data"))
    """

    def build() -> pl.DataFrame:
        raw = fetch(OCEAN_HEAT, cache_dir, force=force_reload)

        # NCEI serves the seasonal file without a header, so reading one consumes the first season.
        return transform_ocean_heat(pl.read_csv(raw, has_header=False, new_columns=["Date", "Temp"]))

    reading = builder_fingerprint(build, transform_ocean_heat, OCEAN_HEAT_BASELINE_OFFSET)

    return cached(cache_dir, "ocean_heat", build, polars_parquet(), params={"reading": reading}, force=force_reload)
