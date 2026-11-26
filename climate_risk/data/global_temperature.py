"""Global mean surface temperature: the HadCRUT5 annual global series."""

from pathlib import Path

import polars as pl
import xarray as xr

from climate_risk.data.fetch import fetch
from climate_risk.data.source import DataSource

HADCRUT_GLOBAL_ANNUAL = DataSource(
    url=(
        "https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.0.2.0/analysis/diagnostics/"
        "HadCRUT.5.0.2.0.analysis.summary_series.global.annual.nc"
    ),
    filename="HadCRUT.5.0.2.0.analysis.summary_series.global.annual.nc",
    license="Open Government License v3",
    citation=(
        "HadCRUT.5.0.2.0 global annual series, obtained from https://www.metoffice.gov.uk/hadobs/hadcrut5 and "
        "© British Crown Copyright, Met Office, provided under an Open Government License."
    ),
    retrieved="2026-11-26",
)


def transform_global_temperature(raw: xr.Dataset) -> pl.DataFrame:
    """
    Keep the year and the best-estimate anomaly.

    Parameters
    ----------
    raw : Dataset
        The published file, with ``tas_mean`` over ``time`` (one step per year).

    Returns
    -------
    series : DataFrame
        ``year`` and ``gmst`` (degrees C against 1961-1990, not detrended), sorted by year. The last year of a file
        built part-way through it covers only the months published by then.
    """
    return pl.DataFrame(
        {"year": raw["time"].dt.year.to_numpy().astype("int64"), "gmst": raw["tas_mean"].to_numpy().astype("float64")}
    ).sort("year")


def load_global_temperature(cache_dir: Path, *, force_reload: bool = False) -> pl.DataFrame:
    """
    Load the HadCRUT5 global annual temperature anomaly.

    Parameters
    ----------
    cache_dir : Path
        Directory the source caches live under.
    force_reload : bool, optional
        Download again. Default False.

    Returns
    -------
    series : DataFrame
        ``year`` and ``gmst``.
    """
    with xr.open_dataset(fetch(HADCRUT_GLOBAL_ANNUAL, cache_dir, force=force_reload)) as raw:
        return transform_global_temperature(raw)
