"""NOAA CPC Relative Oceanic Nino Index (RONI): 3-month seasonal values, one series for the world."""

import io

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RONI_URL = "https://cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt"
RONI_TABLE_URL = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni"
RONI_LICENCE = "NOAA/NWS public domain (US Government work)."
USER_AGENT = "climate-risk (+https://github.com/jessegrabowski/laos-climate-change)"

# A season is named by its three months and takes the middle one as its centre.
SEASON_CENTRE_MONTH = {
    season: month
    for month, season in enumerate(
        ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"], start=1
    )
}


def download_roni(out_dir: Path) -> Path:
    """
    Save the RONI file with today's date in its name, or return the copy already saved today.

    Recent values are revised for up to two months, so the access date is part of the record.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"RONI.ascii.{date.today():%Y-%m-%d}.txt"
    if not target.exists():
        response = requests.get(RONI_URL, headers={"User-Agent": USER_AGENT}, timeout=60)
        response.raise_for_status()
        target.write_text(response.text)
    return target


def read_roni(path: Path) -> pd.DataFrame:
    """Read the seasonal file into one row per month with ``year``, ``month`` and ``roni``."""
    table = pd.read_csv(io.StringIO(path.read_text()), sep=r"\s+")
    table["month"] = table["SEAS"].map(SEASON_CENTRE_MONTH)
    # CPC labels a season by the year of its centre month (DJF 1950 is centred on January 1950, NDJ 1950
    # on December 1950), so the year needs no shift.
    return table.rename(columns={"YR": "year", "ANOM": "roni"})[["year", "month", "roni"]]


def annual_roni(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    One row per calendar year with ``roni_annual_mean`` and ``roni_peak``.

    The mean is over the twelve seasons centred in the year; the peak is the seasonal value
    furthest from zero, keeping its sign. Years missing a season are dropped, so a part-year is never
    read as a quiet one.
    """
    by_year = monthly.groupby("year")["roni"]
    strongest = by_year.apply(lambda values: values.iloc[int(np.argmax(values.abs().to_numpy()))])
    frame = pd.DataFrame({"roni_annual_mean": by_year.mean(), "roni_peak": strongest, "seasons": by_year.size()})
    return frame[frame["seasons"] == 12].drop(columns="seasons").reset_index()
