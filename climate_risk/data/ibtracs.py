"""
IBTrACS tropical-cyclone best tracks (declaration only).

The file is read by CLIMADA in its own environment (``experiments/tropical_cyclones/windfields.py``), which turns the
tracks into wind fields. This module records where it comes from and fetches it into the directory that script reads.
"""

from pathlib import Path

from climate_risk.data.fetch import fetch
from climate_risk.data.source import DataSource

IBTRACS = DataSource(
    url=(
        "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
        "v04r01/access/netcdf/IBTrACS.ALL.v04r01.nc"
    ),
    filename="IBTrACS.ALL.v04r01.nc",
    license="NOAA: the data may be redistributed and used without restriction (the file's own licence attribute).",
    citation=(
        "Knapp, K. R. et al. (2010), The International Best Track Archive for Climate Stewardship (IBTrACS). "
        "Bulletin of the American Meteorological Society 91, 363-376. https://doi.org/10.1175/2009BAMS2755.1. "
        "Data: NOAA National Centers for Environmental Information, version v04r01, https://doi.org/10.25921/82ty-9e16"
    ),
    retrieved="2026-11-26",
)


def download_ibtracs(cache_dir: Path, *, force_reload: bool = False) -> Path:
    """
    Download the all-basins NetCDF file unless it is already there.

    Parameters
    ----------
    cache_dir : Path
        The repo data directory; the file goes to ``raw/covariates/ibtracs`` under it.
    force_reload : bool, optional
        Download again. Default False.

    Returns
    -------
    path : Path
        The downloaded file.
    """
    return fetch(IBTRACS, cache_dir / "raw" / "covariates" / "ibtracs", force=force_reload)
