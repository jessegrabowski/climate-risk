"""
ERA5, ERA5-Land and GloFAS downloads from the Copernicus CDS and EWDS, clipped to a box.

Every request is checked against the dataset's own form (its ``form.json`` in the catalogue) rather
than guessed. Requests queue on the server, so each download skips files already on disk and a
request that fails twice stops the source with the error text.
"""

import logging
import re

from pathlib import Path

import cdsapi

_log = logging.getLogger(__name__)

CDS_URL = "https://cds.climate.copernicus.eu/api"
EWDS_URL = "https://ewds.climate.copernicus.eu/api"
RC_FILE = Path("~/.cdsapirc").expanduser()
ATTEMPTS = 2

MONTHS = [f"{month:02d}" for month in range(1, 13)]
DAYS = [f"{day:02d}" for day in range(1, 32)]

ERA5_LAND = "reanalysis-era5-land-monthly-means"
ERA5_SINGLE = "reanalysis-era5-single-levels-monthly-means"
GLOFAS = "cems-glofas-historical"
ERA5_DAILY = "derived-era5-single-levels-daily-statistics"

ERA5_LAND_VARIABLES = [
    "volumetric_soil_water_layer_1",
    "2m_temperature",
    "total_precipitation",
    "potential_evaporation",
]
ERA5_VARIABLES = ["total_precipitation"]


class SourceFailed(RuntimeError):
    """A request failed on every attempt; the message carries the server's error text."""


def read_key(rc_file: Path = RC_FILE) -> str:
    """Read the API key from ``~/.cdsapirc``. The key is never logged or printed."""
    match = re.search(r"^key:\s*(\S+)", rc_file.read_text(), flags=re.MULTILINE)
    if match is None:
        raise SourceFailed(f"{rc_file} holds no 'key:' line.")
    return match.group(1)


def make_client(url: str) -> cdsapi.Client:
    """A client for the CDS or the EWDS, built from the one key in ``~/.cdsapirc``."""
    return cdsapi.Client(url=url, key=read_key(), quiet=True)


def cds_area(box: tuple[float, float, float, float]) -> list[float]:
    """Convert (west, south, east, north) to the CDS order [north, west, south, east]."""
    west, south, east, north = box
    return [north, west, south, east]


def _scrub(text: str) -> str:
    return text.replace(read_key(), "<key>")


def retrieve(client: cdsapi.Client, dataset: str, request: dict, target: Path) -> Path:
    """
    Download one request to ``target`` unless it is already there.

    Raises
    ------
    SourceFailed
        After two failed attempts, carrying the error text with the key removed.
    """
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    errors: list[str] = []
    for attempt in range(ATTEMPTS):
        try:
            partial.unlink(missing_ok=True)
            client.retrieve(dataset, request, str(partial))
            partial.replace(target)
            return target
        except Exception as error:  # the client raises many types; report them all the same way
            errors.append(_scrub(str(error)))
            _log.warning("%s %s attempt %d failed: %s", dataset, target.name, attempt + 1, errors[-1])
    raise SourceFailed(f"{dataset} -> {target.name} failed {ATTEMPTS} times. Last error: {errors[-1]}")


def era5_land_request(years: list[int], box: tuple[float, float, float, float]) -> dict:
    """Monthly means of the four ERA5-Land variables for the given years."""
    return {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ERA5_LAND_VARIABLES,
        "year": [str(year) for year in years],
        "month": MONTHS,
        "time": ["00:00"],
        "area": cds_area(box),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def era5_request(years: list[int], box: tuple[float, float, float, float]) -> dict:
    """Monthly mean total precipitation from ERA5 single levels, the CHIRPS cross-check."""
    return {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ERA5_VARIABLES,
        "year": [str(year) for year in years],
        "month": MONTHS,
        "time": ["00:00"],
        "area": cds_area(box),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def glofas_request(years: list[int], box: tuple[float, float, float, float]) -> dict:
    """
    Daily river discharge from GloFAS historical v5.0 for the given years, in the form's own key names.

    Only ``version_5_0`` is requested. The caller must not fall back to v4.0 if it is refused.
    """
    return {
        "system_version": ["version_5_0"],
        "hydrological_model": ["lisflood"],
        "product_type": ["consolidated"],
        "variable": ["average_river_discharge_in_the_last_24_hours"],
        "timespan": ["time_mean"],
        "year": [str(year) for year in years],
        "month": MONTHS,
        "day": DAYS,
        "area": cds_area(box),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def era5_daily_request(year: int) -> dict:
    """Global daily total precipitation (sum over the UTC day) for one year, from the ERA5 daily statistics."""
    return {
        "product_type": "reanalysis",
        "variable": ["total_precipitation"],
        "year": str(year),
        "month": MONTHS,
        "day": DAYS,
        "daily_statistic": "daily_sum",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def global_request(request: dict) -> dict:
    """The same request without an ``area`` key, which means the whole globe."""
    return {key: value for key, value in request.items() if key != "area"}
