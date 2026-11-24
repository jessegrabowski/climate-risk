"""
CHIRPS v3 daily precipitation (the ``rnl`` product), clipped to a box as it is read.

Each day is a global GeoTIFF of about 13 MB, stored one row per block, so reading only the rows of
a box over HTTP moves a small fraction of the file. Daily CHIRPS is not an independent daily
product: it is the CHIRPS pentad total shared out over the days by ERA5 (``rnl``), so the timing
inside a pentad comes from ERA5 and only the pentad totals come from CHIRPS.
"""

import logging
import time
import urllib.parse

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
import xarray as xr

from rasterio.windows import from_bounds

_log = logging.getLogger(__name__)

CHIRPS_ROOT = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0"
CHIRPS_README = f"{CHIRPS_ROOT}/README-CHIRPSv3.0.txt"
DAILY_RNL_ROOT = f"{CHIRPS_ROOT}/daily/final/rnl"
CHIRPS_LICENCE = "CHIRPS is public domain, free to use with citation (Funk et al. 2015; CHIRPS v3, UCSB CHC)."

USER_AGENT = "climate-risk (+https://github.com/jessegrabowski/laos-climate-change)"
ATTEMPTS = 8
FIRST_YEAR = 1981
DAYS_PER_PENTAD = 5
PENTADS_PER_MONTH = 6

Box = tuple[float, float, float, float]  # west, south, east, north


def daily_url(date: pd.Timestamp) -> str:
    """URL of the global daily rnl file for one date."""
    return f"{DAILY_RNL_ROOT}/{date.year}/chirps-v3.0.rnl.{date:%Y.%m.%d}.tif"


PENTAD_ROOT = f"{CHIRPS_ROOT}/pentads/global/cogs"


def pentad_url(year: int, month: int, pentad: int) -> str:
    """URL of the global pentad COG (pentad 1 to 6 within the month)."""
    return f"{PENTAD_ROOT}/chirps-v3.0.{year}.{month:02d}.{pentad}.cog"


def pentad_year_path(out_dir: Path, year: int) -> Path:
    """Where the clipped pentad file for one year lives."""
    return out_dir / f"chirps_v3_pentad_{year}.nc"


def year_path(out_dir: Path, year: int) -> Path:
    """Where the clipped daily file for one year lives."""
    return out_dir / f"chirps_v3_rnl_daily_{year}.nc"


def _read_url(url: str, label: str, box: Box) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    west, south, east, north = box
    last_error: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            with rasterio.Env(GDAL_HTTP_USERAGENT=USER_AGENT, CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
                with rasterio.open(f"/vsicurl/{url}") as source:
                    window = from_bounds(west, south, east, north, source.transform).round_offsets().round_lengths()
                    values = source.read(1, window=window).astype("float32")
                    transform = source.window_transform(window)
            rows, columns = values.shape
            longitudes = transform.c + transform.a * (np.arange(columns) + 0.5)
            latitudes = transform.f + transform.e * (np.arange(rows) + 0.5)
            return values, longitudes, latitudes
        except Exception as error:  # network and GDAL errors both surface here; retried then re-raised
            last_error = error
            time.sleep(min(5.0 * 2.0**attempt, 120.0))  # a 429 means slow down
    raise RuntimeError(f"CHIRPS {label} failed after {ATTEMPTS} attempts: {last_error}")


def download_year(year: int, box: Box, out_dir: Path, *, workers: int = 8) -> Path:
    """
    Download one year of daily rnl CHIRPS clipped to ``box``, skipping a year already on disk.

    Returns the NetCDF path. The file is written under a temporary name and moved into place at the
    end, so an interrupted year is redone rather than mistaken for finished.
    """
    target = year_path(out_dir, year)
    if target.exists():
        return target

    out_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        days = list(pool.map(lambda date: _read_url(daily_url(date), f"{date:%Y-%m-%d}", box), dates))

    values = np.stack([day[0] for day in days])
    values[values < 0] = np.nan  # fill value for no data is negative
    array = xr.DataArray(
        values,
        coords={"time": dates, "lat": days[0][2], "lon": days[0][1]},
        dims=("time", "lat", "lon"),
        name="precip",
        attrs={"units": "mm/day", "source": "CHIRPS v3.0 final, daily rnl"},
    )
    partial = target.with_suffix(".nc.part")
    array.to_dataset().to_netcdf(partial, encoding={"precip": {"zlib": True, "complevel": 4}})
    partial.replace(target)
    return target


def _read_boxes(url: str, label: str, boxes: dict[str, Box]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Read one window per box from one file, opening (and so fetching its header) once."""
    last_error: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            out = {}
            with rasterio.Env(GDAL_HTTP_USERAGENT=USER_AGENT, CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.cog"):
                with rasterio.open(f"/vsicurl/{url}") as source:
                    for name, (west, south, east, north) in boxes.items():
                        window = from_bounds(west, south, east, north, source.transform).round_offsets().round_lengths()
                        values = source.read(1, window=window).astype("float32")
                        transform = source.window_transform(window)
                        rows, columns = values.shape
                        out[name] = (
                            values,
                            transform.c + transform.a * (np.arange(columns) + 0.5),
                            transform.f + transform.e * (np.arange(rows) + 0.5),
                        )
            return out
        except Exception as error:  # network and GDAL errors both surface here; retried then re-raised
            last_error = error
            time.sleep(min(5.0 * 2.0**attempt, 120.0))  # a 429 means slow down
    raise RuntimeError(f"CHIRPS {label} failed after {ATTEMPTS} attempts: {last_error}")


def download_pentad_year(year: int, boxes: dict[str, Box], out_root: Path, *, workers: int = 2) -> list[Path]:
    """
    Download one year of CHIRPS v3 pentad totals (72 fields) for several countries at once.

    ``boxes`` maps an ISO3 code to its box. Each pentad COG is opened once and every country's
    window read from it, so adding a country adds a few tiles per file, not a request. Output goes to
    ``out_root/<ISO3>/chirps/``; countries whose year is already on disk are skipped. The time axis is
    the first day of each pentad.
    """
    targets = {iso: pentad_year_path(out_root / iso / "chirps", year) for iso in boxes}
    todo = {iso: box for iso, box in boxes.items() if not targets[iso].exists()}
    if todo:
        keys = [(month, pentad) for month in range(1, 13) for pentad in range(1, PENTADS_PER_MONTH + 1)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fields = list(
                pool.map(lambda key: _read_boxes(pentad_url(year, *key), f"{year}-{key[0]:02d}-p{key[1]}", todo), keys)
            )
        starts = pd.to_datetime([f"{year}-{m:02d}-{1 + DAYS_PER_PENTAD * (p - 1):02d}" for m, p in keys])
        for iso in todo:
            array = xr.DataArray(
                np.stack([field[iso][0] for field in fields]),
                coords={"time": starts, "lat": fields[0][iso][2], "lon": fields[0][iso][1]},
                dims=("time", "lat", "lon"),
                name="precip",
                attrs={"units": "mm/pentad", "source": "CHIRPS v3.0 final, pentad"},
            )
            array = array.where(array >= 0)  # the no-data fill is negative
            targets[iso].parent.mkdir(parents=True, exist_ok=True)
            partial = targets[iso].with_suffix(".nc.part")
            array.to_dataset().to_netcdf(partial, encoding={"precip": {"zlib": True, "complevel": 4}})
            partial.replace(targets[iso])
    return list(targets.values())


def pentad_annual(pentads: xr.DataArray) -> xr.Dataset:
    """Annual total and maximum pentad total per pixel from one year of pentad fields."""
    return xr.Dataset({"precip_total": pentads.sum("time", min_count=72), "rx_pentad": pentads.max("time")})


def pentad_totals(daily: xr.DataArray) -> xr.DataArray:
    """
    Rebuild CHIRPS pentad totals from the daily field, six per month.

    Pentads are days 1-5, 6-10, 11-15, 16-20, 21-25 and 26 to month end, as CHIRPS defines them, and
    the daily values sum to the pentad total by construction of the rnl product.
    """
    time_index = pd.DatetimeIndex(daily["time"].to_numpy())
    pentad = np.minimum((time_index.day - 1) // DAYS_PER_PENTAD, PENTADS_PER_MONTH - 1)
    label = time_index.year * 10000 + time_index.month * 100 + pentad
    labelled = daily.assign_coords(pentad_id=("time", label))
    return labelled.groupby("pentad_id").sum(min_count=1)


def annual_extremes(daily: xr.DataArray) -> xr.Dataset:
    """
    Annual total, maximum 5-day total and maximum pentad total, per pixel, for one calendar year.

    The sliding 5-day window stays inside the year, so a window straddling 31 December is not counted.
    """
    return xr.Dataset(
        {
            "precip_total": daily.sum("time", min_count=1),
            "rx5day": daily.rolling(time=DAYS_PER_PENTAD, min_periods=DAYS_PER_PENTAD).sum().max("time"),
            "rx_pentad": pentad_totals(daily).max("pentad_id"),
        }
    )


# --- CHIRPS v2.0 daily "improved" via the IRI Data Library -------------------------------------
# The UCSB host throttled and then blocked long runs, so the daily series is also available from
# Columbia's IRI Data Library, which cuts a box out on the server and sends only that. This is
# CHIRPS version 2.0 (0.05 degrees, from 1981), not v3.0: the pentad totals differ between versions.

IRI_ROOT = "https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p05/.prcp"
IRI_LICENCE = "CHIRPS v2.0, UCSB CHC, public domain with citation (Funk et al. 2015); served by the IRI Data Library."


def iri_year_url(year: int, box: Box) -> str:
    """Server-side subset of one calendar year of daily CHIRPS v2 to ``box``, as NetCDF."""
    west, south, east, north = box
    start, end = urllib.parse.quote(f"(1 Jan {year})"), urllib.parse.quote(f"(31 Dec {year})")
    return f"{IRI_ROOT}/X/{west}/{east}/RANGE/Y/{south}/{north}/RANGE/T/{start}/{end}/RANGE/data.nc"


def iri_year_path(out_dir: Path, year: int) -> Path:
    """Where the clipped CHIRPS v2 daily file for one year lives."""
    return out_dir / f"chirps_v2_daily_{year}.nc"


def download_iri_year(year: int, box: Box, out_dir: Path) -> Path:
    """
    Download one year of daily CHIRPS v2 for ``box`` from the IRI Data Library, skipping a year on disk.

    The file is checked to hold exactly one field per day of the year before it is kept, and its
    time axis is rewritten as calendar dates (IRI stores Julian days).
    """
    target = iri_year_path(out_dir, year)
    if target.exists():
        return target

    out_dir.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".nc.raw")
    last_error: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            with requests.get(
                iri_year_url(year, box), headers={"User-Agent": USER_AGENT}, stream=True, timeout=300
            ) as r:
                r.raise_for_status()
                with open(partial, "wb") as handle:
                    for block in r.iter_content(1 << 20):
                        handle.write(block)
            break
        except requests.RequestException as error:
            last_error = error
            time.sleep(min(5.0 * 2.0**attempt, 120.0))
    else:
        raise RuntimeError(f"CHIRPS v2 {year} failed after {ATTEMPTS} attempts: {last_error}")

    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    with xr.open_dataset(partial, decode_times=False) as raw:
        if raw.sizes["T"] != len(dates):
            raise RuntimeError(f"CHIRPS v2 {year}: got {raw.sizes['T']} days, expected {len(dates)}")
        julian_first = float(raw["T"].values[0])
        expected = (dates[0] - pd.Timestamp("1970-01-01")).days + 2440587.5
        if abs(julian_first - expected) > 1.0:
            raise RuntimeError(f"CHIRPS v2 {year}: time axis starts at JD {julian_first}, expected {expected}")
        array = xr.DataArray(
            raw["prcp"].to_numpy().astype("float32"),
            coords={"time": dates, "lat": raw["Y"].to_numpy(), "lon": raw["X"].to_numpy()},
            dims=("time", "lat", "lon"),
            name="precip",
            attrs={"units": "mm/day", "source": "CHIRPS v2.0 daily improved via IRI Data Library"},
        )
    array = array.where(array >= 0)  # the no-data fill is negative
    array.to_dataset().to_netcdf(target.with_suffix(".nc.part"), encoding={"precip": {"zlib": True, "complevel": 4}})
    target.with_suffix(".nc.part").replace(target)
    partial.unlink()
    return target


def iri_global_url(year: int, step: float = 0.25) -> str:
    """
    One calendar year of daily CHIRPS v2 for the whole 50N-50S belt, averaged to ``step`` degrees on the server.

    Box-averaging on the IRI side sends about 0.85 GB a year instead of about 21 GB at the native 0.05 degrees.
    The statistics built from it (rx5day of a 0.25-degree box average) are therefore coarser than native ones.
    """
    start, end = urllib.parse.quote(f"(1 Jan {year})"), urllib.parse.quote(f"(31 Dec {year})")
    return f"{IRI_ROOT}/T/{start}/{end}/RANGE/X/{step}/boxAverage/Y/{step}/boxAverage/data.nc"


def download_iri_global_year(year: int, target: Path, step: float = 0.25) -> Path:
    """Download one global box-averaged CHIRPS v2 year to ``target`` (skipped if it exists), checking the day count."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".nc.part")
    last_error: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            with requests.get(
                iri_global_url(year, step), headers={"User-Agent": USER_AGENT}, stream=True, timeout=600
            ) as r:
                r.raise_for_status()
                with open(partial, "wb") as handle:
                    for block in r.iter_content(1 << 20):
                        handle.write(block)
            with xr.open_dataset(partial, decode_times=False) as check:
                expected = 366 if pd.Timestamp(f"{year}-12-31").dayofyear == 366 else 365
                if check.sizes["T"] != expected:
                    raise RuntimeError(f"CHIRPS v2 {year}: got {check.sizes['T']} days, expected {expected}")
            partial.replace(target)
            return target
        except (requests.RequestException, RuntimeError, OSError) as error:
            last_error = error
            time.sleep(min(5.0 * 2.0**attempt, 120.0))
    raise RuntimeError(f"CHIRPS v2 global {year} failed after {ATTEMPTS} attempts: {last_error}")
