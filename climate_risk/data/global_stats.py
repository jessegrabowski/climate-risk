"""
Reduce one year of a global field to annual statistics on its own native grid.

Each function reads one downloaded year, works through it in slabs so the whole year never has to sit
in memory, and returns a small ``xarray.Dataset`` with one value per pixel. The raw file can then be
deleted. Longitudes are always returned as -180..180, sorted, so the sources line up.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from climate_risk.data.covariate_table import gumbel_return_level

WINDOW_DAYS = 5
LAT_BAND = 120  # pixel rows handled at once for the daily ERA5 field
TIME_SLAB = 15  # days handled at once for the GloFAS field
BASELINE = (1991, 2020)


def _time_name(data: xr.Dataset | xr.DataArray) -> str:
    return next(name for name in ("valid_time", "time") if name in data.dims)


def wrap_longitude(data: xr.Dataset) -> xr.Dataset:
    """Return the data with longitudes in -180..180, sorted ascending."""
    lon = data["longitude"].to_numpy()
    if lon.max() > 180:
        data = data.assign_coords(longitude=((lon + 180) % 360) - 180)
    return data.sortby("longitude")


def sliding_max_sum(values: np.ndarray, window: int = WINDOW_DAYS) -> np.ndarray:
    """
    Largest sum over any ``window`` consecutive days along axis 0, per pixel.

    The window stays inside the array, so a window crossing the end of the year is not counted. A pixel
    with any missing day gives NaN.
    """
    cumulative = np.cumsum(values, axis=0, dtype="float64")
    padded = np.concatenate([np.zeros_like(cumulative[:1]), cumulative], axis=0)
    return np.asarray((padded[window:] - padded[:-window]).max(axis=0), dtype="float32")


def daily_precip_stats(data: xr.Dataset, variable: str, to_mm: float) -> xr.Dataset:
    """
    From one year of daily precipitation: ``precip_total`` and ``rx5day`` in mm.

    ``rx5day`` is the largest sum over five consecutive days inside the year. ``to_mm`` converts the file's
    unit to millimetres (1000 for ERA5 metres, 1 for CHIRPS). Negative values are treated as missing.
    The dataset must use ``latitude`` and ``longitude`` and one time-like dimension.
    """
    rain = data[variable]
    time = next(name for name in rain.dims if name not in ("latitude", "longitude"))
    total, rx5 = [], []
    for start in range(0, rain.sizes["latitude"], LAT_BAND):
        band = rain.isel(latitude=slice(start, start + LAT_BAND)).transpose(time, "latitude", "longitude").to_numpy()
        band = band.astype("float64") * to_mm
        band[band < 0] = np.nan
        total.append(band.sum(axis=0).astype("float32"))
        rx5.append(sliding_max_sum(band))
    out = xr.Dataset(
        {
            "precip_total": (("latitude", "longitude"), np.concatenate(total)),
            "rx5day": (("latitude", "longitude"), np.concatenate(rx5)),
        },
        coords={"latitude": rain["latitude"].to_numpy(), "longitude": rain["longitude"].to_numpy()},
    )
    out["precip_total"].attrs["units"] = "mm"
    out["rx5day"].attrs["units"] = "mm"
    return wrap_longitude(out)


def era5_daily_stats(path: Path) -> xr.Dataset:
    """ERA5 daily total precipitation (metres) for one year, reduced to ``precip_total`` and ``rx5day`` (mm)."""
    return daily_precip_stats(xr.open_dataset(path), "tp", 1000.0)


def chirps_global_stats(path: Path) -> xr.Dataset:
    """CHIRPS v2.0 daily precipitation, box-averaged to 0.25 degrees by the IRI, reduced the same way."""
    data = xr.open_dataset(path, decode_times=False).rename({"X": "longitude", "Y": "latitude"})
    return daily_precip_stats(data, "prcp", 1.0)


def era5_land_stats(path: Path) -> xr.Dataset:
    """
    From one year of ERA5-Land monthly means: soil moisture, temperature, precipitation and PET.

    ``tp`` and ``pev`` are mean daily accumulations in metres, so they are multiplied by the days in each
    month. ``pev`` is negative for evaporation in ERA5-Land; its sign is flipped so ``pet_annual`` is a
    positive number of millimetres.
    """
    data = xr.open_dataset(path)
    time = _time_name(data)
    days = xr.DataArray(data[time].dt.days_in_month.to_numpy(), dims=time)
    out = xr.Dataset(
        {
            "soilm_mean": data["swvl1"].mean(time),
            "soilm_max_month": data["swvl1"].max(time),
            "t2m_mean": data["t2m"].mean(time) - 273.15,
            "precip_land": (data["tp"] * days).sum(time, min_count=12) * 1000.0,
            "pet_annual": -(data["pev"] * days).sum(time, min_count=12) * 1000.0,
        }
    )
    out = out.drop_vars([name for name in out.coords if name not in out.dims])
    return wrap_longitude(out)


def glofas_annual_max(path: Path) -> xr.Dataset:
    """Largest daily discharge in the year per pixel (m3/s), read in time slabs."""
    data = xr.open_dataset(path)
    time = _time_name(data)
    running: np.ndarray | None = None
    for start in range(0, data.sizes[time], TIME_SLAB):
        slab = data["avg_dis"].isel({time: slice(start, start + TIME_SLAB)}).to_numpy()
        slab_max = np.fmax.reduce(slab, axis=0)
        running = slab_max if running is None else np.fmax(running, slab_max)
    out = xr.Dataset(
        {"q_annual_max": (("latitude", "longitude"), np.asarray(running, dtype="float32"))},
        coords={"latitude": data["latitude"], "longitude": data["longitude"]},
    )
    return wrap_longitude(out)


def glofas_days_over(path: Path, level: xr.DataArray) -> xr.Dataset:
    """Number of days in the year with discharge above each pixel's return ``level``."""
    data = xr.open_dataset(path)
    time = _time_name(data)
    threshold = wrap_longitude(level.to_dataset(name="level"))["level"]
    limit = threshold.reindex(latitude=data["latitude"], longitude=data["longitude"], method="nearest").to_numpy()
    # the raw file is 0..360 only if the source says so; GloFAS is already -180..180, so no shift is needed
    count = np.zeros(limit.shape, dtype="int16")
    for start in range(0, data.sizes[time], TIME_SLAB):
        slab = data["avg_dis"].isel({time: slice(start, start + TIME_SLAB)}).to_numpy()
        with np.errstate(invalid="ignore"):
            count += (slab > limit).sum(axis=0, dtype="int16")
    out = xr.Dataset(
        {"q_days_over_rp5": (("latitude", "longitude"), count)},
        coords={"latitude": data["latitude"], "longitude": data["longitude"]},
    )
    return wrap_longitude(out)


def return_level_from_maxima(maxima: xr.DataArray, period: float = 5.0, band: int = 200) -> xr.DataArray:
    """
    Gumbel return level per pixel from a (year, latitude, longitude) stack of annual maxima.

    Fitted by L-moments, in latitude bands to keep memory small. Pixels with fewer than ten valid years,
    or where the maxima do not vary, are NaN.
    """
    values = maxima.transpose("year", "latitude", "longitude").to_numpy()
    out = np.full(values.shape[1:], np.nan, dtype="float32")
    for start in range(0, values.shape[1], band):
        block = values[:, start : start + band, :]
        valid = np.isfinite(block).sum(axis=0) >= 10
        level = gumbel_return_level(np.where(np.isfinite(block), block, np.nan).reshape(block.shape[0], -1), period)
        out[start : start + band, :] = np.where(valid, level.reshape(block.shape[1:]), np.nan)
    return xr.DataArray(
        out, coords={"latitude": maxima["latitude"], "longitude": maxima["longitude"]}, dims=("latitude", "longitude")
    )


def standardise_stack(stack: xr.DataArray, baseline: tuple[int, int] = BASELINE) -> xr.DataArray:
    """Z-score a (year, latitude, longitude) stack against each pixel's baseline-period mean and spread."""
    inside = stack.sel(year=slice(*baseline))
    return (stack - inside.mean("year")) / inside.std("year", ddof=1)
