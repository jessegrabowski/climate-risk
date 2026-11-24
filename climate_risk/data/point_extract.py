"""
Read yearly global covariate grids at points (event locations, background points).

Each point takes the value of the pixel that contains it, in the year it is given. Points outside a
source's coverage (CHIRPS north of 50N, GloFAS south of 60S) get NaN.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# source file stem -> variables
POINT_SOURCES = {
    "era5_daily": ["precip_total", "rx5day"],
    "era5_anom": ["precip_anom_std"],
    "chirps_v2": ["precip_total", "rx5day"],
    "era5land": ["soilm_mean", "soilm_max_month", "t2m_mean", "precip_land", "pet_annual"],
    "glofas": ["q_annual_max"],
    "glofas_days": ["q_days_over_rp5"],
}
RENAME = {
    ("era5_daily", "precip_total"): "precip_era5",
    ("chirps_v2", "precip_total"): "chirps_precip",
    ("chirps_v2", "rx5day"): "chirps_rx5day",
}


def values_at_points(points: pd.DataFrame, directory: Path) -> pd.DataFrame:
    """
    Add one column per statistic to ``points``, which needs columns ``longitude``, ``latitude`` and ``year``.

    A pixel is used only if the point lies within its half-width of the pixel centre, so a point outside a
    source's coverage gets NaN instead of the nearest edge pixel.
    """
    result = points.copy()
    for stem, variables in POINT_SOURCES.items():
        for variable in variables:
            result[RENAME.get((stem, variable), variable)] = np.nan
    for year, rows in points.groupby("year"):
        for stem, variables in POINT_SOURCES.items():
            path = directory / f"{stem}_{int(year)}.nc"
            if not path.exists():
                continue
            data = xr.open_dataset(path)
            step_lon = float(abs(np.median(np.diff(data["longitude"].to_numpy()))))
            step_lat = float(abs(np.median(np.diff(data["latitude"].to_numpy()))))
            picked = data.sel(
                longitude=xr.DataArray(rows["longitude"].to_numpy(), dims="p"),
                latitude=xr.DataArray(rows["latitude"].to_numpy(), dims="p"),
                method="nearest",
            )
            inside = (np.abs(picked["longitude"].to_numpy() - rows["longitude"].to_numpy()) <= step_lon / 2 + 1e-9) & (
                np.abs(picked["latitude"].to_numpy() - rows["latitude"].to_numpy()) <= step_lat / 2 + 1e-9
            )
            for variable in (name for name in variables if name in data.data_vars):
                name = RENAME.get((stem, variable), variable)
                result.loc[rows.index, name] = np.where(inside, picked[variable].to_numpy(), np.nan)
    return result
