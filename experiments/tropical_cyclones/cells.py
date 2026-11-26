"""
Step 1 of the cyclone build (pixi environment): the grid cells winds are needed on.

A cell of the ERA5 0.25-degree lattice is kept if GHS-POP puts anyone in it in any epoch. Unpopulated cells have zero
weight in every population-weighted figure, so computing their wind would change nothing. Latitudes beyond 61 degrees
(CLIMADA's default limit for the wind model) are left out and count as never hit.

Usage: ``pixi run python experiments/tropical_cyclones/cells.py`` -> data/processed/covariates/tc_cells.npz
"""

from pathlib import Path

import numpy as np
import xarray as xr

GLOBAL = Path("data/processed/covariates/global")
TARGET = Path("data/processed/covariates/tc_cells.npz")
MAX_LATITUDE = 61.0

template = xr.open_dataset(GLOBAL / "era5_daily_2020.nc")
lon, lat = template["longitude"].to_numpy(), template["latitude"].to_numpy()
populated = np.zeros((len(lat), len(lon)), dtype=bool)
for path in sorted(GLOBAL.glob("population_era5_daily_*.npy")):
    weights = np.load(path)
    assert weights.shape == populated.shape, f"{path.name}: {weights.shape} is not (latitude, longitude)"
    populated |= weights > 0
populated &= np.abs(lat)[:, None] <= MAX_LATITUDE
rows, columns = np.nonzero(populated)
np.savez(TARGET, row=rows, column=columns, latitude=lat, longitude=lon)
print(f"{len(rows)} populated cells of {populated.size}", flush=True)
