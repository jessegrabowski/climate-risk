"""
Build ``data/processed/covariates/<ISO3>_cell_year.parquet`` from the raw downloads.

    pixi run python experiments/covariates/build_cell_year_table.py            # every country
    pixi run python experiments/covariates/build_cell_year_table.py NPL ZMB

One row per model cell and calendar year (1981-2025). Needs that country's downloads finished.
"""

import logging
import sys

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from common import COUNTRIES, FIRST_YEAR, LAST_YEAR, PROCESSED, country_boundary, raw_dir

from climate_risk.data import chirps, roni
from climate_risk.data.covariate_table import (
    CellReducer,
    gumbel_return_level,
    long_frame,
    month_lengths,
    standardise_against_baseline,
)
from climate_risk.geo.raster import build_cell_grid

_log = logging.getLogger("build_cell_year_table")

RESOLUTION_KM = 30.0
YEARS = np.arange(FIRST_YEAR, LAST_YEAR + 1)
METRES_TO_MM = 1000.0
KELVIN = 273.15


def _coords(dataset: xr.Dataset) -> tuple[str, str, str]:
    time = "valid_time" if "valid_time" in dataset.coords else "time"
    lat = "latitude" if "latitude" in dataset.coords else "lat"
    lon = "longitude" if "longitude" in dataset.coords else "lon"
    return time, lat, lon


def open_monthly(folder: Path) -> xr.Dataset:
    """All decade files of one monthly source, joined along time."""
    parts = [xr.open_dataset(path) for path in sorted(folder.glob("*.nc"))]
    time = _coords(parts[0])[0]
    return xr.concat([p.drop_vars(["number", "expver"], errors="ignore") for p in parts], dim=time).sortby(time)


def monthly_to_annual(dataset: xr.Dataset, variable: str, reducer: CellReducer, *, kind: str) -> np.ndarray:
    """
    Cell values per year from monthly means.

    ``kind='total'`` turns a mean daily accumulation into an annual total (sum of mean x days in month);
    ``kind='mean'`` is the day-weighted mean of the months; ``kind='max_month'`` the largest monthly value.
    """
    time, lat_name, lon_name = _coords(dataset)
    cell_monthly = reducer.mean(dataset[variable].transpose(time, lat_name, lon_name).to_numpy())  # (month, cell)
    stamps = pd.DatetimeIndex(dataset[time].to_numpy())
    days = month_lengths(stamps)
    out = np.full((len(YEARS), cell_monthly.shape[1]), np.nan)
    for i, year in enumerate(YEARS):
        rows = np.flatnonzero(stamps.year == year)
        if len(rows) != 12:
            continue  # a part-year is not an annual value
        block, weight = cell_monthly[rows], days[rows][:, None]
        out[i] = {
            "total": (block * weight).sum(0),
            "mean": (block * weight).sum(0) / weight.sum(),
            "max_month": block.max(0),
        }[kind]
    return out


def main(iso: str) -> None:
    RAW = raw_dir(iso)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    grid = build_cell_grid(country_boundary(iso), resolution_km=RESOLUTION_KM)
    cell_ids = grid.cells["cell_id"].to_numpy()
    _log.info("%s grid: %d cells at %.0f km", iso, len(cell_ids), RESOLUTION_KM)
    columns: dict[str, np.ndarray] = {}

    # --- CHIRPS: per-pixel annual statistics, then the cell mean
    reducer = None
    rx5, rxp, total = [], [], []
    for year in YEARS:
        with xr.open_dataarray(chirps.iri_year_path(RAW / "chirps", int(year))) as daily:
            daily = daily.load()
        if reducer is None:
            reducer = CellReducer(grid, daily["lon"].to_numpy(), daily["lat"].to_numpy())
        stats = chirps.annual_extremes(daily)
        rx5.append(reducer.mean(stats["rx5day"].to_numpy()))
        rxp.append(reducer.mean(stats["rx_pentad"].to_numpy()))
        total.append(reducer.mean(stats["precip_total"].to_numpy()))
    columns["rx5day"], columns["rx_pentad"] = np.array(rx5), np.array(rxp)
    columns["precip_anom_std"] = standardise_against_baseline(np.array(total), YEARS)
    columns["precip_chirps_annual"] = np.array(total)

    # --- ERA5 total precipitation (cross-check)
    if any((RAW / "era5").glob("*.nc")):
        era5 = open_monthly(RAW / "era5")
        _, lat, lon = _coords(era5)
        reducer = CellReducer(grid, era5[lon].to_numpy(), era5[lat].to_numpy())
        columns["precip_era5"] = monthly_to_annual(era5, "tp", reducer, kind="total") * METRES_TO_MM
    else:
        _log.warning("%s: no ERA5 files, so precip_era5 is left out", iso)

    # --- ERA5-Land
    land = open_monthly(RAW / "era5land")
    _, lat, lon = _coords(land)
    reducer = CellReducer(grid, land[lon].to_numpy(), land[lat].to_numpy())
    columns["soilm_mean"] = monthly_to_annual(land, "swvl1", reducer, kind="mean")
    columns["soilm_max_month"] = monthly_to_annual(land, "swvl1", reducer, kind="max_month")
    columns["t2m_mean"] = monthly_to_annual(land, "t2m", reducer, kind="mean") - KELVIN
    # ERA5-Land counts evaporation as negative, so the sign is flipped to give a positive demand in mm.
    columns["pet_annual"] = -monthly_to_annual(land, "pev", reducer, kind="total") * METRES_TO_MM

    # --- GloFAS: daily cell maximum, annual maximum, Gumbel return level fitted to 1980-2025
    files = sorted((RAW / "glofas").glob("glofas_v5_*_*.nc"))
    level = None
    have = {int(f.stem.split("_")[2]) for f in files}
    if have >= set(range(1980, LAST_YEAR + 1)):
        annual_max: dict[int, np.ndarray] = {}
        daily_by_year: dict[int, np.ndarray] = {}
        reducer = None
        for path in files:
            with xr.open_dataset(path) as dataset:
                time, lat, lon = _coords(dataset)
                if reducer is None:
                    reducer = CellReducer(grid, dataset[lon].to_numpy(), dataset[lat].to_numpy())
                variable = next(name for name in dataset.data_vars if name in ("avg_dis", "dis24"))
                # A daily mean is stamped with the end of its day, so the day it describes is one earlier.
                dates = pd.DatetimeIndex(dataset[time].to_numpy()) - pd.Timedelta(days=1)
                cell_daily = reducer.maximum(dataset[variable].to_numpy())  # (day, cell)
            for year in np.unique(dates.year):
                daily_by_year[int(year)] = cell_daily[dates.year == year]
        for year, block in daily_by_year.items():
            annual_max[year] = np.nanmax(block, axis=0)
        fit_years = sorted(y for y in annual_max if 1980 <= y <= 2025 and len(daily_by_year[y]) >= 365)
        level = gumbel_return_level(np.array([annual_max[y] for y in fit_years]))
        columns["q_annual_max"] = np.array([annual_max.get(int(y), np.full(len(cell_ids), np.nan)) for y in YEARS])
        columns["q_days_over_rp5"] = np.array(
            [
                (daily_by_year[int(y)] > level).sum(0).astype(float)
                if int(y) in daily_by_year
                else np.full(len(cell_ids), np.nan)
                for y in YEARS
            ]
        )
    else:
        _log.warning("%s: GloFAS years missing (%d of 46), so q_* columns are left out", iso, len(have))

    # --- assemble
    table = None
    for name, values in columns.items():
        frame = long_frame(values, YEARS, cell_ids, name)
        table = frame if table is None else table.merge(frame, on=["cell_id", "year"])
    monthly = roni.read_roni(next((raw_dir("NPL") / "roni").glob("RONI.ascii.*.txt")))
    table = table.merge(roni.annual_roni(monthly), on="year", how="left")
    table = table.merge(grid.cells[["cell_id", "lon", "lat"]], on="cell_id").assign(iso3=iso)
    table.to_parquet(PROCESSED / f"{iso}_cell_year.parquet", index=False)
    if level is not None:
        pd.DataFrame({"cell_id": cell_ids, "q_rp5_level": level}).to_parquet(
            PROCESSED / f"{iso}_q_rp5_levels.parquet", index=False
        )
    _log.info("Wrote %d rows x %d columns", *table.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for code in sys.argv[1:] or COUNTRIES:
        main(code)
