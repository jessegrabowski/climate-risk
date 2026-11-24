"""
Turn the global yearly covariate grids into a country-year table (and an admin-1 year table).

For every country (GADM outline) and every year it reports, for each statistic:
  <name>_mean     area-weighted mean over the country (valid pixels only)
  <name>_max      largest pixel value inside the country
  <name>_popmean  population-weighted mean (GHS-POP, nearest epoch), for the ERA5 and ERA5-Land fields only

Run after global_boundaries.py and global_annual_stats.py:
    pixi run python experiments/covariates/global_country_year.py
Output: data/processed/covariates/global/global_country_year.parquet and global_admin1_year.parquet
"""

import logging

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from climate_risk.data.covariate_table import PolygonReducer
from climate_risk.data.global_population import nearest_epoch, population_on_lattice

OUT = Path("data/processed/covariates/global")
CACHE = Path("data")
YEARS = range(1980, 2026)

# source file stem -> (variables, weight by population?)
SOURCES = {
    "era5_daily": (["precip_total", "rx5day"], True),
    "era5_anom": (["precip_anom_std"], True),
    "chirps_v2": (["precip_total", "rx5day"], False),  # CHIRPS v2.0, IRI mirror, 0.25-degree box average, 50N-50S
    "era5land": (["soilm_mean", "soilm_max_month", "t2m_mean", "precip_land", "pet_annual"], True),
    "glofas": (["q_annual_max"], False),
    "glofas_days": (["q_days_over_rp5"], False),
}

# column names that would otherwise clash between sources
RENAME = {
    ("era5_daily", "precip_total"): "precip_era5",
    ("chirps_v2", "precip_total"): "chirps_precip",
    ("chirps_v2", "rx5day"): "chirps_rx5day",
}

_log = logging.getLogger("global_country_year")


def population(grid_key: str, epoch: int, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    cache = OUT / f"population_{grid_key}_{epoch}.npy"
    if not cache.exists():
        np.save(cache, population_on_lattice(epoch, lon, lat, CACHE))
    return np.load(cache)


def reduce_source(stem: str, units: gpd.GeoDataFrame, key_columns: list[str], tag: str) -> pd.DataFrame:
    variables, weighted = SOURCES[stem]
    reducer = None
    rows = []
    for year in YEARS:
        path = OUT / f"{stem}_{year}.nc"
        if not path.exists():
            continue
        data = xr.open_dataset(path).load()
        lon, lat = data["longitude"].to_numpy(), data["latitude"].to_numpy()
        if reducer is None:
            reducer = PolygonReducer(units.geometry, lon, lat)
        frame = units[key_columns].copy()
        frame["year"] = year
        for variable in variables:
            name = RENAME.get((stem, variable), variable)
            field = data[variable].to_numpy()
            frame[f"{name}_mean"] = reducer.mean(field)
            frame[f"{name}_max"] = reducer.maximum(field)
            if weighted:
                weights = population(stem if stem != "era5_anom" else "era5_daily", nearest_epoch(year), lon, lat)
                frame[f"{name}_popmean"] = reducer.weighted_mean(field, weights)
        rows.append(frame)
        _log.info("%s %s %d", tag, stem, year)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[*key_columns, "year"])


def build(level: str, key_columns: list[str]) -> pd.DataFrame:
    units = gpd.read_file(OUT / f"boundaries_{level}.gpkg").reset_index(drop=True)
    merged = None
    for stem in SOURCES:
        part = reduce_source(stem, units, key_columns, level)
        merged = part if merged is None else merged.merge(part, on=[*key_columns, "year"], how="outer")
    return merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build("country", ["GID_0", "NAME_0"]).to_parquet(OUT / "global_country_year.parquet")
    build("admin1", ["GID_0", "GID_1", "NAME_1"]).to_parquet(OUT / "global_admin1_year.parquet")
