"""
Step 3 of the cyclone build (pixi environment): wind exposure per country-year and per ``sea`` ADM2 atom-year.

Each year's winds go back onto the ERA5 lattice with three fields: the annual maximum wind, and 0/1 flags for cells
at or above 17.5 m/s (34 knots) and 33 m/s (64 knots). The existing resampler then takes GHS-POP weighted means:
    tc_wind_popmean   population-weighted mean of the annual maximum wind (m/s)
    tc_share_34kt     share of the population in cells with an annual maximum of at least 17.5 m/s
    tc_share_64kt     the same for 33 m/s
Countries never hit get 0.

Usage: ``pixi run python experiments/tropical_cyclones/exposure.py``
Output: data/processed/covariates/global/tc_<year>.nc, data/processed/covariates/tc_country_year.parquet and
        tc_atom_year_sea.parquet
"""

import logging

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from climate_risk.config.registry import load_place, resolve_isos
from climate_risk.data.gadm import load_units_in_country
from climate_risk.data.resample import atom_year_table

CACHE = Path("data")
COVARIATES = CACHE / "processed" / "covariates"
GLOBAL = COVARIATES / "global"
YEARS = range(1980, 2026)
THRESHOLDS = {"tc_over_34kt": 17.5, "tc_over_64kt": 33.0}
NAMES = {
    "tc_wind_popmean": "tc_wind_popmean",
    "tc_over_34kt_popmean": "tc_share_34kt",
    "tc_over_64kt_popmean": "tc_share_64kt",
}
_log = logging.getLogger("tc_exposure")


def write_lattice() -> None:
    """One small NetCDF per year on the ERA5 lattice."""
    cells = np.load(COVARIATES / "tc_cells.npz")
    for year in YEARS:
        wind = np.zeros((len(cells["latitude"]), len(cells["longitude"])), dtype=np.float32)
        wind[cells["row"], cells["column"]] = np.load(COVARIATES / "tc_wind" / f"{year}.npz")["wind"]
        fields = {"tc_wind": wind}
        fields |= {name: (wind >= limit).astype(np.float32) for name, limit in THRESHOLDS.items()}
        dataset = xr.Dataset(
            {name: (("latitude", "longitude"), values) for name, values in fields.items()},
            coords={"latitude": cells["latitude"], "longitude": cells["longitude"]},
            attrs={"source": "IBTrACS v04r01 + CLIMADA TropCyclone H08", "year": str(year)},
        )
        dataset.to_netcdf(GLOBAL / f"tc_{year}.nc", encoding={n: {"zlib": True, "complevel": 4} for n in fields})


def exposure(shapes: gpd.GeoSeries, ids: pd.Series, id_column: str) -> pd.DataFrame:
    """Population-weighted exposure of every shape and year."""
    table = atom_year_table(shapes, ids, YEARS, GLOBAL, cache_dir=CACHE, sources=["tc"], id_column=id_column)
    out = table[[id_column, "year", *NAMES]].rename(columns=NAMES)
    # A shape the weights do not reach (north of 61 degrees only, or no population) was never hit.
    missing = out["tc_share_34kt"].isna()
    _log.info("%d shape-years without a populated cell, set to 0", int(missing.sum()))
    return out.fillna({name: 0.0 for name in NAMES.values()})


def main() -> None:
    write_lattice()
    countries = gpd.read_file(GLOBAL / "boundaries_country.gpkg").reset_index(drop=True)
    exposure(countries.geometry, countries["GID_0"], "ISO").to_parquet(COVARIATES / "tc_country_year.parquet")
    isos = resolve_isos(load_place("sea"))
    atoms = pd.concat([load_units_in_country(iso, 2, CACHE) for iso in isos], ignore_index=True)
    exposure(gpd.GeoSeries(atoms.geometry), atoms["gid"], "atom_id").to_parquet(COVARIATES / "tc_atom_year_sea.parquet")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
