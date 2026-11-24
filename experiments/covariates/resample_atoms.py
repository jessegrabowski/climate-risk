"""
Run the resampler: check it against the country table on the Laos 10 km lattice, then build ADM2 tables.

1. Laos 10 km lattice (the grid of notebooks/geo_damage_model/12_laos_emdat.ipynb): the yearly area-weighted mean
   over all cells must match the LAO row of global_country_year.parquet to within 1% (see TOLERANCE).
2. GADM level-2 units for LAO, NPL, CRI, ZMB, MAR and BFA -> data/processed/covariates/adm2_year_<ISO>.parquet.

Usage: ``pixi run python experiments/covariates/resample_atoms.py [check|adm2]``  (default: both)
"""

import logging
import sys

from pathlib import Path

import numpy as np
import pandas as pd

from climate_risk.data.gadm import load_units_in_country
from climate_risk.data.resample import atom_year_table
from climate_risk.geo.raster import build_cell_grid, dissolve_place_boundary

CACHE = Path("data")
GLOBAL = CACHE / "processed" / "covariates" / "global"
COUNTRY_TABLE = CACHE / "processed" / "covariates" / "global_country_year.parquet"
YEARS = range(1981, 2026)
ISOS = ("LAO", "NPL", "CRI", "ZMB", "MAR", "BFA")
RESOLUTION_KM = 10.0
TOLERANCE = 0.01  # relative; anomaly columns centred on 0 use an absolute 0.01 instead

_log = logging.getLogger("resample_atoms")


def check_laos() -> pd.DataFrame:
    """Lattice mean (cells weighted by their area inside Laos) against the LAO row of the country table."""
    units = load_units_in_country("LAO", 1, CACHE)
    boundary = dissolve_place_boundary(units[["geometry"]], iso3="LAO")
    grid = build_cell_grid(boundary, resolution_km=RESOLUTION_KM)
    cells = grid.cells.reset_index(drop=True)
    _log.info("Laos lattice: %d cells", len(cells))
    table = atom_year_table(
        grid.footprints.intersection(grid.place), cells["cell_id"], YEARS, GLOBAL, cache_dir=CACHE, id_column="cell_id"
    )

    area = cells.set_index("cell_id")["place_area_km2"]
    table["weight"] = table["cell_id"].map(area)
    # Only the area-weighted columns: a population-weighted lattice mean would need each cell's population, which
    # this check does not carry, so a cell-weighted average of popmeans is not comparable to the country popmean.
    value_columns = [c for c in table.columns if c.endswith("_mean")]
    lattice = (
        table.assign(**{c: table[c] * table["weight"] for c in value_columns})
        .groupby("year")[[*value_columns, "weight"]]
        .sum(min_count=1)
    )
    valid_weight = {
        c: table.assign(w=table["weight"].where(table[c].notna()))["w"].groupby(table["year"]).sum()
        for c in value_columns
    }
    lattice = pd.DataFrame({c: lattice[c] / valid_weight[c] for c in value_columns})

    country = pd.read_parquet(COUNTRY_TABLE).query("ISO == 'LAO'").set_index("year")
    rows = []
    for column in value_columns:
        if column not in country.columns:
            continue
        a, b = lattice[column], country[column].reindex(lattice.index)
        both = a.notna() & b.notna()
        if not both.any():
            continue
        scale = np.maximum(np.abs(b[both]), 1.0)  # relative gap, with an absolute floor of 1 for columns near 0
        worst = float((np.abs(a[both] - b[both]) / scale).max())
        rows.append({"column": column, "years": int(both.sum()), "worst_relative_gap": worst, "ok": worst <= TOLERANCE})
    return pd.DataFrame(rows)


def build_adm2() -> None:
    """One parquet per country: every GADM level-2 unit by year."""
    for iso in ISOS:
        units = load_units_in_country(iso, 2, CACHE).reset_index(drop=True)
        table = atom_year_table(units.geometry, units["gid"], YEARS, GLOBAL, cache_dir=CACHE, id_column="gid")
        table.insert(1, "ISO", iso)
        path = CACHE / "processed" / "covariates" / f"adm2_year_{iso}.parquet"
        table.to_parquet(path)
        _log.info("%s: %d units x %d years -> %s", iso, len(units), len(YEARS), path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("check", "all"):
        report = check_laos()
        pd.set_option("display.width", 200)
        print(report.to_string(index=False), flush=True)
        report.to_csv(Path("results/2026-11-24-hazard-frequency/resampler_laos_check.csv"), index=False)
    if step in ("adm2", "all"):
        build_adm2()
