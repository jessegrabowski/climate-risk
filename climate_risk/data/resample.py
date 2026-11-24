"""
Resample the yearly global covariate grids onto any set of target shapes (the "atoms").

The global statistics sit in ``data/processed/covariates/global/<source>_<year>.nc``, one file per source and
year, each on its own lattice (ERA5 0.25 degrees, ERA5-Land 0.1, CHIRPS 0.25, GloFAS 0.05). An atom is whatever the
model integrates over: a 10 km lattice cell or an administrative unit. :func:`atom_year_table` gives one row per atom
and year, with every statistic as an area-weighted mean, and as a population-weighted mean where a GHS-POP weight
exists. Pixel overlaps use ``exactextract`` coverage times cos(latitude), the same weighting as the country table.
"""

import logging

from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from climate_risk.data.covariate_table import PolygonReducer
from climate_risk.data.global_population import nearest_epoch, population_on_lattice

_log = logging.getLogger(__name__)

# source file stem -> variables it carries
SOURCES: dict[str, tuple[str, ...]] = {
    "era5_daily": ("precip_total", "rx5day"),
    "era5_anom": ("precip_anom_std",),
    "chirps_v2": ("precip_total", "rx5day"),  # CHIRPS v2.0, IRI mirror, 0.25-degree box average, 50N-50S
    "era5land": ("soilm_mean", "soilm_max_month", "t2m_mean", "precip_land", "pet_annual"),
    "glofas": ("q_annual_max",),
    "glofas_days": ("q_days_over_rp5",),
}

# Sources that also get a population-weighted mean, and the lattice whose GHS-POP block sums they use.
POPULATION_LATTICE = {"era5_daily": "era5_daily", "era5_anom": "era5_daily", "era5land": "era5land"}

# Column names that would clash between sources.
RENAME = {
    ("era5_daily", "precip_total"): "precip_era5",
    ("chirps_v2", "precip_total"): "chirps_precip",
    ("chirps_v2", "rx5day"): "chirps_rx5day",
}


def _population(
    directory: Path, lattice: str, epoch: int, lon: np.ndarray, lat: np.ndarray, cache_dir: Path
) -> np.ndarray:
    """GHS-POP summed onto a source lattice, cached next to the grids."""
    cache = directory / f"population_{lattice}_{epoch}.npy"
    if not cache.exists():
        np.save(cache, population_on_lattice(epoch, lon, lat, cache_dir))

    return np.load(cache)  # type: ignore[no-any-return]


def atom_year_table(
    atoms: gpd.GeoSeries,
    atom_ids: Iterable[object],
    years: Iterable[int],
    directory: Path,
    *,
    cache_dir: Path = Path("data"),
    sources: Iterable[str] | None = None,
    id_column: str = "atom_id",
) -> pd.DataFrame:
    """
    Area-weighted (and population-weighted) yearly means of the global statistics over each atom.

    Parameters
    ----------
    atoms : GeoSeries
        Target shapes in longitude/latitude. For lattice cells pass the footprints clipped to the place
        (``grid.footprints.intersection(grid.place)``), so a half-inside cell counts only the part inside.
    atom_ids : iterable
        One identifier per atom, in the order of ``atoms``.
    years : iterable of int
        Calendar years. A year with no file for a source leaves that source's columns NaN.
    directory : Path
        Where the yearly ``<source>_<year>.nc`` files live (``data/processed/covariates/global``).
    cache_dir : Path, optional
        The repo data directory, where GHS-POP is cached. Default ``data``.
    sources : iterable of str, optional
        Subset of :data:`SOURCES` to use. Default all.
    id_column : str, optional
        Name of the identifier column in the result. Default ``atom_id``.

    Returns
    -------
    table : DataFrame
        One row per atom and year: ``id_column``, ``year``, then ``<name>_mean`` for every statistic and
        ``<name>_popmean`` for the ERA5 and ERA5-Land ones. Atoms a lattice does not reach (CHIRPS north of
        50 degrees, ERA5-Land small islands) are NaN.
    """
    ids = np.asarray(list(atom_ids))
    years = list(years)
    if len(ids) != len(atoms):
        raise ValueError(f"{len(ids)} ids for {len(atoms)} atoms.")
    wanted = list(sources) if sources is not None else list(SOURCES)
    shapes = gpd.GeoSeries(atoms.reset_index(drop=True))

    table = pd.DataFrame({id_column: np.repeat(ids, len(years)), "year": np.tile(years, len(ids))})
    for stem in wanted:
        reducer: PolygonReducer | None = None
        columns: dict[str, np.ndarray] = {}
        for index, year in enumerate(years):
            path = directory / f"{stem}_{year}.nc"
            if not path.exists():
                continue
            with xr.open_dataset(path) as opened:
                data = opened.load()
            lon, lat = data["longitude"].to_numpy(), data["latitude"].to_numpy()
            if reducer is None:
                reducer = PolygonReducer(shapes, lon, lat)
            for variable in SOURCES[stem]:
                name = RENAME.get((stem, variable), variable)
                field = data[variable].to_numpy()
                columns.setdefault(f"{name}_mean", np.full((len(ids), len(years)), np.nan))[:, index] = reducer.mean(
                    field
                )
                if stem in POPULATION_LATTICE:
                    weights = _population(directory, POPULATION_LATTICE[stem], nearest_epoch(year), lon, lat, cache_dir)
                    columns.setdefault(f"{name}_popmean", np.full((len(ids), len(years)), np.nan))[:, index] = (
                        reducer.weighted_mean(field, weights)
                    )
            _log.info("%s %d", stem, year)
        for column, values in columns.items():
            table[column] = values.ravel()  # atom-major, matching the repeat/tile above

    return table
