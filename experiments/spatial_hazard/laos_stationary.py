"""
C0 of task 2026-11-24b: the stationary Laos event model with population-dependent detection.

A script port of ``notebooks/geo_damage_model/12_laos_emdat.ipynb`` (cells 1 to 14: geography, covariate, record, model, fit),
with the plots and the posterior predictive checks left out. The notebook is not modified. Goal: reproduce its river slope,
-1.06 with a 90% interval from -1.86 to -0.20, within sampling noise.

Usage: ``pixi run python experiments/spatial_hazard/laos_stationary.py``  ->  results/2026-11-24-spatial-hazard/c0_*.csv
"""

import time

from pathlib import Path

import arviz as az
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pymc as pm
import pytensor.tensor as pt

from climate_risk.data.gadm import load_units_in_country
from climate_risk.data.geo_disasters import load_resolved_units
from climate_risk.data.ghsl import population_on_cells
from climate_risk.data_functions.emdat_processing import DISASTER_CLASSES, event_geography, load_emdat_events
from climate_risk.data_functions.rivers_data_loader import load_rivers_data
from climate_risk.geo.crs import PROJECTED_CRS, to_km
from climate_risk.geo.distance import get_distance_to
from climate_risk.geo.raster import assign_cells_to_units, build_cell_grid, dissolve_place_boundary
from climate_risk.models.aggregation import Aggregation

CACHE_DIR = Path("data")
RESULTS = Path("results/2026-11-24-spatial-hazard")
ISO = "LAO"
SEED = 20261022


RESOLUTION_KM = 10.0
LEVELS = ("ADM2", "ADM1", "ADM0")

units = {
    "ADM2": load_units_in_country(ISO, 2, CACHE_DIR),
    "ADM1": load_units_in_country(ISO, 1, CACHE_DIR),
}
boundary = dissolve_place_boundary(units["ADM1"][["geometry"]], iso3=ISO)
units["ADM0"] = gpd.GeoDataFrame(
    {"gid": [ISO], "name": ["Laos"], "admin_level": [0]}, geometry=boundary.geometry.to_numpy(), crs=boundary.crs
)
grid = build_cell_grid(boundary, resolution_km=RESOLUTION_KM)
atoms = grid.cells.reset_index(drop=True)
n_atoms = len(atoms)
column_of_cell = pd.Series(np.arange(n_atoms), index=atoms["cell_id"].to_numpy())

# Every (atom, unit) pair that overlaps, with units as positions into units[level].
overlaps = {}
for level in LEVELS:
    pairs = assign_cells_to_units(grid, units[level])
    position = pd.Series(np.arange(len(units[level])), index=units[level]["gid"].to_numpy())
    overlaps[level] = pd.DataFrame(
        {
            "atom": column_of_cell.loc[pairs["cell_id"]].to_numpy(),
            "unit": position.loc[pairs["gid"]].to_numpy(),
            "coverage": pairs["coverage"].to_numpy(),
        }
    )

# The unit covering most of each atom, which is the unit an event centered there is reported in.
unit_of_atom = {
    level: overlaps[level]
    .sort_values("coverage", ascending=False)
    .drop_duplicates("atom")
    .set_index("atom")["unit"]
    .reindex(np.arange(n_atoms))
    .to_numpy()
    for level in LEVELS
}
n_units = {level: len(units[level]) for level in LEVELS}
unit_offset = dict(zip(LEVELS, np.cumsum([0, *[n_units[level] for level in LEVELS[:-1]]]), strict=True))
n_all_units = sum(n_units.values())

pd.Series(
    {
        "atoms": n_atoms,
        **{f"{level} units": n_units[level] for level in LEVELS},
        **{
            f"{level} units dominating no atom": n_units[level] - len(np.unique(unit_of_atom[level]))
            for level in LEVELS
        },
    }
)

POPULATION_EPOCH = 2005
COVARIATE_NAMES = ("log distance to river",)


def standardized(values):
    """Values shifted to mean zero and scaled to unit standard deviation."""
    return (values - values.mean()) / values.std()


rivers = load_rivers_data(CACHE_DIR)
river_km = to_km(
    get_distance_to(
        rivers.cx[
            boundary.total_bounds[0] - 1 : boundary.total_bounds[2] + 1,
            boundary.total_bounds[1] - 1 : boundary.total_bounds[3] + 1,
        ],
        atoms,
    )["distance_to_closest"].to_numpy()
)
covariates = standardized(np.log(river_km + 1.0))[:, None]

population = population_on_cells(grid, POPULATION_EPOCH, CACHE_DIR)
unit_population = {
    level: np.bincount(unit_of_atom[level], weights=population, minlength=n_units[level]) for level in LEVELS
}
population_z = np.concatenate(
    [
        standardized(np.log(np.maximum(unit_population[level], 1.0))) if n_units[level] > 1 else np.zeros(1)
        for level in LEVELS
    ]
)

# Projected atom centers, scaled so the longer side of Laos spans one unit, the domain 11's lengthscale prior assumes.
projected = atoms.to_crs(PROJECTED_CRS).geometry
atom_xy = np.column_stack([projected.x, projected.y])
DOMAIN_KM = to_km(np.ptp(atom_xy, axis=0).max())
atom_centroids = (atom_xy - atom_xy.min(axis=0)) / np.ptp(atom_xy, axis=0).max()

pd.Series(
    {
        "river distance, median km": np.median(river_km),
        "river distance, max km": river_km.max(),
        "population, millions": population.sum() / 1e6,
        "correlation of log population density with river covariate": np.corrcoef(
            np.log1p(population / atoms["place_area_km2"]), covariates[:, 0]
        )[0, 1],
        "domain side, km": DOMAIN_KM,
    }
).round(2)

FIRST_YEAR, LAST_YEAR = 1990, 2023
LEVEL_OF_ADMIN = {2: "ADM2", 1: "ADM1"}

events = load_emdat_events(CACHE_DIR).filter(
    (pl.col("ISO") == ISO)
    & pl.col("Disaster Type").is_in(list(DISASTER_CLASSES))
    & pl.col("date").dt.year().is_between(FIRST_YEAR, LAST_YEAR)
)
resolved = pl.from_pandas(load_resolved_units([ISO], CACHE_DIR)).select(
    pl.col("DisNo."),
    pl.col("gid"),
    pl.col("name"),
    pl.col("admin_level").cast(pl.Int8),
    pl.col("geocoding_q").cast(pl.Int8),
    pl.col("overlap"),
)
geography = event_geography(events, resolved=resolved)

position = {
    level: pd.Series(np.arange(n_units[level]), index=units[level]["gid"].to_numpy()) for level in ("ADM2", "ADM1")
}


def named_units(rows):
    """The units an event names, as positions into units[level], keyed by level."""
    named = {}
    for admin, gid in rows.filter(pl.col("gid").is_not_null()).select("admin_level", "gid").iter_rows():
        level = LEVEL_OF_ADMIN[admin]
        named.setdefault(level, set()).add(int(position[level].loc[gid]))

    return {level: tuple(sorted(found)) for level, found in named.items()}


def provinces_of(named):
    """Positions of the provinces an event names, directly or through a district, empty when it names neither."""
    districts = overlaps["ADM2"]["unit"].isin(named.get("ADM2", ()))

    return frozenset(named.get("ADM1", ())) | frozenset(
        unit_of_atom["ADM1"][overlaps["ADM2"]["atom"][districts]].tolist()
    )


event_rows = []
for (disno,), rows in geography.group_by("DisNo.", maintain_order=True):
    named = named_units(rows)
    event_rows.append(
        {
            "DisNo.": disno,
            "level": "ADM1" if "ADM1" in named else "ADM2" if named else "ADM0",
            "named": named or {"ADM0": (0,)},
            "provinces": provinces_of(named),
            "source": "+".join(sorted(set(rows["geometry_source"]))),
        }
    )

record = pd.DataFrame(event_rows).merge(
    events.select("DisNo.", "Disaster Type", pl.col("date").dt.year().alias("year")).to_pandas(), on="DisNo."
)

pd.concat(
    [
        record.groupby("level").size().rename("events"),
        record.groupby(["level", "source"]).size().unstack(fill_value=0),
    ],
    axis=1,
)

HSGP_BOUNDARY_FACTOR = 4.0
HSGP_BASIS_PER_AXIS = 50
OFFSET_SD_PRIOR = 0.5
OFFSET_LEVELS = ("ADM2", "ADM1")
# A prior guess at the event centers over the whole record, recorded or not, which fixes where the intercept sits.
PRIOR_EVENT_CENTERS = 100
sample_kwargs = {
    "chains": 8,
    "cores": 8,
    "tune": 500,
    "draws": 250,
    "target_accept": 0.9,
    "nuts_sampler": "nutpie",
    "random_seed": SEED,
}

# Each (level, atom) in one flat index, level-major, and the all-units index of the atom's unit at that level.
center_unit = np.concatenate([unit_offset[level] + unit_of_atom[level] for level in LEVELS])


def union_cells(named):
    """The (level, atom) cells of an event's union, read at the level it is reported at."""
    level = "ADM1" if "ADM1" in named else next(iter(named))
    atoms_named = np.unique(
        np.concatenate(
            [overlaps[at]["atom"][overlaps[at]["unit"].isin(found)].to_numpy() for at, found in named.items()]
        )
    )

    return LEVELS.index(level), LEVELS.index(level) * n_atoms + atoms_named


record_key = record["named"].map(lambda named: tuple(sorted(named.items())))
distinct = (
    record.assign(key=record_key)
    .groupby("key", sort=False)
    .agg(multiplicity=("DisNo.", "size"), named=("named", "first"))
)
cells_by_record = [union_cells(named) for named in distinct["named"]]
detection_terms = {
    "multiplicity": distinct["multiplicity"].to_numpy(),
    "union": Aggregation(
        units=tuple(f"record {index}" for index in range(len(distinct))),
        rows=np.repeat(np.arange(len(distinct)), [len(cells) for _, cells in cells_by_record]),
        columns=np.concatenate([cells for _, cells in cells_by_record]),
        weights=np.ones(sum(len(cells) for _, cells in cells_by_record)),
        n_cells=len(LEVELS) * n_atoms,
    ),
}
COORDS = {
    "covariate": list(COVARIATE_NAMES),
    "atom": np.arange(n_atoms),
    "unit": np.arange(n_all_units),
    "level": list(LEVELS),
    "record": detection_terms["union"].units,
    **{level: units[level]["gid"].to_list() for level in OFFSET_LEVELS},
}


def latent_log_rate():
    """The log rate of event centers at every atom, built inside the current model."""
    intercept = pm.Normal("intercept", mu=np.log(PRIOR_EVENT_CENTERS / n_atoms), sigma=1.0)
    slopes = pm.Normal("slopes", mu=0.0, sigma=1.0, dims="covariate")

    # Mode at 0.1 of the domain, about 100 km, with a tail out past the width of the country.
    field_lengthscale = pm.InverseGamma("field_lengthscale", alpha=3.0, beta=0.4)
    field_amplitude = pm.HalfNormal("field_amplitude", sigma=1.0)
    gp = pm.gp.HSGP(
        m=[HSGP_BASIS_PER_AXIS, HSGP_BASIS_PER_AXIS],
        c=HSGP_BOUNDARY_FACTOR,
        cov_func=field_amplitude**2 * pm.gp.cov.Matern32(input_dim=2, ls=field_lengthscale),
    )
    field = gp.prior("field", X=atom_centroids, dims="atom")

    offset_part = 0.0
    for level in OFFSET_LEVELS:
        offset_sd = pm.HalfNormal(f"{level}_offset_sd", sigma=OFFSET_SD_PRIOR)
        standard = pm.Normal(f"{level}_offset_standard", mu=0.0, sigma=1.0, dims=level)
        offsets = pm.Deterministic(f"{level}_offset", offset_sd * standard, dims=level)
        offset_part = offset_part + offsets[unit_of_atom[level]]

    return pm.Deterministic("log_rate", intercept + pm.math.dot(covariates, slopes) + field + offset_part, dims="atom")


def recorded_intensity():
    """Recorded events per (level, atom): centers, times the level's share, times detection by the center's unit there."""
    rate = pm.math.exp(latent_log_rate())
    naming_intercept = pm.Normal("naming_intercept", mu=0.0, sigma=1.5)
    naming_slope = pm.Normal("naming_slope", mu=0.0, sigma=1.0)
    naming = pm.Deterministic(
        "naming_probability", pm.math.invlogit(naming_intercept + naming_slope * population_z), dims="unit"
    )
    level_probability = pm.Dirichlet("level_probability", a=np.ones(len(LEVELS)), dims="level")

    return level_probability[:, None] * rate[None, :] * naming[center_unit].reshape((len(LEVELS), n_atoms))


def counted_logp(count, probability):
    """Log probability of ``count`` events sharing one record, given the record's probability under the model."""
    return count * pt.log(probability)


def build_event_detection_model():
    """The event likelihood with each event recorded with the detection probability of its center's unit."""
    terms = detection_terms

    with pm.Model(coords=COORDS) as model:
        recorded = recorded_intensity()
        recorded_rate = pm.Deterministic("recorded_rate", recorded.sum())

        pm.Poisson("event_count", mu=recorded_rate, observed=terms["multiplicity"].sum())
        pm.CustomDist(
            "record_count",
            terms["union"].aggregate_symbolic(recorded.ravel()) / recorded_rate,
            logp=counted_logp,
            observed=terms["multiplicity"],
            dims="record",
        )

    return model


event_detection_model = build_event_detection_model()

pd.Series(
    {
        "distinct records": len(distinct),
        "largest union, atoms": int(np.bincount(detection_terms["union"].rows).max()),
        "free parameters": sum(value.size for value in event_detection_model.initial_point().values()),
    }
)

# ---- fit (notebook cell 13) ----
if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with event_detection_model:
        fit = pm.sample(**sample_kwargs)
    seconds = time.time() - started
    fit.to_netcdf(CACHE_DIR / "idata" / "spatial_hazard_c0.nc")
    every_free = az.summary(fit, var_names=[variable.name for variable in event_detection_model.free_RVs])
    summary = az.summary(
        fit,
        var_names=[
            "intercept",
            "slopes",
            "field_amplitude",
            "field_lengthscale",
            "naming_intercept",
            "naming_slope",
            "recorded_rate",
        ],
        ci_prob=0.9,
        ci_kind="hdi",
    )
    summary.to_csv(RESULTS / "c0_summary.csv")
    print(summary.round(2).to_string(), flush=True)
    print(
        {
            "seconds": round(seconds),
            "divergences": int(fit.sample_stats["diverging"].sum()),
            "max_r_hat": float(every_free["r_hat"].max()),
            "min_ess_bulk": float(every_free["ess_bulk"].min()),
        },
        flush=True,
    )
