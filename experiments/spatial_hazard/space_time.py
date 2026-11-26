"""
C1 of task 2026-11-24b: a space-time event model with rx5day, on ADM2 atoms.

Starts from notebooks/geo_damage_model/12_laos_emdat.ipynb (event likelihood with population-dependent detection) and
changes three things, as the task asks:
  * atoms are GADM ADM2 units (not a 10 km lattice), so each atom-year has its own area, and the intensity is per km2;
  * the rate has a year dimension: log intensity = intercept + country offset + river slope x log distance to river
    + rx5day slope x rx5day z-score (standardised within the atom) + a shared random-walk year effect;
  * each event's term uses its own year.
The HSGP field and the ADM1/ADM2 offsets of the notebook are not part of this specification and are left out.

Four fits per place, all on the same record terms so PSIS-LOO compares them:
    M0  stationary: river only, no year effect, no rx5day
    M1  river + year effect
    M2  river + year effect + rx5day
    M3  year effect + rx5day, no river     (river slope with and without rx5day: M1 vs M2 vs M0)

Usage: ``pixi run python experiments/spatial_hazard/space_time.py lao|sea``
Output: results/2026-11-24-spatial-hazard/c1_<place>.csv and data/idata/spatial_hazard_<place>_<model>.nc
"""

import json
import sys
import time

from pathlib import Path

import arviz as az
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pymc as pm
import pytensor.tensor as pt
import rasterio

from exactextract import exact_extract
from exactextract.raster import RasterioRasterSource

from climate_risk.config.registry import load_place, resolve_isos
from climate_risk.data.gadm import load_units_in_country
from climate_risk.data.geo_disasters import load_resolved_units
from climate_risk.data.ghsl import population_raster
from climate_risk.data.resample import atom_year_table
from climate_risk.data_functions.emdat_processing import DISASTER_CLASSES, event_geography, load_emdat_events
from climate_risk.data_functions.rivers_data_loader import load_rivers_data
from climate_risk.geo.crs import to_km
from climate_risk.geo.distance import get_distance_to
from climate_risk.models.aggregation import Aggregation

CACHE = Path("data")
GLOBAL = CACHE / "processed" / "covariates" / "global"
RESULTS = Path("results/2026-11-24-spatial-hazard")
FIRST_YEAR, LAST_YEAR = 1990, 2023
YEARS = np.arange(FIRST_YEAR, LAST_YEAR + 1)
LEVELS = ("ADM2", "ADM1", "ADM0")
LEVEL_OF_ADMIN = {2: "ADM2", 1: "ADM1"}
POPULATION_EPOCH = 2005
PRIOR_EVENT_CENTERS_PER_COUNTRY = 100  # the notebook's prior guess for Laos, applied to every country
EQUAL_AREA_CRS = "EPSG:6933"
SEED = 20261022
SAMPLE_KWARGS = {
    "chains": 4,
    "tune": 1000,
    "draws": 1000,
    "target_accept": 0.9,
    "nuts_sampler": "nutpie",
    "random_seed": SEED,
}

MODELS = {  # name: (river, rx5day, year effect)
    "M0_stationary": (True, False, False),
    "M1_river_year": (True, False, True),
    "M2_river_rx5day_year": (True, True, True),
    "M3_rx5day_year": (False, True, True),
}


def standardized(values: np.ndarray) -> np.ndarray:
    """Shift to mean zero and scale to unit standard deviation."""
    return (values - values.mean()) / values.std()


def build_geography(isos: tuple[str, ...]) -> dict:
    """ADM2 atoms, their ADM1 and ADM0 parents, areas, river distance and population for the countries."""
    adm2 = pd.concat([load_units_in_country(iso, 2, CACHE).assign(iso=iso) for iso in isos], ignore_index=True)
    adm1 = pd.concat([load_units_in_country(iso, 1, CACHE).assign(iso=iso) for iso in isos], ignore_index=True)
    atoms = gpd.GeoDataFrame(adm2, geometry="geometry", crs=load_units_in_country(isos[0], 2, CACHE).crs)
    adm1 = gpd.GeoDataFrame(adm1, geometry="geometry", crs=atoms.crs)
    # GADM ids nest: LAO.1.2_1 is a district of LAO.1_1.
    atoms["parent"] = atoms["gid"].map(lambda gid: gid.rsplit(".", 1)[0] + "_1")
    missing = set(atoms["parent"]) - set(adm1["gid"])
    if missing:
        raise ValueError(f"{len(missing)} ADM2 units have no ADM1 parent, e.g. {sorted(missing)[:3]}")
    atoms["area_km2"] = atoms.to_crs(EQUAL_AREA_CRS).area.to_numpy() / 1e6

    countries = pd.DataFrame({"gid": list(isos)})
    unit_index = {
        "ADM2": pd.Series(np.arange(len(atoms)), index=atoms["gid"].to_numpy()),
        "ADM1": pd.Series(np.arange(len(adm1)), index=adm1["gid"].to_numpy()),
        "ADM0": pd.Series(np.arange(len(countries)), index=countries["gid"].to_numpy()),
    }
    unit_of_atom = {
        "ADM2": np.arange(len(atoms)),
        "ADM1": unit_index["ADM1"].loc[atoms["parent"]].to_numpy(),
        "ADM0": unit_index["ADM0"].loc[atoms["iso"]].to_numpy(),
    }
    n_units = {"ADM2": len(atoms), "ADM1": len(adm1), "ADM0": len(countries)}

    # Rivers: log distance from each atom's representative point to the nearest major river, in km.
    points = gpd.GeoDataFrame(geometry=atoms.representative_point().to_numpy(), crs=atoms.crs)
    rivers = load_rivers_data(CACHE)
    west, south, east, north = atoms.total_bounds
    river_km = to_km(
        get_distance_to(rivers.cx[west - 1 : east + 1, south - 1 : north + 1], points)["distance_to_closest"].to_numpy()
    )

    # Population of every unit, from GHS-POP summed over its polygon, then the detection covariate: log population
    # standardised within country at each level (ADM0 is one unit per country, so it is zero).
    with rasterio.open(population_raster(POPULATION_EPOCH, CACHE)) as source:
        pop_adm2 = exact_extract(RasterioRasterSource(source), atoms[["geometry"]], ["sum"], output="pandas")[
            "sum"
        ].to_numpy()
    pop = {
        "ADM2": pop_adm2,
        "ADM1": np.bincount(unit_of_atom["ADM1"], weights=pop_adm2, minlength=n_units["ADM1"]),
        "ADM0": np.bincount(unit_of_atom["ADM0"], weights=pop_adm2, minlength=n_units["ADM0"]),
    }
    country_of_unit = {
        "ADM2": atoms["iso"].to_numpy(),
        "ADM1": adm1["iso"].to_numpy(),
        "ADM0": countries["gid"].to_numpy(),
    }
    population_z = []
    for level in LEVELS:
        z = np.zeros(n_units[level])
        if level != "ADM0":
            for iso in isos:
                mask = country_of_unit[level] == iso
                logs = np.log(np.maximum(pop[level][mask], 1.0))
                z[mask] = (logs - logs.mean()) / logs.std() if mask.sum() > 1 and logs.std() > 0 else 0.0
        population_z.append(z)

    return {
        "atoms": atoms,
        "adm1": adm1,
        "unit_index": unit_index,
        "unit_of_atom": unit_of_atom,
        "n_units": n_units,
        "river_z": standardized(np.log(river_km + 1.0)),
        "population_z": np.concatenate(population_z),
        "isos": isos,
    }


def rx5day_z(geo: dict) -> np.ndarray:
    """rx5day (ERA5, population-weighted) for each atom-year, z-scored within the atom over the record. Missing -> 0."""
    table = atom_year_table(
        geo["atoms"].geometry, geo["atoms"]["gid"], YEARS, GLOBAL, cache_dir=CACHE, sources=["era5_daily"]
    )
    values = (
        table.pivot(index="atom_id", columns="year", values="rx5day_popmean").reindex(geo["atoms"]["gid"]).to_numpy()
    )
    mean, sd = np.nanmean(values, axis=1, keepdims=True), np.nanstd(values, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (values - mean) / sd
    geo["rx5day_missing_atoms"] = int(np.isnan(z).all(axis=1).sum())
    return np.nan_to_num(z, nan=0.0)


def build_record(geo: dict) -> dict:
    """The events as distinct (reporting level, union of atoms, year) records with their multiplicity."""
    isos = list(geo["isos"])
    events = load_emdat_events(CACHE).filter(
        pl.col("ISO").is_in(isos)
        & pl.col("Disaster Type").is_in(list(DISASTER_CLASSES))
        & pl.col("date").dt.year().is_between(FIRST_YEAR, LAST_YEAR)
    )
    resolved = pl.from_pandas(load_resolved_units(isos, CACHE)).select(
        pl.col("DisNo."),
        pl.col("gid"),
        pl.col("name"),
        pl.col("admin_level").cast(pl.Int8),
        pl.col("geocoding_q").cast(pl.Int8),
        pl.col("overlap"),
    )
    geography = event_geography(events, resolved=resolved)
    meta = {
        row["DisNo."]: (row["ISO"], row["year"])
        for row in events.select("DisNo.", "ISO", pl.col("date").dt.year().alias("year")).iter_rows(named=True)
    }
    atoms = geo["atoms"]
    atoms_of_adm1 = {gid: np.flatnonzero(atoms["parent"].to_numpy() == gid) for gid in geo["adm1"]["gid"]}
    atoms_of_country = {iso: np.flatnonzero(atoms["iso"].to_numpy() == iso) for iso in isos}
    index = geo["unit_index"]

    keys: dict[tuple, int] = {}
    unions: dict[tuple, np.ndarray] = {}
    for (disno,), rows in geography.group_by("DisNo.", maintain_order=True):
        iso, year = meta[disno]
        named: dict[str, list[str]] = {}
        for admin, gid in rows.filter(pl.col("gid").is_not_null()).select("admin_level", "gid").iter_rows():
            if admin in LEVEL_OF_ADMIN and gid in index[LEVEL_OF_ADMIN[admin]].index:
                named.setdefault(LEVEL_OF_ADMIN[admin], []).append(gid)
        if "ADM1" in named:
            level, found = (
                "ADM1",
                np.unique(
                    np.concatenate(
                        [atoms_of_adm1[g] for g in named["ADM1"]]
                        + [index["ADM2"].loc[named.get("ADM2", [])].to_numpy()]
                    )
                ),
            )
        elif "ADM2" in named:
            level, found = "ADM2", np.unique(index["ADM2"].loc[named["ADM2"]].to_numpy())
        else:
            level, found = "ADM0", atoms_of_country[iso]
        key = (level, year, tuple(found.tolist()))
        keys[key] = keys.get(key, 0) + 1
        unions[key] = found
    ordered = list(keys)
    return {
        "keys": ordered,
        "multiplicity": np.array([keys[k] for k in ordered]),
        "unions": [unions[k] for k in ordered],
        "n_events": int(sum(keys.values())),
    }


def build_model(geo: dict, record: dict, rx: np.ndarray, *, river: bool, rain: bool, year_effect: bool) -> pm.Model:
    """The space-time event likelihood with detection. See the module docstring for the structure."""
    atoms = geo["atoms"]
    n_atoms, n_years, n_levels = len(atoms), len(YEARS), len(LEVELS)
    country_idx = pd.factorize(atoms["iso"], sort=True)[0]
    n_countries = country_idx.max() + 1
    area = atoms["area_km2"].to_numpy()
    n_all_units = sum(geo["n_units"].values())
    offsets = np.cumsum([0, *[geo["n_units"][lv] for lv in LEVELS[:-1]]])
    center_unit = np.stack([offsets[i] + geo["unit_of_atom"][lv] for i, lv in enumerate(LEVELS)])  # (L, A)
    population_z = geo["population_z"]

    rows, columns = [], []
    for k, (level, year, _) in enumerate(record["keys"]):
        union = record["unions"][k]
        rows.append(np.full(len(union), k))
        columns.append(LEVELS.index(level) * n_atoms * n_years + union * n_years + (year - FIRST_YEAR))
    union_operator = Aggregation(
        units=tuple(f"record {k}" for k in range(len(record["keys"]))),
        rows=np.concatenate(rows),
        columns=np.concatenate(columns),
        weights=np.ones(sum(len(r) for r in rows)),
        n_cells=n_levels * n_atoms * n_years,
    )
    coords = {
        "atom": np.arange(n_atoms),
        "year": YEARS,
        "unit": np.arange(n_all_units),
        "level": list(LEVELS),
        "record": union_operator.units,
        "country": np.arange(n_countries),
    }

    with pm.Model(coords=coords) as model:
        total_area = float(area.sum())
        intercept = pm.Normal(
            "intercept", mu=np.log(PRIOR_EVENT_CENTERS_PER_COUNTRY * n_countries / (total_area * n_years)), sigma=1.0
        )
        eta = intercept + pt.zeros((n_atoms, n_years))
        if n_countries > 1:
            country_sd = pm.HalfNormal("country_offset_sd", sigma=0.5)
            country_standard = pm.Normal("country_offset_standard", mu=0.0, sigma=1.0, dims="country")
            eta = eta + (country_sd * country_standard)[country_idx][:, None]
        if river:
            river_slope = pm.Normal("river_slope", mu=0.0, sigma=1.0)
            eta = eta + river_slope * geo["river_z"][:, None]
        if rain:
            rx_slope = pm.Normal("rx5day_slope", mu=0.0, sigma=1.0)
            eta = eta + rx_slope * rx
        if year_effect:
            year_sd = pm.HalfNormal("year_effect_sd", sigma=0.3)
            step = pm.Normal("year_effect_step", mu=0.0, sigma=1.0, shape=n_years - 1)
            walk = pm.Deterministic(
                "year_effect", pt.concatenate([pt.zeros(1), pt.cumsum(year_sd * step)]), dims="year"
            )
            eta = eta + walk[None, :]
        intensity = area[:, None] * pt.exp(eta)  # centers per atom-year

        naming_intercept = pm.Normal("naming_intercept", mu=0.0, sigma=1.5)
        naming_slope = pm.Normal("naming_slope", mu=0.0, sigma=1.0)
        naming = pm.Deterministic(
            "naming_probability", pm.math.invlogit(naming_intercept + naming_slope * population_z), dims="unit"
        )
        level_probability = pm.Dirichlet("level_probability", a=np.ones(n_levels), dims="level")
        recorded = (
            level_probability[:, None, None] * intensity[None, :, :] * naming[center_unit][:, :, None]
        )  # (L, A, Y)
        recorded_rate = pm.Deterministic("recorded_rate", recorded.sum())

        pm.Poisson("event_count", mu=recorded_rate, observed=record["n_events"])
        pm.CustomDist(
            "record_count",
            union_operator.aggregate_symbolic(recorded.ravel()) / recorded_rate,
            logp=lambda count, probability: count * pt.log(probability),
            observed=record["multiplicity"],
            dims="record",
        )
    return model


def main(place: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    isos = tuple(resolve_isos(load_place(place)))
    started = time.time()
    geo = build_geography(isos)
    rx = rx5day_z(geo)
    record = build_record(geo)
    size = {
        "place": place,
        "countries": len(isos),
        "atoms": len(geo["atoms"]),
        "years": len(YEARS),
        "atom_years": len(geo["atoms"]) * len(YEARS),
        "events": record["n_events"],
        "distinct_records": len(record["keys"]),
        "atoms_without_rx5day": geo["rx5day_missing_atoms"],
        "setup_seconds": round(time.time() - started),
    }
    print(json.dumps(size), flush=True)
    (RESULTS / f"c1_{place}_size.json").write_text(json.dumps(size, indent=2))

    rows, loos = [], {}
    for name, (river, rain, year_effect) in MODELS.items():
        model = build_model(geo, record, rx, river=river, rain=rain, year_effect=year_effect)
        began = time.time()
        with model:
            fit = pm.sample(**SAMPLE_KWARGS)
            pm.compute_log_likelihood(fit, var_names=["record_count"], progressbar=False)
        seconds = time.time() - began
        free = az.summary(fit, var_names=[v.name for v in model.free_RVs], kind="diagnostics")
        loos[name] = az.loo(fit, var_name="record_count")
        row = {
            "place": place,
            "model": name,
            "seconds": round(seconds),
            "divergences": int(fit.sample_stats["diverging"].sum()),
            "max_r_hat": float(free["r_hat"].max()),
            "min_ess_bulk": float(free["ess_bulk"].min()),
        }
        for term in ("river_slope", "rx5day_slope"):
            if term in fit.posterior:
                draws = fit.posterior[term].to_numpy().ravel()
                lo, hi = az.hdi(draws, prob=0.89)
                row |= {
                    f"{term}_mean": float(draws.mean()),
                    f"{term}_hdi89_lb": float(lo),
                    f"{term}_hdi89_ub": float(hi),
                }
        rows.append(row)
        print(json.dumps(row), flush=True)
        fit.to_netcdf(CACHE / "idata" / f"spatial_hazard_{place}_{name}.nc")
        pd.DataFrame(rows).to_csv(RESULTS / f"c1_{place}.csv", index=False)

    table = az.compare(loos, round_to=2)
    table.to_csv(RESULTS / f"c1_{place}_loo.csv")
    print(table[["rank", "elpd", "se", "elpd_diff", "dse", "p"]], flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lao")
