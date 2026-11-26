"""
Task 2026-11-26, part 3: the C1 Southeast Asia model with the notebook's spatial structure put back.

Each hazard has its own rate per ADM2 atom and year:
    log intensity = intercept + country offset + ADM1 offset + ADM2 offset + smooth spatial field (HSGP)
                    + covariate slopes + a random-walk year effect
and the hazards share the detection term of notebook 12 (the chance an event is recorded rises with the population of
the unit it is reported in, and a level it is reported at). The field and offsets follow
notebooks/geo_damage_model/12_laos_emdat.ipynb: Matern 3/2, 50 basis functions per axis, boundary factor 4,
lengthscale prior InverseGamma(3, 0.4) in units of the domain's longer side, offset SD priors HalfNormal(0.5).

Models
    M4          3a: all hydrometeorological and climatological events, river + rx5day
    M4_placebo  3a: the same on geophysical events
    M5          3b: floods (with wet mass movements) and storms, each with river + rx5day + tropical-cyclone share

Usage: ``pixi run python experiments/spatial_hazard/space_time_v2.py M4 [target_accept]``
Output: results/2026-11-26-spatial-hazard-v2/<model>.json and data/idata/spatial_v2_<model>.nc
"""

import json
import sys
import time

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import polars as pl
import pymc as pm
import pytensor.tensor as pt

from space_time import (
    CACHE,
    FIRST_YEAR,
    LAST_YEAR,
    LEVEL_OF_ADMIN,
    LEVELS,
    PRIOR_EVENT_CENTERS_PER_COUNTRY,
    SAMPLE_KWARGS,
    YEARS,
    build_geography,
    counted_logp,
    rx5day_z,
)

from climate_risk.config.registry import load_place, resolve_isos
from climate_risk.data.geo_disasters import load_resolved_units
from climate_risk.data_functions.emdat_processing import DISASTER_CLASSES, event_geography, load_emdat_events
from climate_risk.models.aggregation import Aggregation
from climate_risk.models.identification import GEOPHYSICAL_TYPES

PLACE = "sea"
RESULTS = Path("results/2026-11-26-spatial-hazard-v2")
TC_TABLE = CACHE / "processed" / "covariates" / "tc_atom_year_sea.parquet"
HSGP_BOUNDARY_FACTOR = 4.0
HSGP_BASIS_PER_AXIS = 50
OFFSET_SD_PRIOR = 0.5
KM_PER_DEGREE = 111.2

MODELS: dict[str, dict] = {
    "M4": {"hazards": {"all": tuple(DISASTER_CLASSES)}, "covariates": ("river", "rx5day")},
    "M4_placebo": {"hazards": {"geophysical": tuple(GEOPHYSICAL_TYPES)}, "covariates": ("river", "rx5day")},
    "M5": {
        "hazards": {"flood": ("Flood", "Mass movement (wet)"), "storm": ("Storm",)},
        "covariates": ("river", "rx5day", "tc_share_34kt"),
    },
}


def atom_positions(geo: dict) -> tuple[np.ndarray, float]:
    """Atom positions on a local flat map, scaled so the longer side of the region spans 1; also that side in km."""
    points = geo["atoms"].geometry.representative_point()
    lon, lat = points.x.to_numpy(), points.y.to_numpy()
    xy = np.column_stack([lon * np.cos(np.deg2rad(lat.mean())), lat]) * KM_PER_DEGREE
    side = float(np.ptp(xy, axis=0).max())
    return (xy - xy.min(axis=0)) / side, side


def build_record(geo: dict, types: tuple[str, ...]) -> dict:
    """As ``space_time.build_record``, for any set of EM-DAT types."""
    isos = list(geo["isos"])
    events = load_emdat_events(CACHE).filter(
        pl.col("ISO").is_in(isos)
        & pl.col("Disaster Type").is_in(list(types))
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
            parts = [atoms_of_adm1[g] for g in named["ADM1"]] + [index["ADM2"].loc[named.get("ADM2", [])].to_numpy()]
            level, found = "ADM1", np.unique(np.concatenate(parts))
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


def union_operator(record: dict, n_atoms: int, label: str) -> Aggregation:
    """The sparse matrix that sums recorded intensity over each record's union, level and year."""
    n_years = len(YEARS)
    rows, columns = [], []
    for k, (level, year, _) in enumerate(record["keys"]):
        union = record["unions"][k]
        rows.append(np.full(len(union), k))
        columns.append(LEVELS.index(level) * n_atoms * n_years + union * n_years + (year - FIRST_YEAR))
    all_rows = np.concatenate(rows)
    return Aggregation(
        units=tuple(f"{label} record {k}" for k in range(len(record["keys"]))),
        rows=all_rows,
        columns=np.concatenate(columns),
        weights=np.ones(len(all_rows)),
        n_cells=len(LEVELS) * n_atoms * n_years,
    )


def tc_share(geo: dict) -> np.ndarray:
    """Share of each atom's population under tropical-storm winds in each year, scaled by its overall SD."""
    table = pl.read_parquet(TC_TABLE).to_pandas()
    values = (
        table.pivot(index="atom_id", columns="year", values="tc_share_34kt")
        .reindex(index=geo["atoms"]["gid"], columns=YEARS)
        .to_numpy()
    )
    if np.isnan(values).mean() > 0.01:
        raise ValueError("More than 1% of atom-years have no cyclone value")
    values = np.nan_to_num(values, nan=0.0)
    return (values - values.mean()) / values.std()


def build_model(geo: dict, records: dict[str, dict], covariates: dict[str, np.ndarray]) -> pm.Model:
    """One rate per hazard (shared detection), each with covariates, offsets, a spatial field and a year effect."""
    atoms = geo["atoms"]
    n_atoms, n_years, n_levels = len(atoms), len(YEARS), len(LEVELS)
    country_idx = pd.factorize(atoms["iso"], sort=True)[0]
    n_countries = country_idx.max() + 1
    area = atoms["area_km2"].to_numpy()
    offsets = np.cumsum([0, *[geo["n_units"][lv] for lv in LEVELS[:-1]]])
    center_unit = np.stack([offsets[i] + geo["unit_of_atom"][lv] for i, lv in enumerate(LEVELS)])
    positions, _ = atom_positions(geo)
    operators = {name: union_operator(record, n_atoms, name) for name, record in records.items()}
    coords = {
        "atom": np.arange(n_atoms),
        "year": YEARS,
        "unit": np.arange(sum(geo["n_units"].values())),
        "level": list(LEVELS),
        "country": np.arange(n_countries),
        "ADM1": np.arange(geo["n_units"]["ADM1"]),
        "ADM2": np.arange(geo["n_units"]["ADM2"]),
        **{f"record_{name}": operator.units for name, operator in operators.items()},
    }

    with pm.Model(coords=coords) as model:
        naming_intercept = pm.Normal("naming_intercept", mu=0.0, sigma=1.5)
        naming_slope = pm.Normal("naming_slope", mu=0.0, sigma=1.0)
        naming = pm.Deterministic(
            "naming_probability", pm.math.invlogit(naming_intercept + naming_slope * geo["population_z"]), dims="unit"
        )
        level_probability = pm.Dirichlet("level_probability", a=np.ones(n_levels), dims="level")

        for name, record in records.items():
            prefix = "" if len(records) == 1 else f"{name}_"
            total_area = float(area.sum())
            intercept = pm.Normal(
                f"{prefix}intercept",
                mu=np.log(PRIOR_EVENT_CENTERS_PER_COUNTRY * n_countries / (total_area * n_years)),
                sigma=1.0,
            )
            country_sd = pm.HalfNormal(f"{prefix}country_offset_sd", sigma=0.5)
            country_standard = pm.Normal(f"{prefix}country_offset_standard", mu=0.0, sigma=1.0, dims="country")
            spatial = intercept + (country_sd * country_standard)[country_idx]

            lengthscale = pm.InverseGamma(f"{prefix}field_lengthscale", alpha=3.0, beta=0.4)
            amplitude = pm.HalfNormal(f"{prefix}field_amplitude", sigma=1.0)
            gp = pm.gp.HSGP(
                m=[HSGP_BASIS_PER_AXIS, HSGP_BASIS_PER_AXIS],
                c=HSGP_BOUNDARY_FACTOR,
                cov_func=amplitude**2 * pm.gp.cov.Matern32(input_dim=2, ls=lengthscale),
            )
            spatial = spatial + gp.prior(f"{prefix}field", X=positions, dims="atom")
            for level in ("ADM2", "ADM1"):
                sd = pm.HalfNormal(f"{prefix}{level}_offset_sd", sigma=OFFSET_SD_PRIOR)
                standard = pm.Normal(f"{prefix}{level}_offset_standard", mu=0.0, sigma=1.0, dims=level)
                spatial = spatial + (sd * standard)[geo["unit_of_atom"][level]]

            eta = spatial[:, None] + pt.zeros((n_atoms, n_years))
            for covariate, values in covariates.items():
                slope = pm.Normal(f"{prefix}{covariate}_slope", mu=0.0, sigma=1.0)
                eta = eta + slope * values
            year_sd = pm.HalfNormal(f"{prefix}year_effect_sd", sigma=0.3)
            step = pm.Normal(f"{prefix}year_effect_step", mu=0.0, sigma=1.0, shape=n_years - 1)
            walk = pm.Deterministic(
                f"{prefix}year_effect", pt.concatenate([pt.zeros(1), pt.cumsum(year_sd * step)]), dims="year"
            )
            intensity = area[:, None] * pt.exp(eta + walk[None, :])

            recorded = level_probability[:, None, None] * intensity[None, :, :] * naming[center_unit][:, :, None]
            recorded_rate = pm.Deterministic(f"{prefix}recorded_rate", recorded.sum())
            pm.Poisson(f"{prefix}event_count", mu=recorded_rate, observed=record["n_events"])
            pm.CustomDist(
                f"{prefix}record_count",
                operators[name].aggregate_symbolic(recorded.ravel()) / recorded_rate,
                logp=counted_logp,
                observed=record["multiplicity"],
                dims=f"record_{name}",
            )
    return model


def main(label: str, target_accept: float) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    spec = MODELS[label]
    geo = build_geography(tuple(resolve_isos(load_place(PLACE))))
    available = {
        "river": lambda: geo["river_z"][:, None],
        "rx5day": lambda: rx5day_z(geo),
        "tc_share_34kt": lambda: tc_share(geo),
    }
    covariates = {name: available[name]() for name in spec["covariates"]}
    records = {name: build_record(geo, types) for name, types in spec["hazards"].items()}
    _, side_km = atom_positions(geo)
    info = {
        "model": label,
        "target_accept": target_accept,
        "atoms": len(geo["atoms"]),
        "years": len(YEARS),
        "events": {name: record["n_events"] for name, record in records.items()},
        "distinct_records": {name: len(record["keys"]) for name, record in records.items()},
        "domain_longer_side_km": round(side_km),
        # Matern 3/2 rule of thumb: the basis resolves lengthscales above 3.42 * c * half-width / m.
        "smallest_resolved_lengthscale": 3.42 * HSGP_BOUNDARY_FACTOR * 0.5 / HSGP_BASIS_PER_AXIS,
    }
    print(json.dumps(info), flush=True)

    model = build_model(geo, records, covariates)
    names = [f"{'' if len(records) == 1 else name + '_'}record_count" for name in records]
    began = time.time()
    with model:
        fit = pm.sample(**{**SAMPLE_KWARGS, "target_accept": target_accept})
        pm.compute_log_likelihood(fit, var_names=names, progressbar=False)
    info["seconds"] = round(time.time() - began)
    suffix = "" if target_accept == SAMPLE_KWARGS["target_accept"] else f"_ta{round(target_accept * 100)}"
    fit.to_netcdf(CACHE / "idata" / f"spatial_v2_{label}{suffix}.nc")

    info["divergences"] = int(fit.sample_stats["diverging"].sum())
    info["draws_total"] = int(fit.sample_stats["diverging"].size)
    free = az.summary(fit, var_names=[v.name for v in model.free_RVs], kind="diagnostics")
    info["max_r_hat"] = float(free["r_hat"].max())
    info["min_ess_bulk"] = float(free["ess_bulk"].min())
    info["free_parameters"] = len(free)
    for term in [v.name for v in model.free_RVs]:
        if term.endswith(("_slope", "field_lengthscale", "field_amplitude", "offset_sd")):
            draws = fit.posterior[term].to_numpy().ravel()
            low, high = az.hdi(draws, prob=0.89)
            info[term] = {"mean": float(draws.mean()), "hdi89": [float(low), float(high)]}
    for name in names:
        loo = az.loo(fit, var_name=name)
        info[f"loo_{name}"] = {
            "elpd": float(loo.elpd),
            "p": float(loo.p),
            "pareto_k_gt_0.7": int((loo.pareto_k > 0.7).sum()),
        }
    (RESULTS / f"{label}{suffix}.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else SAMPLE_KWARGS["target_accept"])
