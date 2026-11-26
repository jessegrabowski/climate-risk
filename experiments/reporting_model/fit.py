"""
Task 2026-11-26, part 4: a global temperature term, with EM-DAT reporting modelled from outside information.

Hydrometeorological and geophysical counts per country-year, fitted jointly (the structure of the Test 2 joint model:
class intercepts and slopes, correlated country intercepts, zero-inflated negative binomial per class).

    4a   non-climate covariates + reporting covariates (one set of slopes SHARED by both classes) + global mean
         surface temperature (GMST, a slope per class). No year effect.
    4c   4a + local hazards (rx5day z-scored within country; share of population under 34-knot cyclone winds)
    ye_a, ye_c   the comparison: a shared random-walk year effect in place of GMST and reporting covariates
    4a_ocean     4a with the ocean heat series used by Test 1's temperature shock in place of GMST (a check only)

Every model is run on all years and on 1990 onward.

Pass rule (4a and 4c, both samples): the geophysical temperature slope's 89% interval includes 0 and the size of its
mean is under a third of the hydro slope's; the hydro interval excludes 0.

Usage: ``pixi run python experiments/reporting_model/fit.py``
"""

import gc
import sys

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import polars as pl
import pymc as pm
import pytensor.tensor as pt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hazard_frequency"))

from model import within_country_z

from climate_risk.data.global_temperature import load_global_temperature
from climate_risk.data.reporting import load_reporting_indicators, zero_before_first_observation
from climate_risk.data_functions.combine_data import build_time_series
from climate_risk.data_functions.emdat_processing import load_emdat_events
from climate_risk.models.frequency import FEATURES, MAX_MU, MIN_MU
from climate_risk.models.identification import (
    JOINT_CLASSES,
    JOINT_TARGETS,
    _prepare,
    _year_index,
    with_geophysical_disasters,
)
from climate_risk.replication_data import create_replication_data
from climate_risk.sample import sample_or_load

CACHE = Path("data")
FITS = CACHE / "idata" / "reporting_model"
RESULTS = Path("results/2026-11-26-reporting-model")
COVARIATES = CACHE / "processed" / "covariates"
SAMPLE_KWARGS = {"chains": 4, "draws": 2000, "tune": 1000, "random_seed": 1, "target_accept": 0.95}

NON_CLIMATE = ["ln_population_density", "ln_gdp_pc", "square_ln_gdp_pc", "ln_population"]
INTERNET_MOBILE = ["internet_users_pct", "mobile_per_100"]
STEP = "from_2000"
HAZARDS = ["rx5day_popmean", "tc_share_34kt"]
TARGETS = [JOINT_TARGETS[name] for name in JOINT_CLASSES]


def build_panel() -> pl.DataFrame:
    """The published rows with geophysical counts, reporting covariates, temperature series and local hazards."""
    published = create_replication_data(CACHE).drop_nulls([f for f in FEATURES if f != "ln_population"])
    panel = with_geophysical_disasters(published, load_emdat_events(CACHE))
    panel = panel.with_columns(pl.col("date").dt.year().cast(pl.Int64).alias("year"))

    reporting = zero_before_first_observation(load_reporting_indicators(CACHE)).rename({"country_code": "ISO"})
    gmst = load_global_temperature(CACHE)
    ocean = build_time_series(CACHE, frequency="annual").select(
        pl.col("date").dt.year().cast(pl.Int64).alias("year"), pl.col("Temp").alias("ocean_heat")
    )
    rain = pl.read_parquet(COVARIATES / "global_country_year.parquet").select(
        "ISO", pl.col("year").cast(pl.Int64), "rx5day_popmean"
    )
    storms = pl.read_parquet(COVARIATES / "tc_country_year.parquet").select(
        "ISO", pl.col("year").cast(pl.Int64), "tc_share_34kt"
    )
    joined = (
        panel.join(reporting, on=["ISO", "year"], how="left")
        .join(gmst, on="year", how="left")
        .join(ocean, on="year", how="left")
        .join(rain, on=["ISO", "year"], how="left")
        .join(storms, on=["ISO", "year"], how="left")
        .with_columns((pl.col("year") >= 2000).cast(pl.Float64).alias(STEP))
        .sort("ISO", "year")
    )
    assert joined.height == published.height, "a join duplicated rows"

    return within_country_z(joined, ["rx5day_popmean"])


def build_model(panel: pl.DataFrame, *, temperature: str | None, hazards: list[str], year_effect: bool) -> pm.Model:
    """
    The joint model. ``temperature`` names the global series (None for the year-effect comparison).

    With ``year_effect`` True there are no reporting covariates: the shared walk stands in for them.
    """
    climate = ([temperature] if temperature else []) + hazards
    reporting = [] if year_effect else INTERNET_MOBILE
    features = [*NON_CLIMATE, *climate]
    data = _prepare(panel, [*features, *reporting], TARGETS)
    iso_idx, isos = pd.factorize(data["ISO"], sort=True)
    years, year_idx = _year_index(data)
    coords: dict[str, object] = {
        "feature": features,
        "reporting": [*reporting, STEP],
        "obs_id": np.arange(len(data)),
        "ISO": isos,
        "year": years,
        "disaster": list(JOINT_CLASSES),
    }

    with pm.Model(coords=coords) as model:
        X = pm.Data("X", data[[f"{f}__standardized" for f in features]].to_numpy(), dims=["obs_id", "feature"])
        intercept = pm.Normal("intercept", 0.0, 1.0, dims="disaster")
        beta = pm.Normal("beta", 0.0, 1.0, dims=["feature", "disaster"])
        chol, corr, _ = pm.LKJCholeskyCov("country_cov", n=2, eta=2, sd_dist=pm.HalfNormal.dist(1.0), compute_corr=True)
        pm.Deterministic("country_corr", corr[0, 1])
        offset = pm.Normal("country_offset", 0.0, 1.0, dims=["ISO", "disaster"])
        log_rate = intercept + X @ beta + (offset @ chol.T)[iso_idx]

        if year_effect:
            sd = pm.HalfNormal("year_effect_sd", sigma=0.3)
            step = pm.Normal("year_effect_step", 0.0, 1.0, shape=len(years) - 1)
            walk = pm.Deterministic("year_effect", pt.concatenate([pt.zeros(1), pt.cumsum(sd * step)]), dims="year")
            log_rate = log_rate + walk[year_idx][:, None]
        else:
            columns = [data[f"{f}__standardized"].to_numpy() for f in reporting] + [data[STEP].to_numpy()]
            gamma = pm.Normal("gamma_reporting", 0.0, 1.0, dims="reporting")
            log_rate = log_rate + (pt.as_tensor(np.column_stack(columns)) @ gamma)[:, None]

        mu = pt.clip(pt.softplus(log_rate), MIN_MU, MAX_MU)
        psi = pm.Beta("psi", 1.0, 1.0, dims="disaster")
        alpha = pm.Gamma("alpha", 2.0, 1.0, dims="disaster")
        pm.ZeroInflatedNegativeBinomial(
            "y", psi=psi, mu=mu, alpha=alpha, observed=data[TARGETS].to_numpy(), dims=["obs_id", "disaster"]
        )

    return model


def interval(draws: np.ndarray, mass: float = 0.89) -> tuple[float, float]:
    """Narrowest interval holding ``mass`` of the draws."""
    ordered = np.sort(draws)
    width = int(np.ceil(mass * len(ordered)))
    spans = ordered[width - 1 :] - ordered[: len(ordered) - width + 1]
    start = int(np.argmin(spans))
    return float(ordered[start]), float(ordered[start + width - 1])


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FITS.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    needed = [*NON_CLIMATE[:-1], "population", *INTERNET_MOBILE, "gmst", "ocean_heat", *HAZARDS]
    losses, slopes, models, loos = [], [], [], {}
    for sample, first in (("all", 1900), ("from1990", 1990)):
        base = panel.filter(pl.col("year") >= first)
        rows = base.drop_nulls(needed)  # one row set per sample, so every model compares on the same rows
        lost = base.filter(pl.any_horizontal(pl.col(c).is_null() for c in needed))
        losses.append(
            {
                "sample": sample,
                "rows_published": base.height,
                "rows_used": rows.height,
                "rows_lost": base.height - rows.height,
                "lost_1991_1995": lost.filter(pl.col("year").is_between(1991, 1995)).height,
                **{f"blank_{c}": int(base[c].is_null().sum()) for c in needed},
                **{f"events_{name}": float(rows[JOINT_TARGETS[name]].fill_null(0).sum()) for name in JOINT_CLASSES},
            }
        )
        plan = {
            "4a": dict(temperature="gmst", hazards=[], year_effect=False),
            "4c": dict(temperature="gmst", hazards=HAZARDS, year_effect=False),
            "ye_a": dict(temperature=None, hazards=[], year_effect=True),
            "ye_c": dict(temperature=None, hazards=HAZARDS, year_effect=True),
            "4a_ocean": dict(temperature="ocean_heat", hazards=[], year_effect=False),
        }
        for name, spec in plan.items():  # one fit in memory at a time
            print(f"== {name} {sample}: {rows.height} rows", flush=True)
            model = build_model(rows, **spec)
            idata = sample_or_load(str(FITS / f"{name}_{sample}.nc"), model=model, sample_kwargs=SAMPLE_KWARGS)
            loos[(name, sample)] = az.loo(idata, var_name="y", pointwise=True)
            free = [v.name for v in model.free_RVs]
            diagnostics = az.summary(idata, var_names=free, kind="diagnostics")
            models.append(
                {
                    "model": name,
                    "sample": sample,
                    "rows": rows.height,
                    "elpd": float(loos[(name, sample)].elpd),
                    "pareto_k_gt_0.7": int((loos[(name, sample)].pareto_k > 0.7).sum()),
                    "divergences": int(np.asarray(idata["sample_stats"]["diverging"].values).sum()),
                    "max_r_hat": float(diagnostics["r_hat"].max()),
                }
            )
            posterior = idata["posterior"]
            terms = [spec["temperature"]] if spec["temperature"] else []
            for term in [*terms, *spec["hazards"]]:
                sd = float(rows[term].std())
                for disaster in JOINT_CLASSES:
                    draws = posterior["beta"].sel(feature=term, disaster=disaster).to_numpy().ravel()
                    low, high = interval(draws)
                    slopes.append(
                        {
                            "model": name,
                            "sample": sample,
                            "term": term,
                            "class": disaster,
                            "mean_per_sd": float(draws.mean()),
                            "hdi89_lb": low,
                            "hdi89_ub": high,
                            "sd_of_covariate": sd,
                            "mean_per_unit": float(draws.mean()) / sd,
                        }
                    )
            if not spec["year_effect"]:
                for term in posterior["reporting"].to_numpy():
                    draws = posterior["gamma_reporting"].sel(reporting=term).to_numpy().ravel()
                    low, high = interval(draws)
                    slopes.append(
                        {
                            "model": name,
                            "sample": sample,
                            "term": str(term),
                            "class": "both",
                            "mean_per_sd": float(draws.mean()),
                            "hdi89_lb": low,
                            "hdi89_ub": high,
                            "sd_of_covariate": None,
                            "mean_per_unit": None,
                        }
                    )
            del idata, model, posterior
            gc.collect()
            pl.DataFrame(models).write_csv(RESULTS / "models.csv")
            pl.DataFrame(slopes, infer_schema_length=None).write_csv(RESULTS / "slopes.csv")
            pl.DataFrame(losses).write_csv(RESULTS / "rows.csv")

    comparisons = []
    for sample in ("all", "from1990"):
        for name, base in (("4a", "ye_a"), ("4c", "ye_c"), ("4a_ocean", "ye_a")):
            # Pointwise over country-year x class cells.
            difference = (np.asarray(loos[(name, sample)].elpd_i) - np.asarray(loos[(base, sample)].elpd_i)).ravel()
            comparisons.append(
                {
                    "model": name,
                    "against": base,
                    "sample": sample,
                    "elpd_difference": float(difference.sum()),
                    "dse": float(np.sqrt(difference.size * difference.var(ddof=1))),
                }
            )
    pl.DataFrame(comparisons).write_csv(RESULTS / "loo_vs_year_effect.csv")

    table = pl.DataFrame(slopes, infer_schema_length=None).filter(pl.col("term").is_in(["gmst", "ocean_heat"]))
    verdicts = []
    for (name, sample), group in table.group_by("model", "sample", maintain_order=True):
        by_class = {row["class"]: row for row in group.iter_rows(named=True)}
        hydro, geo = by_class["hydro"], by_class["geophysical"]
        conditions = {
            "geo_includes_0": geo["hdi89_lb"] <= 0 <= geo["hdi89_ub"],
            "geo_under_third_of_hydro": abs(geo["mean_per_sd"]) < abs(hydro["mean_per_sd"]) / 3,
            "hydro_excludes_0": not hydro["hdi89_lb"] <= 0 <= hydro["hdi89_ub"],
        }
        verdicts.append({"model": name, "sample": sample, **conditions, "pass": all(conditions.values())})
    pl.DataFrame(verdicts).write_csv(RESULTS / "verdict.csv")
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(12)
    print(table, pl.DataFrame(verdicts), pl.DataFrame(comparisons), pl.DataFrame(models), sep="\n", flush=True)


if __name__ == "__main__":
    main()
