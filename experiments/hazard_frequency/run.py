"""
Frequency model with local hazard covariates in place of ``co2`` (task 2026-11-24, part 1).

Base is the Test 1 year-effect model without ``co2`` (fit 1b). Hazard covariates come from
``data/processed/covariates/global_country_year.parquet`` (join on ISO + date) and are standardised WITHIN each
country, so the country intercept keeps the level and a slope reads as "a wetter-than-usual year for this country".

    H1  base + precip_anom_std_popmean + rx5day_popmean
    H2  H1 + soilm_mean_popmean                       (rows without ERA5-Land soil moisture drop out)
    H3  H2 + roni_peak with a slope that varies by region (EM-DAT continent)
    H3_post2000  H3 on 2000 onward
    placebo      H3 on geophysical counts (earthquakes, volcanoes, dry mass movements)

Each hazard fit is compared by PSIS-LOO with a base fit on exactly the same rows.
Usage: ``pixi run python experiments/hazard_frequency/run.py [all|summarise]``
"""

import json
import sys

from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
import pymc as pm

from model import build_hazard_model, within_country_z

from climate_risk.data_functions.emdat_processing import load_emdat_events
from climate_risk.models.frequency import FEATURES, TARGET
from climate_risk.models.identification import (
    GEOPHYSICAL_COLUMN,
    _prepare,
    build_year_effect_model,
    with_geophysical_disasters,
)
from climate_risk.replication_data import create_replication_data
from climate_risk.sample import sample_or_load

CACHE = Path("data")
FITS = CACHE / "idata" / "hazard_frequency"
SAVED_1B = CACHE / "idata" / "identification" / "test1_time_driver_1b_no_co2.nc"
RESULTS = Path("results/2026-11-24-hazard-frequency")
COVARIATES = CACHE / "processed" / "covariates" / "global_country_year.parquet"
SAMPLE_KWARGS = {"chains": 4, "draws": 2000, "tune": 1000, "random_seed": 1, "target_accept": 0.95}

BASE = [feature for feature in FEATURES if feature != "co2"]
PRECIP = ["precip_anom_std_popmean", "rx5day_popmean"]
SOIL = ["soilm_mean_popmean"]
RONI = "roni_peak"
HAZARD = [*PRECIP, *SOIL, RONI]


def build_panel() -> pl.DataFrame:
    """Panel + hazard covariates (z-scored within country) + EM-DAT continent + geophysical counts."""
    events = load_emdat_events(CACHE)
    panel = with_geophysical_disasters(create_replication_data(CACHE), events)
    hazard = (
        pl.read_parquet(COVARIATES).with_columns(pl.col("date").dt.date()).select("ISO", "date", *HAZARD)
    )  # the table stores midnight datetimes; the panel stores dates
    regions = events.select("ISO", pl.col("Region").alias("region")).unique(subset="ISO", keep="first")

    return within_country_z(panel.join(hazard, on=["ISO", "date"], how="left"), HAZARD).join(
        regions, on="ISO", how="left"
    )


def run_specs(panel: pl.DataFrame) -> dict[str, dict]:
    """Every fit to make: panel rows, covariates, regional term, target and which base fit it is compared with."""
    complete = panel.drop_nulls(HAZARD)
    late = complete.filter(pl.col("date").dt.year() >= 2000)
    specs: dict[str, dict] = {
        "H1": dict(panel=panel.drop_nulls(PRECIP), features=[*BASE, *PRECIP], base="base_all"),
        "base_all": dict(panel=panel.drop_nulls(PRECIP), features=BASE),
        "base_soil": dict(panel=complete, features=BASE),
        "H2": dict(panel=complete, features=[*BASE, *PRECIP, *SOIL], base="base_soil"),
        "H3": dict(panel=complete, features=[*BASE, *PRECIP, *SOIL], regional=RONI, base="base_soil"),
        "base_post2000": dict(panel=late, features=BASE),
        "H3_post2000": dict(panel=late, features=[*BASE, *PRECIP, *SOIL], regional=RONI, base="base_post2000"),
        "base_placebo": dict(panel=complete, features=BASE, target=GEOPHYSICAL_COLUMN),
        "placebo": dict(
            panel=complete,
            features=[*BASE, *PRECIP, *SOIL],
            regional=RONI,
            target=GEOPHYSICAL_COLUMN,
            base="base_placebo",
        ),
    }
    return specs


def model_of(spec: dict) -> pm.Model:
    return build_hazard_model(
        spec["panel"], spec["features"], regional_feature=spec.get("regional"), target=spec.get("target", TARGET)
    )


def check_matches_reference(panel: pl.DataFrame) -> None:
    """Without a hazard term the new builder must be the 1b model: same log-probability at the same point."""
    reference = build_year_effect_model(panel.drop_nulls(PRECIP), BASE)
    mine = build_hazard_model(panel.drop_nulls(PRECIP), BASE)
    point = reference.initial_point()
    a, b = reference.compile_logp()(point), mine.compile_logp()(point)
    assert np.isclose(a, b), f"builder differs from 1b: {a} vs {b}"
    print(f"builder check: logp {a:.3f} == {b:.3f}", flush=True)


def interval(draws: np.ndarray, mass: float = 0.89) -> tuple[float, float]:
    """Narrowest interval holding ``mass`` of the draws."""
    ordered = np.sort(draws)
    width = int(np.ceil(mass * len(ordered)))
    spans = ordered[width - 1 :] - ordered[: len(ordered) - width + 1]
    start = int(np.argmin(spans))
    return float(ordered[start]), float(ordered[start + width - 1])


def coefficient_rows(name: str, idata, spec: dict, rows: int, events: float) -> list[dict]:
    """One row per hazard slope (and per regional RONI slope) with mean, sd and 89% HDI."""
    posterior = idata["posterior"]
    diverging = int(np.asarray(idata["sample_stats"]["diverging"].values).sum())
    keep = [
        v for v in ("beta", "beta_regional_shared", "regional_sd", "year_effect_sd", "psi", "alpha") if v in posterior
    ]
    max_rhat = float(az.summary(idata, var_names=keep, kind="diagnostics")["r_hat"].max())
    draws: dict[str, np.ndarray] = {}
    for feature in spec["features"]:
        if feature in HAZARD:
            draws[feature] = posterior["beta"].sel(feature=feature).to_numpy().ravel()
    if spec.get("regional"):
        draws["roni_peak (shared mean)"] = posterior["beta_regional_shared"].to_numpy().ravel()
        draws["roni_peak regional sd"] = posterior["regional_sd"].to_numpy().ravel()
        for region in posterior["region"].to_numpy():
            draws[f"roni_peak [{region}]"] = posterior["beta_regional"].sel(region=region).to_numpy().ravel()
    out = []
    for term, values in draws.items():
        low, high = interval(values)
        out.append(
            {
                "fit": name,
                "term": term,
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "hdi89_lb": low,
                "hdi89_ub": high,
                "rows": rows,
                "events": events,
                "divergences": diverging,
                "max_r_hat": max_rhat,
            }
        )
    return out


def main(step: str) -> None:
    panel = build_panel()
    check_matches_reference(panel)
    specs = run_specs(panel)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FITS.mkdir(parents=True, exist_ok=True)

    fits = {}
    for name, spec in specs.items():
        target = spec.get("target", TARGET)
        features = [*spec["features"], *([spec["regional"]] if spec.get("regional") else [])]
        kept = _prepare(spec["panel"], features, [target])
        spec["rows"], spec["events"] = len(kept), float(kept[target].sum())
        path = FITS / f"{name}.nc"
        if name == "base_all" and SAVED_1B.exists():
            path = SAVED_1B  # the identification run's 1b fit, same sampler settings
        if step == "summarise" and not path.exists():
            continue
        print(f"== {name}: {spec['rows']} rows, {spec['events']:.0f} events", flush=True)
        fits[name] = sample_or_load(str(path), model=model_of(spec), sample_kwargs=SAMPLE_KWARGS)
        if name == "base_all":
            assert fits[name]["posterior_predictive"].sizes["obs_id"] == spec["rows"], "saved 1b is on other rows"

    rows: list[dict] = []
    loo_rows: list[dict] = []
    for name, idata in fits.items():
        spec = specs[name]
        if "base" in spec or name.startswith("base"):
            target = spec.get("target", TARGET)
            if "base" in spec:
                rows += coefficient_rows(name, idata, spec, spec["rows"], spec["events"])
                if spec["base"] in fits:
                    base = spec["base"]
                    assert specs[base]["rows"] == spec["rows"], f"{name} and {base} are on different rows"
                    loos = {name: az.loo(idata, var_name=target), base: az.loo(fits[base], var_name=target)}
                    table = az.compare(loos, round_to=2)
                    diff = float(loos[name].elpd - loos[base].elpd)
                    dse = float(table.loc[base, "dse"]) if base in table.index else float("nan")
                    loo_rows.append(
                        {
                            "fit": name,
                            "base": base,
                            "rows": spec["rows"],
                            "elpd_fit": float(loos[name].elpd),
                            "elpd_base": float(loos[base].elpd),
                            "elpd_diff_vs_base": diff,
                            "dse": dse,
                            "pareto_k_gt_0.7": int((loos[name].pareto_k > 0.7).sum()),
                        }
                    )
    pl.DataFrame(rows).write_csv(RESULTS / "coefficients.csv")
    pl.DataFrame(loo_rows).write_csv(RESULTS / "loo_vs_base.csv")
    drops = {
        "rows_base": len(_prepare(panel, BASE, [TARGET])),
        "rows_with_precip": specs["H1"]["rows"],
        "rows_with_soil": specs["H2"]["rows"],
    }
    drops["country_years_dropped_for_soil"] = drops["rows_with_precip"] - drops["rows_with_soil"]
    (RESULTS / "row_counts.json").write_text(json.dumps(drops, indent=2))
    print(pl.DataFrame(rows), pl.DataFrame(loo_rows), drops, sep="\n", flush=True)


if __name__ == "__main__":
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(12)
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
