"""
Robustness of the rx5day frequency result (task 2026-11-24b, part A). Base set-up = H1: base covariates + precipitation
anomaly + rx5day, all ERA5 population-weighted and z-scored within country.

    A0  H1 itself                                    (reused fit)
    A1  H1 + a linear trend per country
    A2  rx5day detrended within each country (linear, 1981-2025) before z-scoring
    A3  falsification: next year's rx5day in place of this year's
    A4  data source on the rows CHIRPS covers: ERA5 popmean, ERA5 area mean, CHIRPS area mean
    A5  aggregation: rx5day area mean, rx5day country maximum
    A6  placebo (geophysical counts): A0, A1 and A2 repeated

Each fit is compared by PSIS-LOO with a base fit that has the same rows and every term except rx5day.
Usage: ``pixi run python experiments/hazard_frequency/robustness.py``
"""

import gc

from pathlib import Path

import arviz as az
import numpy as np
import polars as pl

from model import build_hazard_model, within_country_z
from run import BASE, CACHE, COVARIATES, FITS, SAMPLE_KWARGS, interval

from climate_risk.data_functions.emdat_processing import load_emdat_events
from climate_risk.models.frequency import TARGET
from climate_risk.models.identification import GEOPHYSICAL_COLUMN, _prepare, with_geophysical_disasters
from climate_risk.replication_data import create_replication_data
from climate_risk.sample import sample_or_load

RESULTS = Path("results/2026-11-24-rx5day-robustness")
RX = "rx5day_popmean"
PRECIP_ONLY = ["precip_anom_std_popmean"]
EXTRA = ["rx5day_mean", "rx5day_max", "chirps_rx5day_mean"]


def build_panel() -> pl.DataFrame:
    """Panel with the rx5day variants (detrended, lead, area mean, max, CHIRPS) z-scored within country."""
    table = (
        pl.read_parquet(COVARIATES)
        .with_columns(pl.col("date").dt.date())
        .select("ISO", "date", "year", *PRECIP_ONLY, RX, *EXTRA)
        .sort("ISO", "year")
    )
    # Linear trend per country over every year the covariate table has (1981-2025), removed before z-scoring.
    fitted = []
    for _, group in table.group_by("ISO", maintain_order=True):
        values, years = group[RX].to_numpy(), group["year"].to_numpy().astype(float)
        ok = np.isfinite(values)
        if ok.sum() >= 5:
            slope, intercept = np.polyfit(years[ok], values[ok], 1)
            fitted.append(values - (slope * years + intercept))
        else:
            fitted.append(np.full(len(values), np.nan))
    table = table.with_columns(
        pl.Series("rx5day_detrended", np.concatenate(fitted)),
        pl.col(RX).shift(-1).over("ISO").alias("rx5day_lead"),  # next year's value, same country
    )
    columns = [*PRECIP_ONLY, RX, *EXTRA, "rx5day_detrended", "rx5day_lead"]
    events = load_emdat_events(CACHE)
    panel = with_geophysical_disasters(create_replication_data(CACHE), events)
    joined = panel.join(table.drop("year"), on=["ISO", "date"], how="left")

    return within_country_z(joined, columns)


def specs(panel: pl.DataFrame) -> dict[str, dict]:
    """Every fit: panel rows, covariates, trend flag, target, and which base fit it is compared with."""
    chirps = panel.drop_nulls(["chirps_rx5day_mean", "rx5day_mean", RX, *PRECIP_ONLY])
    base = [*BASE, *PRECIP_ONLY]
    out: dict[str, dict] = {}
    # (name, panel, trend, target) for each base; fits then add one rx5day column to ``base``
    # Every fit except the CHIRPS ones uses one row set (no missing value in any rx5day variant), so they compare.
    common = panel.drop_nulls([*PRECIP_ONLY, RX, "rx5day_detrended", "rx5day_lead", "rx5day_mean", "rx5day_max"])
    groups = {
        "": (common, False, TARGET),
        "trend": (common, True, TARGET),
        "chirps": (chirps, False, TARGET),
        "placebo": (common, False, GEOPHYSICAL_COLUMN),
        "placebo_trend": (common, True, GEOPHYSICAL_COLUMN),
    }
    for key, (rows, trend, target) in groups.items():
        base_name = f"base_{key}" if key else "base"
        out[base_name] = dict(panel=rows, features=base, trend=trend, target=target)

        def add(name: str, column: str, key: str = key, rows=rows, trend=trend, target=target, base_name=base_name):
            out[name] = dict(
                panel=rows, features=[*base, column], trend=trend, target=target, base=base_name, term=column
            )

        if key == "":
            add("A0", RX)
            add("A2_detrended", "rx5day_detrended")
            add("A3_lead", "rx5day_lead")
            add("A5_area_mean", "rx5day_mean")
            add("A5_max", "rx5day_max")
        elif key == "trend":
            add("A1_trend", RX)
        elif key == "chirps":
            add("A4_era5_popmean", RX)
            add("A4_era5_area_mean", "rx5day_mean")
            add("A4_chirps_area_mean", "chirps_rx5day_mean")
        elif key == "placebo":
            add("A6_placebo_A0", RX)
            add("A6_placebo_A2_detrended", "rx5day_detrended")
        elif key == "placebo_trend":
            add("A6_placebo_A1_trend", RX)

    return out


def main() -> None:
    panel = build_panel()
    plan = specs(panel)
    FITS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    # One fit in memory at a time (each is about 1 GB): keep only what the table needs, then release it.
    results: dict[str, dict] = {}
    for name, spec in plan.items():
        prepared = _prepare(spec["panel"], spec["features"], [spec["target"]])
        # A0 is H1 when the rows are the same 7,710; otherwise it is refitted.
        path = FITS / "H1.nc" if name == "A0" and len(prepared) == 7710 else FITS / f"rob_{name}.nc"
        model = build_hazard_model(spec["panel"], spec["features"], country_trend=spec["trend"], target=spec["target"])
        print(f"== {name}: {len(prepared)} rows", flush=True)
        idata = sample_or_load(str(path), model=model, sample_kwargs=SAMPLE_KWARGS)
        kept = {"rows": len(prepared), "loo": az.loo(idata, var_name=spec["target"])}
        if "base" in spec:
            draws = idata["posterior"]["beta"].sel(feature=spec["term"]).to_numpy().ravel()
            low, high = interval(draws)
            rhat = az.summary(idata, var_names=["beta", "psi", "alpha"], kind="diagnostics")["r_hat"]
            kept |= {
                "slope": float(draws.mean()),
                "hdi89_lb": low,
                "hdi89_ub": high,
                "divergences": int(np.asarray(idata["sample_stats"]["diverging"].values).sum()),
                "max_r_hat": float(rhat.max()),
            }
        results[name] = kept
        del idata, model
        gc.collect()

    rows = []
    for name, spec in plan.items():
        if "base" not in spec:
            continue
        fit, base = results[name], spec["base"]
        assert fit["rows"] == results[base]["rows"], f"{name} and {base} differ in rows"
        loo, loo_base = fit["loo"], results[base]["loo"]
        table = az.compare({name: loo, base: loo_base}, round_to=3)
        rows.append(
            {
                "fit": name,
                "target": "geophysical" if spec["target"] == GEOPHYSICAL_COLUMN else "hydro",
                "rx5day variable": spec["term"],
                "country trends": spec["trend"],
                "rows": fit["rows"],
                "slope": fit["slope"],
                "hdi89_lb": fit["hdi89_lb"],
                "hdi89_ub": fit["hdi89_ub"],
                "elpd_gain_vs_own_base": float(loo.elpd - loo_base.elpd),
                "dse": float(table["dse"].max()),  # the non-reference row carries it, whichever ranks first
                "divergences": fit["divergences"],
                "max_r_hat": fit["max_r_hat"],
            }
        )
    out = pl.DataFrame(rows)
    out.write_csv(RESULTS / "robustness.csv")
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(14)
    print(out.with_columns(pl.col(pl.Float64).round(3)), flush=True)


if __name__ == "__main__":
    main()
