"""
Test 1 (time_driver) of the CO2 identification task: which common time driver, if any, gives a climate link?

All fits use the year-effect model on the same 7,710 country-years, so their PSIS-LOO (Pareto-smoothed
importance sampling leave-one-out cross-validation) scores can be compared directly.

    1a  year effect + co2                      reused from data/idata/frequency_stage2_yeareffect_long.nc
    1b  year effect, co2 removed
    1c  year effect, co2 replaced by a Hamilton-filter shock to annual global ocean temperature
    1c' year effect, co2 replaced by the existing STL (seasonal-trend decomposition using LOESS) deviation
    1c  repeated with the year effect's step-size prior at 0.1 and 0.6, to see how far the coefficient moves

Usage (from the repo root): ``pixi run python experiments/identification/time_driver.py``
"""

import json

import arviz as az
import polars as pl
import xarray as xr

from common import CACHE, FITS, RESULTS_DIR, SAMPLE_KWARGS, already_done, record, summarise

from climate_risk.data_functions.combine_data import build_time_series
from climate_risk.models.frequency import FEATURES, TARGET
from climate_risk.models.identification import (
    HAMILTON_COLUMN,
    _prepare,
    build_year_effect_model,
    with_hamilton_temperature_shock,
)
from climate_risk.replication_data import create_replication_data
from climate_risk.sample import sample_or_load

TEST = "test1_time_driver"
STL_COLUMN = "dev_from_trend_ocean_temp"
SAVED_1A = CACHE / "idata" / "frequency_stage2_yeareffect_long.nc"


def swap_co2(replacement: str | None) -> list[str]:
    """The published covariate list with co2 dropped (None) or replaced."""
    return [
        replacement if feature == "co2" and replacement else feature
        for feature in FEATURES
        if feature != "co2" or replacement
    ]


def main() -> None:
    panel = create_replication_data(CACHE)
    temperature = build_time_series(CACHE, frequency="annual").select("date", "Temp")
    panel = with_hamilton_temperature_shock(panel, temperature)

    # (label, covariates, climate term, step-size prior)
    runs = [
        ("1b_no_co2", swap_co2(None), None, 0.3),
        ("1c_hamilton_shock", swap_co2(HAMILTON_COLUMN), HAMILTON_COLUMN, 0.3),
        ("1c_hamilton_shock_sd0.1", swap_co2(HAMILTON_COLUMN), HAMILTON_COLUMN, 0.1),
        ("1c_hamilton_shock_sd0.6", swap_co2(HAMILTON_COLUMN), HAMILTON_COLUMN, 0.6),
        ("1c_prime_stl_deviation", swap_co2(STL_COLUMN), STL_COLUMN, 0.3),
    ]

    fits = {}
    done = already_done()

    # 1a: reuse the saved fit, after checking it was made on a panel of the same size.
    saved = xr.open_datatree(SAVED_1A)
    saved.load()
    rows_1a = len(_prepare(panel, FEATURES, [TARGET]))
    assert saved["posterior_predictive"].sizes["obs_id"] == rows_1a, "saved 1a fit is stale: resample it"
    fits["1a_co2"] = saved
    if (TEST, "1a_co2") not in done:
        row = summarise(
            TEST,
            "all years",
            "1a_co2",
            "co2",
            saved,
            rows=rows_1a,
            events=float(_prepare(panel, FEATURES, [TARGET])[TARGET].sum()),
            prior_sd=0.3,
        )
        record(row)
        print(json.dumps(row), flush=True)

    for label, features, term, prior_sd in runs:
        prepared = _prepare(panel, features, [TARGET])
        idata = sample_or_load(
            str(FITS / f"{TEST}_{label}.nc"),
            model=build_year_effect_model(panel, features, year_effect_sd_prior=prior_sd),
            sample_kwargs=SAMPLE_KWARGS,
        )
        fits[label] = idata
        if (TEST, label) in done:
            continue
        row = summarise(
            TEST,
            "all years",
            label,
            term,
            idata,
            rows=len(prepared),
            events=float(prepared[TARGET].sum()),
            prior_sd=prior_sd,
        )
        record(row)
        print(json.dumps(row), flush=True)

    # LOO is comparable only where the rows are the same, so check that before comparing.
    same_rows = {label: len(_prepare(panel, features, [TARGET])) for label, features, _, _ in runs}
    print("rows per fit:", {"1a_co2": rows_1a, **same_rows}, flush=True)
    comparable = {label: fit for label, fit in fits.items() if "_sd0." not in label}
    loos = {label: az.loo(fit, var_name=TARGET) for label, fit in comparable.items()}
    table = az.compare(loos, round_to=2)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "loo_time_driver.csv")
    print(table[["rank", "elpd", "se", "elpd_diff", "dse", "p"]], flush=True)
    print(
        "rows with Pareto k > 0.7:", {label: int((loo.pareto_k > 0.7).sum()) for label, loo in loos.items()}, flush=True
    )


if __name__ == "__main__":
    pl.Config.set_tbl_rows(30)
    main()
