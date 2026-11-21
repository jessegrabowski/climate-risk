"""
Test 2 (geophysical_control) of the CO2 identification task: do geophysical events act as a control group?

Geophysical events (earthquakes, volcanic activity, dry mass movements) have no climate pathway but share
EM-DAT's reporting history. If ``co2`` really drives hydrometeorological counts, its slope on those counts
should differ from its slope on geophysical ones, once one year effect absorbs the reporting history they share.

    2a  year-effect model on geophysical counts alone; reuse the earlier placebo fit
    2b  joint model, one shared year effect, co2 slope on hydro only (fixed at 0 for geophysical)
    2c  as 2b, but the geophysical co2 slope is free; reports the hydro minus geophysical difference
    2d  as 2c, but each class has its own year effect (checks the shared-walk assumption by PSIS-LOO)

The assumption behind 2b and 2c: on the log scale, reporting changes move both classes by the same amount.

Usage (from the repo root): ``pixi run python experiments/identification/geophysical_control.py``
"""

import json

import arviz as az
import numpy as np
import xarray as xr

from common import CACHE, FITS, RESULTS_DIR, SAMPLE_KWARGS, already_done, record, shortest_interval, summarise

from climate_risk.data_functions.emdat_processing import load_emdat_events
from climate_risk.models.frequency import FEATURES
from climate_risk.models.identification import (
    GEOPHYSICAL_COLUMN,
    JOINT_CLASSES,
    JOINT_TARGETS,
    _prepare,
    build_joint_year_effect_model,
    with_geophysical_disasters,
)
from climate_risk.replication_data import create_replication_data
from climate_risk.sample import sample_or_load

TEST = "test2_geophysical_control"
SAVED_2A = CACHE / "idata" / "placebo_geophysical_yeareffect.nc"
TARGETS = [JOINT_TARGETS[name] for name in JOINT_CLASSES]


def main() -> None:
    cache = CACHE
    panel = with_geophysical_disasters(create_replication_data(cache), load_emdat_events(cache))
    prepared = _prepare(panel, FEATURES, TARGETS)
    rows = len(prepared)
    done = already_done()

    # 2a: reuse the earlier placebo fit, after checking it was made on a panel of the same size.
    saved = xr.open_datatree(SAVED_2A)
    saved.load()
    assert saved["posterior_predictive"].sizes["obs_id"] == rows, "saved 2a fit is stale: resample it"
    if (TEST, "2a_geophysical_alone") not in done:
        row = summarise(
            TEST,
            "all years",
            "2a_geophysical_alone",
            "co2",
            saved,
            rows=rows,
            events=float(prepared[GEOPHYSICAL_COLUMN].sum()),
            prior_sd=0.3,
        )
        record(row)
        print(json.dumps(row), flush=True)

    variants = {
        "2b_joint_co2_hydro_only": {"co2_on_geophysical": False, "shared_year_effect": True},
        "2c_joint_co2_free": {"co2_on_geophysical": True, "shared_year_effect": True},
        "2d_joint_separate_walks": {"co2_on_geophysical": True, "shared_year_effect": False},
    }
    fits = {}
    for label, options in variants.items():
        idata = sample_or_load(
            str(FITS / f"{TEST}_{label}.nc"),
            model=build_joint_year_effect_model(panel, **options),
            sample_kwargs=SAMPLE_KWARGS,
        )
        fits[label] = idata
        if (TEST, label) in done:
            continue
        hydro = summarise(
            TEST,
            "all years",
            label,
            "co2",
            idata,
            rows=rows,
            events=float(prepared[TARGETS[0]].sum()),
            prior_sd=0.3,
            select={"disaster": "hydro"},
        )
        hydro["model"] = f"{label}: hydro"
        record(hydro)
        print(json.dumps(hydro), flush=True)

        if options["co2_on_geophysical"]:
            beta = idata.posterior["beta"].sel(feature="co2")
            geophysical = summarise(
                TEST,
                "all years",
                label,
                "co2",
                idata,
                rows=rows,
                events=float(prepared[TARGETS[1]].sum()),
                prior_sd=0.3,
                select={"disaster": "geophysical"},
            )
            geophysical["model"] = f"{label}: geophysical"
            record(geophysical)
            print(json.dumps(geophysical), flush=True)

            difference = (beta.sel(disaster="hydro") - beta.sel(disaster="geophysical")).to_numpy().ravel()
            lower, upper = shortest_interval(difference)
            gap = {
                **hydro,
                "model": f"{label}: hydro minus geophysical",
                "coef_mean": float(difference.mean()),
                "coef_sd": float(difference.std(ddof=1)),
                "coef_hdi89_lb": lower,
                "coef_hdi89_ub": upper,
                "events": float("nan"),
            }
            record(gap)
            print(json.dumps(gap), flush=True)

    # LOO compares fits to the same data: the three joint fits all score both classes of the same 7,710 rows.
    loos = {label: az.loo(fit, var_name="y") for label, fit in fits.items()}
    table = az.compare(loos, round_to=2)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "loo_geophysical_control.csv")
    print(table[["rank", "elpd", "se", "elpd_diff", "dse", "p"]], flush=True)
    print(
        "rows with Pareto k > 0.7:", {label: int((loo.pareto_k > 0.7).sum()) for label, loo in loos.items()}, flush=True
    )


if __name__ == "__main__":
    main()
