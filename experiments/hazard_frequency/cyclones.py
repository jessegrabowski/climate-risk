"""
Tropical-cyclone wind exposure in the frequency model (task 2026-11-26, part 2).

Base = the A0 set-up: published covariates without co2, precipitation anomaly and rx5day (within-country z), shared year
effect. ``tc_share_34kt`` (share of population under at least 34-knot winds that year) is added; it is NOT z-scored
within country, only scaled over all rows, so the country intercept keeps the level.

Counts: hydro (all hydrometeorological), flood (Flood + wet mass movement), storm (Storm), and geophysical as placebo.
Each is fitted with and without the cyclone share on the same rows, and compared by PSIS-LOO.

Usage: ``pixi run python experiments/hazard_frequency/cyclones.py``
"""

import gc

from pathlib import Path

import arviz as az
import numpy as np
import polars as pl

from model import build_hazard_model
from run import BASE, CACHE, FITS, PRECIP, SAMPLE_KWARGS, build_panel, interval

from climate_risk.data_functions.emdat_processing import load_emdat_events
from climate_risk.models.frequency import TARGET
from climate_risk.models.identification import GEOPHYSICAL_COLUMN, _prepare, recount_events
from climate_risk.sample import sample_or_load

RESULTS = Path("results/2026-11-26-tc-exposure")
TABLE = CACHE / "processed" / "covariates" / "tc_country_year.parquet"
TC = "tc_share_34kt"
CLASSES = {"flood": ("Flood", "Mass movement (wet)"), "storm": ("Storm",)}


def panel_with_cyclones() -> pl.DataFrame:
    """The A0 rows with flood and storm counts and the cyclone exposure variables."""
    complete = [feature for feature in BASE if feature != "ln_population"]
    panel = build_panel().drop_nulls([*complete, "population", *PRECIP])  # the rows every A0 fit uses
    events = load_emdat_events(CACHE)
    for name, types in CLASSES.items():
        panel = recount_events(panel, events, types, column=f"{name}_disasters")
    table = pl.read_parquet(TABLE).select("ISO", pl.col("year").cast(pl.Int32), TC, "tc_share_64kt", "tc_wind_popmean")
    joined = panel.with_columns(pl.col("date").dt.year().cast(pl.Int32).alias("year")).join(
        table, on=["ISO", "year"], how="left"
    )
    assert joined.height == panel.height
    missing = joined.filter(pl.col(TC).is_null())["ISO"].unique().to_list()
    assert not missing, f"no cyclone value for {missing}"
    return joined


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    panel = panel_with_cyclones()
    assert panel.height == 7710
    exposed = panel.group_by("ISO").agg(pl.col(TC).max())
    description = {
        "rows": panel.height,
        "rows_with_exposure": int((panel[TC] > 0).sum()),
        "countries_ever_exposed": int((exposed[TC] > 0).sum()),
        "countries": exposed.height,
        "mean_share": float(panel[TC].mean()),
        "sd_share": float(panel[TC].std()),
    }
    pl.DataFrame([description]).write_csv(RESULTS / "exposure_description.csv")
    print(description, flush=True)

    targets = {
        "hydro": TARGET,
        "flood": "flood_disasters",
        "storm": "storm_disasters",
        "geophysical": GEOPHYSICAL_COLUMN,
    }
    features = [*BASE, *PRECIP]
    reused = {"hydro": FITS / "H1.nc", "geophysical": FITS / "rob_A6_placebo_A0.nc"}
    rows, loo_rows = [], []
    for name, target in targets.items():
        loos = {}
        for label, columns in (("A0", features), ("A0_tc", [*features, TC])):  # one fit in memory at a time
            path = reused[name] if label == "A0" and name in reused else FITS / f"tc_{name}_{label}.nc"
            model = build_hazard_model(panel, columns, target=target)
            print(f"== {name} {label}", flush=True)
            idata = sample_or_load(str(path), model=model, sample_kwargs=SAMPLE_KWARGS)
            prepared = _prepare(panel, columns, [target])
            observed = idata["observed_data"][target].to_numpy()
            assert np.array_equal(observed, prepared[target].to_numpy()), f"{path} was fitted on other counts"
            loos[label] = az.loo(idata, var_name=target, pointwise=True)
            summary = az.summary(idata, var_names=["beta", "psi", "alpha", "year_effect_sd"], kind="diagnostics")
            for term in ("rx5day_popmean", TC):
                if term not in columns:
                    continue
                draws = idata["posterior"]["beta"].sel(feature=term).to_numpy().ravel()
                low, high = interval(draws)
                rows.append(
                    {
                        "counts": name,
                        "fit": label,
                        "term": term,
                        "mean": float(draws.mean()),
                        "hdi89_lb": low,
                        "hdi89_ub": high,
                        "events": float(prepared[target].sum()),
                        "divergences": int(np.asarray(idata["sample_stats"]["diverging"].values).sum()),
                        "max_r_hat": float(summary["r_hat"].max()),
                    }
                )
            del idata, model
            gc.collect()
        difference = np.asarray(loos["A0_tc"].elpd_i) - np.asarray(loos["A0"].elpd_i)
        loo_rows.append(
            {
                "counts": name,
                "elpd_gain_from_tc": float(difference.sum()),
                "dse": float(np.sqrt(difference.size * difference.var(ddof=1))),
                "pareto_k_gt_0.7": int((loos["A0_tc"].pareto_k > 0.7).sum()),
            }
        )
        pl.DataFrame(rows).write_csv(RESULTS / "coefficients.csv")
        pl.DataFrame(loo_rows).write_csv(RESULTS / "loo.csv")
    pl.Config.set_tbl_cols(12)
    print(pl.DataFrame(rows), pl.DataFrame(loo_rows), sep="\n", flush=True)


if __name__ == "__main__":
    main()
