"""
PSIS-LOO of the spatial model with a field and offsets (M4) against the one without (M2), on the same record terms.

M2 is the fit saved by ``space_time.py`` on 24 November. Both models see the same events, so after checking that the
observed records match one by one, the difference is taken record by record.

Usage: ``pixi run python experiments/spatial_hazard/compare_v2.py``
"""

from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
import xarray as xr

IDATA = Path("data/idata")
RESULTS = Path("results/2026-11-26-spatial-hazard-v2")
FITS = {"M4": IDATA / "spatial_v2_M4.nc", "M2": IDATA / "spatial_hazard_sea_M2_river_rx5day_year.nc"}


def main() -> None:
    losses, observed = {}, {}
    for name, path in FITS.items():
        observed[name] = xr.load_dataset(path, group="observed_data")["record_count"].to_numpy()
        likelihood = xr.load_dataset(path, group="log_likelihood")["record_count"].to_numpy()
        tree = az.from_dict(
            {"log_likelihood": {"record_count": likelihood}, "posterior": {"unused": likelihood[..., 0]}}
        )
        losses[name] = az.loo(tree, var_name="record_count", pointwise=True)
    if not np.array_equal(observed["M4"], observed["M2"]):
        raise ValueError("The two fits do not have the same records in the same order")
    difference = np.asarray(losses["M4"].elpd_i) - np.asarray(losses["M2"].elpd_i)
    table = pl.DataFrame(
        {
            "records": [difference.size],
            "elpd_M4": [float(losses["M4"].elpd)],
            "elpd_M2": [float(losses["M2"].elpd)],
            "elpd_difference": [float(difference.sum())],
            "dse": [float(np.sqrt(difference.size * difference.var(ddof=1)))],
            "pareto_k_gt_0.7_M4": [int((losses["M4"].pareto_k > 0.7).sum())],
            "pareto_k_gt_0.7_M2": [int((losses["M2"].pareto_k > 0.7).sum())],
        }
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.write_csv(RESULTS / "loo_vs_m2.csv")


if __name__ == "__main__":
    main()
