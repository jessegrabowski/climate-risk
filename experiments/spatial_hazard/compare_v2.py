"""
PSIS-LOO comparisons between the spatial fits, record by record.

    M4 against M2                 field and offsets against neither (M2 is the 24 November fit of ``space_time.py``)
    M4 finer field against M4     boundary factor 1.5 against 4
    flood-only finer against M5   the flood records of the joint model against the flood-only fit with a finer field

In each pair both models see the same events, so after checking that the observed records match one by one, the
difference is taken record by record. Only the record terms are scored, not the total event count.

Usage: ``pixi run python experiments/spatial_hazard/compare_v2.py``
"""

from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
import xarray as xr

IDATA = Path("data/idata")
RESULTS = Path("results/2026-11-26-spatial-hazard-v2")
M2 = (IDATA / "spatial_hazard_sea_M2_river_rx5day_year.nc", "record_count")
M4 = (IDATA / "spatial_v2_M4.nc", "record_count")
PAIRS = {
    "M4 against M2": (M4, M2),
    "M4 c=1.5 against M4": ((IDATA / "spatial_v2_M4_c15.nc", "record_count"), M4),
    "flood-only c=1.2 m=100 against M5 floods": (
        (IDATA / "spatial_v2_M5_flood_c12_m100.nc", "record_count"),
        (IDATA / "spatial_v2_M5.nc", "flood_record_count"),
    ),
}


def pointwise_loo(path: Path, variable: str) -> tuple[np.ndarray, az.ELPDData]:
    """The observed record counts of one fit and its pointwise PSIS-LOO on them."""
    observed = xr.load_dataset(path, group="observed_data")[variable].to_numpy()
    likelihood = xr.load_dataset(path, group="log_likelihood")[variable].to_numpy()
    tree = az.from_dict({"log_likelihood": {variable: likelihood}, "posterior": {"unused": likelihood[..., 0]}})
    return observed, az.loo(tree, var_name=variable, pointwise=True)


def main() -> None:
    rows = []
    for label, (first, second) in PAIRS.items():
        observed_first, loo_first = pointwise_loo(*first)
        observed_second, loo_second = pointwise_loo(*second)
        if not np.array_equal(observed_first, observed_second):
            raise ValueError(f"{label}: the two fits do not have the same records in the same order")
        difference = np.asarray(loo_first.elpd_i) - np.asarray(loo_second.elpd_i)
        rows.append(
            {
                "comparison": label,
                "records": difference.size,
                "elpd_first": float(loo_first.elpd),
                "elpd_second": float(loo_second.elpd),
                "elpd_difference": float(difference.sum()),
                "dse": float(np.sqrt(difference.size * difference.var(ddof=1))),
                "pareto_k_gt_0.7_first": int((loo_first.pareto_k > 0.7).sum()),
                "pareto_k_gt_0.7_second": int((loo_second.pareto_k > 0.7).sum()),
            }
        )
    RESULTS.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(RESULTS / "loo_comparisons.csv")


if __name__ == "__main__":
    main()
