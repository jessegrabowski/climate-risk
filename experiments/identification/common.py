"""Shared pieces of the identification experiments: one results table, one way to summarise a fit."""

from pathlib import Path

import arviz as az
import numpy as np
import polars as pl

CACHE = Path("data")
FITS = CACHE / "idata" / "identification"
RESULTS_DIR = Path("results/2026-11-21-identification")
RESULTS = RESULTS_DIR / "fits.csv"

# The sampler settings the year-effect run used (commit e0aa878, scratch/yeareffect_run/placebo_fit.py).
SAMPLE_KWARGS = {"chains": 4, "draws": 2000, "tune": 1000, "random_seed": 1, "target_accept": 0.95}


def shortest_interval(draws: np.ndarray, mass: float = 0.89) -> tuple[float, float]:
    """The narrowest interval holding ``mass`` of the draws (a highest-density interval)."""
    ordered = np.sort(draws)
    width = int(np.ceil(mass * len(ordered)))
    spans = ordered[width - 1 :] - ordered[: len(ordered) - width + 1]
    start = int(np.argmin(spans))

    return float(ordered[start]), float(ordered[start + width - 1])


def summarise(
    test: str,
    sample: str,
    model: str,
    term: str | None,
    idata: object,
    *,
    rows: int,
    events: float,
    prior_sd: float | None = None,
    select: dict[str, str] | None = None,
) -> dict:
    """
    One row of the results table for a fitted model.

    ``term`` names the climate covariate whose coefficient is reported, or None when the model has none.
    ``select`` picks one slice of a two-dimensional ``beta`` (e.g. ``{"disaster": "hydro"}``) for the joint model.
    Those draws are read straight from the posterior: ``az.summary`` labels of a two-dimensional variable disagreed
    with the posterior in the joint fits, so they are not used there.
    """
    summary = az.summary(idata, ci_prob=0.89, ci_kind="hdi", var_names=["~beta_used"])
    stats: dict[str, float] = dict.fromkeys(("mean", "sd", "hdi89_lb", "hdi89_ub"), float("nan"))
    if term and select is not None:
        draws = idata["posterior"]["beta"].sel(feature=term, **select).to_numpy().ravel()
        lower, upper = shortest_interval(draws)
        stats = {"mean": float(draws.mean()), "sd": float(draws.std(ddof=1)), "hdi89_lb": lower, "hdi89_ub": upper}
    elif term:
        stats = {k: float(summary.loc[f"beta[{term}]", k]) for k in stats}

    def pick(column: str) -> float:
        return stats[column]

    return {
        "test": test,
        "sample": sample,
        "model": model,
        "term": term or "",
        "year_effect_sd_prior": prior_sd if prior_sd is not None else float("nan"),
        "rows": rows,
        "events": events,
        "coef_mean": pick("mean"),
        "coef_sd": pick("sd"),
        "coef_hdi89_lb": pick("hdi89_lb"),
        "coef_hdi89_ub": pick("hdi89_ub"),
        "divergences": int(np.asarray(idata.sample_stats.diverging.values).sum()),
        "max_r_hat": float(summary["r_hat"].max()),
    }


def record(row: dict) -> None:
    """Append a row to the results file, creating it with a header the first time."""
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame([row])
    if RESULTS.exists():
        frame = pl.concat([pl.read_csv(RESULTS), frame], how="diagonal_relaxed")
    frame.write_csv(RESULTS)


def already_done() -> set[tuple[str, str]]:
    """The (test, model) pairs already in the results file, so a rerun skips them."""
    return set(pl.read_csv(RESULTS)[["test", "model"]].rows()) if RESULTS.exists() else set()
